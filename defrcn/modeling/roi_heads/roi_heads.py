import os
import torch
import logging
import numpy as np
from torch import nn
from typing import Dict, List, Optional, Sequence, Tuple
from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.utils.events import get_event_storage
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.modeling.backbone.resnet import BottleneckBlock, make_stage
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from .box_head import build_box_head
from .fast_rcnn import ROI_HEADS_OUTPUT_REGISTRY, FastRCNNOutputLayers, FastRCNNOutputs
from defrcn.modeling.meta_arch.gdl import decouple_layer

ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)

def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


def select_foreground_proposals(proposals, bg_label):
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


class ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.

    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(ROIHeads, self).__init__()

        # fmt: off
        self.batch_size_per_image     = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.test_score_thresh        = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh          = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img  = cfg.TEST.DETECTIONS_PER_IMAGE
        self.in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes              = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt       = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.feature_strides          = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels         = {k: v.channels for k, v in input_shape.items()}
        self.cls_agnostic_bbox_reg    = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.smooth_l1_beta           = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        # fmt: on

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        # Box2BoxTransform for bounding box regression
        self.box2box_transform = Box2BoxTransform(
            weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS
        )

    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes,
            self.batch_size_per_image,
            self.positive_sample_fraction,
            self.num_classes,
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns `self.batch_size_per_image` random samples from proposals and groundtruth boxes,
        with a fraction of positives that is no larger than `self.positive_sample_fraction.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                   then the ground-truth box is random)
                Other fields such as "gt_classes" that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(
                match_quality_matrix
            )
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # will filter the proposals again (by foreground/background,
                # etc), so we essentially index the data twice.
                for (
                    trg_name,
                    trg_value,
                ) in targets_per_image.get_fields().items():
                    if trg_name.startswith(
                        "gt_"
                    ) and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(
                            trg_name, trg_value[sampled_targets]
                        )
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros(
                        (len(sampled_idxs), 4)
                    )
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append(
                (gt_classes == self.num_classes).sum().item()
            )
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(
        self,
        images,
        features,
        proposals,
        targets=None,
        batched_inputs: Optional[List[Dict[str, torch.Tensor]]] = None,
    ):
        """
        Args:
            images (ImageList):
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`s. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:
                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].

        Returns:
            results (list[Instances]): length `N` list of `Instances`s containing the
                detected instances. Returned during inference only; may be []
                during training.
            losses (dict[str: Tensor]): mapping from a named loss to a tensor
                storing the loss. Used during training only.
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where the heads share the
    cropping and the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        assert len(self.in_features) == 1

        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / self.feature_strides[self.in_features[0]], )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)
        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg, out_channels, self.num_classes, self.cls_agnostic_bbox_reg
        )

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = make_stage(
            BottleneckBlock,
            3,
            first_stride=2,
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        # print('pooler:', x.size())
        x = self.res5(x)
        # print('res5:', x.size())
        return x

    def forward(
        self,
        images,
        features,
        proposals,
        targets=None,
        batched_inputs: Optional[List[Dict[str, torch.Tensor]]] = None,
    ):
        """
        See :class:`ROIHeads.forward`.
        """
        del images

        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            feature_pooled
        )
        del feature_pooled

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )

        if self.training:
            del features
            losses = outputs.losses()
            return [], losses
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class StandardROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(StandardROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg)

    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels,
                height=pooler_resolution,
                width=pooler_resolution,
            ),
        )

        self.cls_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels,
                height=pooler_resolution,
                width=pooler_resolution,
            ),
        )

        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg,
            self.box_head.output_size,
            self.num_classes,
            self.cls_agnostic_bbox_reg,
        )

        self.cls_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg,
            self.box_head.output_size,
            self.num_classes,
            self.cls_agnostic_bbox_reg,
        )

    def forward(
        self,
        images,
        features,
        proposals,
        targets=None,
        batched_inputs: Optional[List[Dict[str, torch.Tensor]]] = None,
    ):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        features_list = [features[f] for f in self.in_features]

        if self.training:
            losses = self._forward_box(features_list, proposals)
            return proposals, losses
        else:
            pred_instances = self._forward_box(features_list, proposals)
            return pred_instances, {}

    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(
            features, [x.proposal_boxes for x in proposals]
        )

        cls_features = self.cls_head(box_features)
        pred_class_logits, _ = self.cls_predictor(
            cls_features
        )

        box_features = self.box_head(box_features)
        _, pred_proposal_deltas = self.box_predictor(
            box_features
        )
        del box_features

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        if self.training:
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            return pred_instances


import copy
import torch.distributed as dist
from torch import distributions
import torch.nn.functional as F
from detectron2.layers import cat
from detectron2.utils import comm
import fvcore.nn.weight_init as weight_init
from torch.utils.tensorboard import SummaryWriter
from .attr_modules import (
    AttributeEmbeddingHead,
    AttributeBilinearMatcher,
    AttributeClusterProjector,
    AttributeGraphReasoner,
    AttributeHypergraphReasoner,
    AttributeIdentityReasoner,
    AttributeIncidenceReasoner,
    AttributePrototypeBank,
)
from .attr_fusion import AttributeFusion
from .attr_prob_class import AttributeProbClassInference
from .attr_monitor import AttributeMonitor
from .bg_filter_monitor import (
    BackgroundFilterTensorboardLogger,
    build_background_filter_payload,
    filter_background_filter_payload,
    select_background_filter_images_by_action_diversity,
)

@ROI_HEADS_REGISTRY.register()
class CommonalityROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where the heads share the
    cropping and the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        assert len(self.in_features) == 1

        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / self.feature_strides[self.in_features[0]], )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)
        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg, out_channels, self.num_classes, self.cls_agnostic_bbox_reg
        )
        
        self.fc_s = nn.Linear(out_channels, out_channels)
        self.fc_l = nn.Linear(out_channels, out_channels)
        
        for layer in [self.fc_s, self.fc_l]:
            weight_init.c2_xavier_fill(layer)
     
        self.cfg = cfg
        self.memory = cfg.MODEL.ROI_HEADS.MEMORY
        self.semantic = cfg.MODEL.ROI_HEADS.SEMANTIC
        self.augmentation = cfg.MODEL.ROI_HEADS.AUGMENTATION
        self.warmup_distill = cfg.MODEL.ROI_HEADS.WARMUP_DISTILL

        # create the queue
        self.queue_len = cfg.MODEL.ROI_HEADS.QUEUE_LEN
        if self.memory:
            self.register_buffer("queue_s", torch.zeros(self.num_classes, self.queue_len, out_channels))
            self.register_buffer("queue_l", torch.zeros(self.num_classes, self.queue_len, out_channels))
            self.register_buffer("queue_ptr", torch.zeros(self.num_classes, dtype=torch.long))
            self.register_buffer("queue_full", torch.zeros(self.num_classes, dtype=torch.long))
        
        
        self.base_index = []
        if self.num_classes in [15, 20]:
            self.novel_index = [15, 16, 17, 18, 19]
            for i in range(self.num_classes):
                if i not in self.novel_index:
                    self.base_index.append(i)
        elif self.num_classes in [60, 80]:
            self.novel_index = [0, 1, 2, 3, 4, 5, 6, 8, 14, 15, 16, 17, 18, 19, 39, 56, 57, 58, 60, 62]
            for i in range(80):
                if i not in self.novel_index:
                    self.base_index.append(i)

        self._init_attribute_modules(cfg, out_channels)

    def _init_attribute_modules(self, cfg, feature_dim):
        self.attr_cfg = cfg.MODEL.ATTRIBUTE.clone()
        self.attr_enabled = self.attr_cfg.ENABLED
        self.attr_loss_weight = self.attr_cfg.LOSS_WEIGHT
        self.attr_backbone_scale = float(
            getattr(self.attr_cfg, "BACKWARD_SCALE", 0.1)
        )
        self.attr_contrastive_weight = self.attr_cfg.CONTRASTIVE_WEIGHT
        self.attr_temperature = max(self.attr_cfg.TEMPERATURE, 1e-6)
        self.attr_cluster_loss_weight = max(
            float(self.attr_cfg.CLUSTER_LOSS_WEIGHT), 0.0
        )
        self.bg_threshold = self.attr_cfg.BG_THRESHOLD
        self.pseudo_threshold = self.attr_cfg.PSEUDO_THRESHOLD
        self.bg_penalty_weight = self.attr_cfg.BG_SUPPRESSION_WEIGHT
        self.attr_warmup_iters = int(
            getattr(self.attr_cfg, "WARMUP_ITERS", 0)
        )
        self.attr_monitor: Optional[AttributeMonitor] = None
        self.novel_class_indices = list(self.novel_index)
        self.bg_class_index = self.num_classes
        self.attr_prototypes_ready = False
        self.latest_attribute_state = {}

        if not self.attr_enabled:
            self.hypergraph_reasoner = None
            self.prototype_bank = None
            return

        self.attr_pool = cfg.MODEL.ATTRIBUTE.POOLED
        self.attr_with_center_norm = cfg.MODEL.ATTRIBUTE.WITH_CENTER_NORM
        self.attribute_head = AttributeEmbeddingHead(
            feature_dim,
            int(self.attr_cfg.HIDDEN_DIM),
            self.attr_cfg.EMBEDDING_DIM,
            self.attr_pool,
            self.attr_with_center_norm,
        )
        class_names = (
            list(self.attr_cfg.CLASS_NAMES)
            if len(self.attr_cfg.CLASS_NAMES)
            else None
        )
        self.prototype_bank = AttributePrototypeBank(
            super_attr_path=self.attr_cfg.SUPER_ATTR_PATH,
            class_names=class_names,
            background_name=self.attr_cfg.BACKGROUND_CLASS,
        )
        monitor_cfg = getattr(self.attr_cfg, "MONITOR", None)
        if monitor_cfg and monitor_cfg.ENABLED and self.attr_enabled:
            log_dir = str(monitor_cfg.LOG_DIR)
            if not log_dir:
                log_dir = "tb_attr"
            log_path = os.path.join(str(cfg.OUTPUT_DIR), log_dir)
            self.attr_monitor = AttributeMonitor(
                log_dir=log_path,
                log_period=int(monitor_cfg.LOG_PERIOD),
                max_gate_log=int(monitor_cfg.MAX_GATE_LOG),
                log_images=bool(monitor_cfg.LOG_IMAGES),
                log_hist=bool(monitor_cfg.LOG_HIST),
            )
        reasoner_flags = {
            "HGNN.ENABLED": bool(self.attr_cfg.HGNN.ENABLED),
            "woHGNN": bool(getattr(self.attr_cfg, "woHGNN", False)),
            "USE_GNN": bool(getattr(self.attr_cfg, "USE_GNN", False)),
            "Incidence_ONLY": bool(getattr(self.attr_cfg, "Incidence_ONLY", False)),
        }
        enabled_reasoners = [name for name, flag in reasoner_flags.items() if flag]
        if len(enabled_reasoners) > 1:
            raise ValueError(
                "Attribute reasoner configs are mutually exclusive, but got: {}".format(
                    ", ".join(enabled_reasoners)
                )
            )
        if self.attr_cfg.HGNN.ENABLED:
            self.hypergraph_reasoner = AttributeHypergraphReasoner(
                feature_dim=self.attr_cfg.EMBEDDING_DIM,
                hidden_dim=self.attr_cfg.HGNN.HIDDEN_DIM,
                num_layers=self.attr_cfg.HGNN.NUM_LAYERS,
                similarity_weight=self.attr_cfg.HGNN.SIMILARITY_WEIGHT,
            )
        elif getattr(self.attr_cfg, "woHGNN", False):
            self.hypergraph_reasoner = AttributeIdentityReasoner()
        elif getattr(self.attr_cfg, "USE_GNN", False):
            self.hypergraph_reasoner = AttributeGraphReasoner(
                feature_dim=self.attr_cfg.EMBEDDING_DIM,
                hidden_dim=self.attr_cfg.HGNN.HIDDEN_DIM,
                num_layers=self.attr_cfg.HGNN.NUM_LAYERS,
            )
        elif getattr(self.attr_cfg, "Incidence_ONLY", False):
            self.hypergraph_reasoner = AttributeIncidenceReasoner(
                num_layers=self.attr_cfg.HGNN.NUM_LAYERS,
            )
        else:
            self.hypergraph_reasoner = None
            
        self.attr_bilinear_cfg = getattr(self.attr_cfg, "BILINEAR", None)
        self.attr_bilinear_enabled = bool(
            self.attr_bilinear_cfg and self.attr_bilinear_cfg.ENABLED
        )
        if self.attr_bilinear_enabled:
            self.attr_bilinear = AttributeBilinearMatcher(
                dim=self.attr_cfg.EMBEDDING_DIM,
                proj_dim=self.attr_bilinear_cfg.PROJ_DIM,
                normalize=self.attr_bilinear_cfg.NORMALIZE,
            )
        else:
            self.attr_bilinear = None

        self.attr_cluster_proj_cfg = getattr(self.attr_cfg, "CLUSTER_PROJECTOR", None)
        self.attr_cluster_proj_enabled = bool(
            self.attr_cluster_proj_cfg and self.attr_cluster_proj_cfg.ENABLED
        )
        if self.attr_cluster_proj_enabled:
            self.attr_cluster_projector = AttributeClusterProjector(
                dim=self.attr_cfg.EMBEDDING_DIM,
                hidden_dim=self.attr_cluster_proj_cfg.HIDDEN_DIM,
            )
            self.attr_cluster_proj_norm = bool(
                getattr(self.attr_cluster_proj_cfg, "NORMALIZE", True)
            )
        else:
            self.attr_cluster_projector = None
            self.attr_cluster_proj_norm = True

        self.attr_prob_class_cfg = getattr(self.attr_cfg, "PROB_CLASS", None)
        self.attr_prob_class_enabled = bool(
            self.attr_prob_class_cfg and self.attr_prob_class_cfg.ENABLED
        )
        if self.attr_prob_class_enabled:
            self.attr_prob_class = AttributeProbClassInference(
                prob_path=self.attr_prob_class_cfg.PATH,
                eps=self.attr_prob_class_cfg.EPS,
                normalize=self.attr_prob_class_cfg.NORMALIZE,
            )
            self.attr_prob_class_eps = float(self.attr_prob_class_cfg.EPS)
        else:
            self.attr_prob_class = None
            self.attr_prob_class_eps = 1e-12
        self.register_buffer(
            "attr_class_prototypes",
            torch.zeros(self.num_classes + 1, self.attr_cfg.EMBEDDING_DIM),
        )
        fusion_cfg = getattr(self.attr_cfg, "FUSION", None)
        if fusion_cfg is not None and fusion_cfg.ENABLED:
            self.attr_fusion = AttributeFusion(
                cfg,
                num_classes=self.num_classes,
                base_indices=self.base_index,
                novel_indices=self.novel_index,
            )
            self.attr_fusion_use_bg = bool(getattr(fusion_cfg, "USE_BG_CONSTRAINT", True))
            self.attr_fusion_bg_thresh = float(getattr(fusion_cfg, "BG_THRESHOLD", 0.5))
            self.attr_fusion_bg_score = float(getattr(fusion_cfg, "BG_SCORE", 0.1))
        else:
            self.attr_fusion = None
            self.attr_fusion_use_bg = False
            self.attr_fusion_bg_thresh = 0.0
            self.attr_fusion_bg_score = 0.0
        if getattr(self.attr_cfg, "FREEZE_ALIGN_BILINEAR", False) and self.attr_bilinear is not None:
            for p in self.attr_bilinear.parameters():
                p.requires_grad = False
            print("freeze attr_bilinear parameters")
        if getattr(self.attr_cfg, "FREEZE_ALIGN_CLUSTER_PROJECTOR", False) and self.attr_cluster_projector is not None:
            for p in self.attr_cluster_projector.parameters():
                p.requires_grad = False
            print("freeze attr_cluster_projector parameters")
        self.margin_threshold = getattr(self.attr_cfg, "MARGIN_THRE", 0.1)
        bg_filter_monitor_cfg = getattr(self.attr_cfg, "BG_FILTER_MONITOR", None)
        self._bg_filter_writer: Optional[SummaryWriter] = None
        self.bg_filter_monitor = None
        self.bg_filter_log_period = 0
        self._bg_filter_last_step = -1
        self._bg_filter_context = None
        self.bg_filter_min_action_types = 2
        if bg_filter_monitor_cfg and bg_filter_monitor_cfg.ENABLED:
            self.bg_filter_min_action_types = max(
                int(getattr(bg_filter_monitor_cfg, "MIN_ACTION_TYPES", 2)),
                1,
            )
            check_every_iter = bool(
                getattr(bg_filter_monitor_cfg, "CHECK_EVERY_ITER", True)
            )
            self.bg_filter_log_period = max(
                1 if check_every_iter else int(bg_filter_monitor_cfg.LOG_PERIOD),
                1,
            )
            writer = self.attr_monitor.writer if self.attr_monitor is not None else None
            if writer is None and (comm.is_main_process() or comm.get_world_size() > 1):
                log_dir = os.path.join(
                    str(cfg.OUTPUT_DIR), str(bg_filter_monitor_cfg.LOG_DIR)
                )
                if comm.get_world_size() > 1:
                    log_dir = os.path.join(log_dir, f"rank{comm.get_rank()}")
                self._bg_filter_writer = SummaryWriter(
                    log_dir=log_dir
                )
                writer = self._bg_filter_writer
            self.bg_filter_monitor = BackgroundFilterTensorboardLogger(
                writer=writer,
                pixel_mean=cfg.MODEL.PIXEL_MEAN,
                pixel_std=cfg.MODEL.PIXEL_STD,
                input_format=cfg.INPUT.FORMAT,
                max_images=int(bg_filter_monitor_cfg.MAX_IMAGES),
                max_rois=int(bg_filter_monitor_cfg.MAX_ROIS),
            )

    def attribute_parameters(self):
        params: List[torch.nn.Parameter] = []
        if hasattr(self, "attribute_head") and self.attribute_head is not None:
            params += list(self.attribute_head.parameters())
        if self.hypergraph_reasoner is not None:
            params += list(self.hypergraph_reasoner.parameters())
        if self.attr_bilinear is not None:
            params += list(self.attr_bilinear.parameters())
        if self.attr_cluster_projector is not None:
            params += list(self.attr_cluster_projector.parameters())
        return [p for p in params if p.requires_grad]

    def _cache_bg_filter_context(
        self,
        images,
        proposals: List[Instances],
        batched_inputs: Optional[List[Dict[str, object]]] = None,
        targets: Optional[List[Instances]] = None,
    ) -> None:
        if images is None or not self.training:
            self._bg_filter_context = None
            return
        image_tensor = getattr(images, "tensor", None)
        image_sizes = getattr(images, "image_sizes", None)
        if image_tensor is None or image_sizes is None:
            self._bg_filter_context = None
            return
        proposal_boxes: List[torch.Tensor] = []
        num_preds_per_image: List[int] = []
        for inst in proposals:
            num_preds_per_image.append(len(inst))
            if inst.has("proposal_boxes"):
                proposal_boxes.append(inst.proposal_boxes.tensor)
        boxes_tensor = torch.cat(proposal_boxes, dim=0) if proposal_boxes else None
        self._bg_filter_context = {
            "images_tensor": image_tensor.detach(),
            "image_sizes": list(image_sizes),
            "proposal_boxes": boxes_tensor.detach() if boxes_tensor is not None else None,
            "gt_boxes": [
                inst.gt_boxes.tensor.detach()
                if inst is not None and inst.has("gt_boxes")
                else None
                for inst in (targets or [])
            ],
            "num_preds_per_image": num_preds_per_image,
            "file_names": [
                str(sample.get("file_name", ""))
                for sample in (batched_inputs or [])
            ],
        }

    def _get_mixed_action_bg_filter_indices(
        self,
        payload: Optional[Dict[str, object]],
    ) -> List[int]:
        return select_background_filter_images_by_action_diversity(
            payload,
            getattr(self, "bg_filter_min_action_types", 2),
        )

    def _should_log_bg_filter(
        self,
        step: Optional[int],
        selected_image_indices: Optional[Sequence[int]] = None,
    ) -> bool:
        if self.bg_filter_monitor is None or step is None:
            return False
        if self.bg_filter_log_period <= 0:
            return False
        if selected_image_indices is not None and not selected_image_indices:
            return False
        if step == self._bg_filter_last_step:
            return False
        if step % self.bg_filter_log_period != 0:
            return False
        self._bg_filter_last_step = step
        return True

    def _log_background_filter_payload(
        self,
        *,
        gt_classes_before: torch.Tensor,
        gt_classes_after: torch.Tensor,
        bg_indices: torch.Tensor,
        candidate_indices: torch.Tensor,
        max_sim: torch.Tensor,
        max_idx: torch.Tensor,
        margin: torch.Tensor,
        suppress_mask: torch.Tensor,
        pseudo_mask: torch.Tensor,
    ) -> None:
        context = self._bg_filter_context
        if self.bg_filter_monitor is None or context is None:
            return
        storage = get_event_storage()
        step = int(storage.iter)
        class_names = (
            list(self.prototype_bank.class_names)
            if self.prototype_bank is not None and self.prototype_bank.class_names
            else [str(i) for i in range(self.num_classes)]
        )
        payload = build_background_filter_payload(
            gt_classes_before=gt_classes_before,
            gt_classes_after=gt_classes_after,
            bg_indices=bg_indices,
            novel_indices=candidate_indices,
            max_sim=max_sim,
            max_idx=max_idx,
            margin=margin,
            suppress_mask=suppress_mask,
            pseudo_mask=pseudo_mask,
            proposal_boxes=context.get("proposal_boxes"),
            gt_boxes=context.get("gt_boxes"),
            num_preds_per_image=context.get("num_preds_per_image", []),
            bg_class_index=self.bg_class_index,
            background_class_name=str(
                getattr(self.attr_cfg, "BACKGROUND_CLASS", "background")
            ),
            class_names=class_names,
            file_names=context.get("file_names"),
        )
        selected_image_indices = self._get_mixed_action_bg_filter_indices(payload)
        if not self._should_log_bg_filter(step, selected_image_indices):
            return
        payload = filter_background_filter_payload(payload, selected_image_indices)
        if not payload:
            return
        self.bg_filter_monitor.log(
            step=step,
            payload=payload,
            images_tensor=context.get("images_tensor"),
            image_sizes=context.get("image_sizes"),
        )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_s, keys_l, gt_class):
        keys_s = keys_s[:self.queue_len]
        keys_l = keys_l[:self.queue_len]
        batch_size = keys_s.shape[0]
        ptr = int(self.queue_ptr[gt_class])
        if ptr + batch_size <= self.queue_len:
            self.queue_s[gt_class, ptr:ptr + batch_size] = keys_s
            self.queue_l[gt_class, ptr:ptr + batch_size] = keys_l
        else:
            self.queue_s[gt_class, ptr:] = keys_s[:self.queue_len - ptr]
            self.queue_s[gt_class, :(ptr + batch_size) % self.queue_len] = keys_s[self.queue_len - ptr:]
            self.queue_l[gt_class, ptr:] = keys_l[:self.queue_len - ptr]
            self.queue_l[gt_class, :(ptr + batch_size) % self.queue_len] = keys_l[self.queue_len - ptr:]
            
        if ptr + batch_size >= self.queue_len:
            self.queue_full[gt_class] = 1
        ptr = (ptr + batch_size) % self.queue_len
        self.queue_ptr[gt_class] = ptr
    
    @torch.no_grad()
    def update_memory(self, features_s, features_l, gt_classes):
        features_s = concat_all_gather(features_s)
        features_l = concat_all_gather(features_l)
        gt_classes = concat_all_gather(gt_classes)
        
        fg_cases = (gt_classes >= 0) & (gt_classes < self.num_classes)
        features_fg_s = features_s[fg_cases]
        features_fg_l = features_l[fg_cases]
        gt_classes_fg = gt_classes[fg_cases]
        
        if len(gt_classes_fg) == 0:
            return
        uniq_c = torch.unique(gt_classes_fg)
            
        for c in uniq_c:
            c = int(c)
            c_index = torch.nonzero(
                gt_classes_fg == c, as_tuple=False
            ).squeeze(1)
            features_c_s = features_fg_s[c_index]
            features_c_l = features_fg_l[c_index]
            self._dequeue_and_enqueue(features_c_s, features_c_l, c)

    @torch.no_grad()
    def predict_prototype(self, feature_pooled_s, feature_pooled_l, gt_classes):
        
        prototypes_s = []
        prototypes_l = []
        for i in range(self.num_classes):
            if self.queue_full[i] or self.queue_ptr[i] == 0:
                prototypes_s.append(self.queue_s[i].mean(dim=0))
                prototypes_l.append(self.queue_l[i].mean(dim=0))
            else:
                prototypes_s.append(self.queue_s[i][:self.queue_ptr[i]].mean(dim=0))
                prototypes_l.append(self.queue_l[i][:self.queue_ptr[i]].mean(dim=0))
                    
        prototypes_s = torch.stack(prototypes_s, dim=0)
        prototypes_l = torch.stack(prototypes_l, dim=0)
        
        predict_cosine_s = F.cosine_similarity(feature_pooled_s[:, None], prototypes_s[None, :], dim=-1)
        predict_classes_s = predict_cosine_s.new_full(predict_cosine_s.size(), -1.0)
        predict_classes_s[:, self.novel_index] = predict_cosine_s[:, self.novel_index]
        
        predict_cosine_l = F.cosine_similarity(feature_pooled_l[:, None], prototypes_l[None, :], dim=-1)
        predict_classes_l = predict_cosine_l.new_full(predict_cosine_l.size(), -1.0)
        predict_classes_l[:, self.novel_index] = predict_cosine_l[:, self.novel_index]
        
        for i in range(len(predict_classes_s)):
            if gt_classes[i] == self.num_classes or gt_classes[i] < 0:
                continue
            predict_classes_s[i, gt_classes[i]] = predict_cosine_s[i, gt_classes[i]]
            predict_classes_l[i, gt_classes[i]] = predict_cosine_l[i, gt_classes[i]]
        
        zeros = torch.zeros((len(predict_classes_s), 1), device=predict_classes_s.device)
        
        predict_classes_s = F.softmax(predict_classes_s*10, dim=-1)
        predict_classes_s = torch.cat([predict_classes_s, zeros], dim=1)
        
        predict_classes_l = F.softmax(predict_classes_l*10, dim=-1)
        predict_classes_l = torch.cat([predict_classes_l, zeros], dim=1)
        
        return predict_classes_s, predict_classes_l
    
    @torch.no_grad()
    def generate_features(self, gt_classes):
        new_features = []
        new_classes = []
        uniq_c = torch.unique(gt_classes)
        kth = 2
        num_samples = 10
        base_features = self.queue_s[self.base_index]

        # base_mean = base_features.mean(dim=1)
        # base_std = base_features.var(dim=0, unbiased=False)
        # base_std = base_std * self.queue_len / (self.queue_len - 1)

        base_mean, base_std = [], []
        for c in self.base_index:
            if self.queue_full[c]:
                c_mean = self.queue_s[c].mean(dim=0)
                c_std = self.queue_s[c].var(dim=0, unbiased=False)
                c_std = c_std * self.queue_len / (self.queue_len - 1)
            else:
                c_mean = self.queue_s[c][:self.queue_ptr[c]].mean(dim=0)
                c_std = self.queue_s[c][:self.queue_ptr[c]].var(dim=0, unbiased=False)
                if self.queue_ptr[c] > 1:
                    c_std = c_std * self.queue_ptr[c] / (self.queue_ptr[c] - 1)
            base_mean.append(c_mean)
            base_std.append(c_std)
        base_mean = torch.stack(base_mean, dim=0)
        base_std = torch.stack(base_std, dim=0)

        for c in self.novel_index:
            if np.random.rand() < 0.7 or (c not in uniq_c):
                continue
            if self.queue_full[c]:
                c_mean = self.queue_s[c].mean(dim=0)
                c_std = self.queue_s[c].var(dim=0, unbiased=False)
                c_std = c_std * self.queue_len / (self.queue_len - 1)
            else:
                c_mean = self.queue_s[c][:self.queue_ptr[c]].mean(dim=0)
                c_std = self.queue_s[c][:self.queue_ptr[c]].var(dim=0, unbiased=False)
                if self.queue_ptr[c] > 1:
                    c_std = c_std * self.queue_ptr[c] / (self.queue_ptr[c] - 1)
            
            dists = torch.norm(c_mean[None,:] - base_mean, p=2, dim=1)
            _, index = dists.sort()
            calibrated_mean = c_mean
            calibrated_std = base_std[index[:kth]].mean(dim=0)
            
            univariate_normal_dists = distributions.normal.Normal(
                calibrated_mean, scale=torch.sqrt(calibrated_std))
            
            feaures_rsample = univariate_normal_dists.rsample(
                (num_samples,))
            classes_rsample = gt_classes.new_full((num_samples, ), c)
            
            new_features.append(feaures_rsample)
            new_classes.append(classes_rsample)
        if len(new_features) == 0:
            return [], []
        else:
            return torch.cat(new_features), torch.cat(new_classes)

    def _attribute_forward(
        self, attr_embeddings: torch.Tensor, outputs: FastRCNNOutputs
    ):
        if attr_embeddings is None or self.prototype_bank is None:
            return {}
        cluster_state = self.prototype_bank.get_cluster_state(attr_embeddings.device)
        cluster_embeddings = cluster_state["embeddings"]
        incidence = cluster_state["incidence"]
        similarity = cluster_state["similarity"]
        
        if self.hypergraph_reasoner is not None:
            cluster_embeddings = self.hypergraph_reasoner(
                cluster_embeddings, incidence, similarity
            )
        cluster_embeddings = self._project_cluster_embeddings(cluster_embeddings)
        class_prototypes = self.prototype_bank.build_class_prototypes(
            cluster_embeddings,
            attr_embeddings.device,
            incidence_override=incidence,
        )
        class_prototypes = self._align_prototypes(class_prototypes)
        (
            attr_embeddings,
            cluster_embeddings,
            class_prototypes,
        ) = self._project_attr_and_keys(
            attr_embeddings, cluster_embeddings, class_prototypes
        )
        self._update_prototype_cache(class_prototypes)
        attr_targets = outputs.gt_classes
        if attr_targets is not None:
            attr_targets = attr_targets.clone()
        penalty, attr_targets = self._apply_background_filtering(
            attr_embeddings, attr_targets, class_prototypes )
        losses = self._compute_attribute_losses(
            attr_embeddings, attr_targets, class_prototypes
        )

        losses["loss_attr_bg"] = (
            penalty if penalty is not None else attr_embeddings.new_tensor(0.0)
        )
        cluster_loss = self._compute_cluster_incidence_loss(
            attr_embeddings,
            attr_targets,
            cluster_embeddings,
            incidence,
        )
        if cluster_loss is not None:
            losses["loss_attr_cluster"] = cluster_loss
        if self.attr_monitor is not None:
            self._log_attribute_monitor(class_prototypes)
        return losses, attr_targets

    def _update_prototype_cache(self, class_prototypes: torch.Tensor):
        if not hasattr(self, "attr_class_prototypes"):
            return
        with torch.no_grad():
            count = min(
                self.attr_class_prototypes.shape[0], class_prototypes.shape[0]
            )
            self.attr_class_prototypes[:count] = class_prototypes[:count].detach()
            if count < self.attr_class_prototypes.shape[0]:
                self.attr_class_prototypes[count:] = 0
        self.attr_prototypes_ready = True

    def _project_cluster_embeddings(
        self, cluster_embeddings: torch.Tensor
    ):
        if self.attr_cluster_projector is None:
            return cluster_embeddings
        projected = self.attr_cluster_projector(cluster_embeddings)
        if self.attr_cluster_proj_norm:
            projected = F.normalize(projected, dim=-1)
        return projected

    def _project_attr_and_keys(
        self,
        attr_embeddings: Optional[torch.Tensor],
        cluster_embeddings: Optional[torch.Tensor],
        class_prototypes: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.attr_bilinear is None:
            return attr_embeddings, cluster_embeddings, class_prototypes
        if attr_embeddings is not None:
            attr_embeddings = self.attr_bilinear.project_query(attr_embeddings)
        if cluster_embeddings is not None:
            cluster_embeddings = self.attr_bilinear.project_key(cluster_embeddings)
        if class_prototypes is not None:
            class_prototypes = self.attr_bilinear.project_key(class_prototypes)
        return attr_embeddings, cluster_embeddings, class_prototypes

    def _align_incidence_to_classes(
        self, incidence: torch.Tensor, num_classes: int
    ) -> Optional[torch.Tensor]:
        if incidence is None or incidence.numel() == 0:
            return None
        if incidence.dim() != 2:
            return None
        if incidence.shape[1] == num_classes:
            return incidence
        if incidence.shape[1] > num_classes:
            return incidence[:, :num_classes]
        pad = incidence.new_zeros((incidence.shape[0], num_classes - incidence.shape[1]))
        return torch.cat([incidence, pad], dim=1)

    def _log_attribute_monitor(
        self,
        class_prototypes: Optional[torch.Tensor],
    ):
        if self.attr_monitor is None:
            return
        try:
            storage = get_event_storage()
            step = int(storage.iter)
        except Exception:
            step = None
        proto = None
        if class_prototypes is not None:
            proto = class_prototypes[: self.num_classes]
        self.attr_monitor.log(step, proto)

    def _align_prototypes(self, class_prototypes: torch.Tensor) -> torch.Tensor:
        expected = self.num_classes + 1
        if class_prototypes.shape[0] == expected:
            return class_prototypes
        aligned = class_prototypes.new_zeros(expected, class_prototypes.shape[1])
        aligned[:self.num_classes] = class_prototypes[self.base_index]
        aligned[self.num_classes] = class_prototypes[-1]
        return aligned

    def _compute_attribute_losses(
        self,
        attr_embeddings: torch.Tensor,
        gt_classes: Optional[torch.Tensor],
        class_prototypes: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        if gt_classes is None:
            return losses
        valid_mask = (gt_classes >= 0) & (
            gt_classes < class_prototypes.shape[0] # 包含背景类
        )

        if not torch.any(valid_mask):
            return losses
        preds = attr_embeddings[valid_mask]
        targets = class_prototypes[gt_classes[valid_mask]]
        proto_loss = 1.0 - F.cosine_similarity(preds, targets, dim=-1)
        losses["loss_attr_proto"] = proto_loss.mean() * self.attr_loss_weight

        if self.attr_contrastive_weight > 0:
            logits = torch.matmul(preds, class_prototypes.t()) / self.attr_temperature
            contrastive = F.cross_entropy(logits, gt_classes[valid_mask], reduction="mean")
            losses["loss_attr_con"] = contrastive * self.attr_contrastive_weight
        return losses

    def _compute_cluster_incidence_loss(
        self,
        attr_embeddings: Optional[torch.Tensor],
        gt_classes: Optional[torch.Tensor],
        cluster_embeddings: torch.Tensor,
        incidence: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if (
            attr_embeddings is None
            or gt_classes is None
            or self.attr_cluster_loss_weight <= 0
        ):
            return None
        if incidence is None or incidence.numel() == 0:
            return None
        device = attr_embeddings.device
        class_targets = self._build_cluster_targets_from_incidence(
            incidence.to(device), device
        )
        if class_targets is None:
            return None
        valid_mask = (gt_classes >= 0) & (gt_classes < class_targets.shape[0])
        if not torch.any(valid_mask):
            return None
        preds = attr_embeddings[valid_mask]
        if preds.numel() == 0:
            return None
        cluster_embeddings = F.normalize(cluster_embeddings.to(device), dim=-1)
        cluster_scores = torch.matmul(preds, cluster_embeddings.t())
        cluster_probs = F.softmax(cluster_scores / self.attr_temperature, dim=1)
        target = class_targets[gt_classes[valid_mask]]
        keep = target.sum(dim=1) > 0
        if not torch.any(keep):
            return None
        pred = cluster_probs[keep]
        target = target[keep]
        loss = F.kl_div(
            (pred + 1e-8).log(),
            target,
            reduction="batchmean",
        )
        return loss * self.attr_cluster_loss_weight 

    def _build_cluster_targets_from_incidence(
        self, incidence: torch.Tensor, device: torch.device
    ) -> Optional[torch.Tensor]:
        num_clusters, num_classes = incidence.shape
        if num_clusters == 0 or num_classes == 0:
            return None
        if self.num_classes == 60:
            aligned_incidence = incidence.new_zeros(num_clusters, self.num_classes + 1)
            aligned_incidence[:, :self.num_classes] = incidence[:, self.base_index]
            aligned_incidence[:, -1] = incidence[:, -1]
        elif self.num_classes == 15:
            aligned_incidence = incidence.new_zeros(num_clusters, self.num_classes + 1)
            aligned_incidence[:, :self.num_classes] = incidence[:, self.base_index]
            aligned_incidence[:, -1] = incidence[:, -1]
        else:
            aligned_incidence = incidence

        matrix = aligned_incidence.to(device=device, dtype=torch.float32).t()  # C x K
        sums = matrix.sum(dim=1, keepdim=True)
        valid = sums.squeeze(1) > 0
        normalized = torch.zeros_like(matrix)
        normalized[valid] = matrix[valid] / sums[valid].clamp(min=1e-12)
        return normalized
    
    def _apply_background_filtering(
        self,
        attr_embeddings: torch.Tensor,
        gt_classes: Optional[torch.Tensor],
        class_prototypes: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.bg_threshold <= 0 or gt_classes is None:
            return None, gt_classes
        if not self.novel_class_indices:
            return None, gt_classes
        bg_mask = gt_classes == self.bg_class_index
        if not torch.any(bg_mask):
            return None, gt_classes
        gt_classes_before = gt_classes.clone()
        bg_indices = torch.nonzero(bg_mask, as_tuple=False).squeeze(1)
        valid_indices = [
            idx for idx in self.novel_class_indices 
            if idx < class_prototypes.shape[0] and idx != self.bg_class_index
        ]
        if not valid_indices:
            return None, gt_classes
        novel_indices = torch.as_tensor(valid_indices, device=attr_embeddings.device)
        novel_prototypes = class_prototypes[novel_indices]

        bg_embeddings = attr_embeddings[bg_mask]
        similarities = torch.matmul(bg_embeddings, novel_prototypes.t())
        topk = min(2, int(similarities.shape[1]))
        top_sims, top_idxs = torch.topk(similarities, k=topk, dim=1)
        max_sim = top_sims[:, 0]
        max_idx = top_idxs[:, 0]
        if topk > 1:
            margin = top_sims[:, 0] - top_sims[:, 1]
        else:
            margin = torch.zeros_like(max_sim)
            is_discriminative = torch.ones_like(max_sim, dtype=torch.bool)
        if topk > 1:
            is_discriminative = margin > getattr(self, "margin_threshold", 0.1)

        penalty = None
        suppress_mask = (max_sim > self.bg_threshold) & is_discriminative
        pseudo_mask = torch.zeros_like(max_sim, dtype=torch.bool)
        
        if torch.any(suppress_mask):
            suppressed = bg_indices[suppress_mask]
            gt_classes[suppressed] = -1
            if self.bg_penalty_weight > 0:
                penalty = (
                    (max_sim[suppress_mask] - self.bg_threshold)
                    .relu()
                    .mean()
                    * self.bg_penalty_weight
                )

        if self.pseudo_threshold > 0:
            pseudo_mask = (max_sim > self.pseudo_threshold) & is_discriminative
            if torch.any(pseudo_mask):
                pseudo = bg_indices[pseudo_mask]
                pseudo_targets = novel_indices[max_idx[pseudo_mask]]
                gt_classes[pseudo] = pseudo_targets

        self._log_background_filter_payload(
            gt_classes_before=gt_classes_before,
            gt_classes_after=gt_classes,
            bg_indices=bg_indices,
            candidate_indices=novel_indices,
            max_sim=max_sim,
            max_idx=max_idx,
            margin=margin,
            suppress_mask=suppress_mask,
            pseudo_mask=pseudo_mask,
        )
        return penalty, gt_classes

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = make_stage(
            BottleneckBlock,
            3,
            first_stride=2,
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        x = self.res5(x)
        return x

    def _build_gt_proposals(self, targets: List[Instances]) -> List[Instances]:
        proposals: List[Instances] = []
        for gt in targets:
            proposal = Instances(gt.image_size)
            proposal.proposal_boxes = gt.gt_boxes
            if gt.has("gt_classes"):
                proposal.gt_classes = gt.gt_classes
            proposals.append(proposal)
        return proposals

    def _build_gt_pred_instances(
        self,
        targets: List[Instances],
        scores: Optional[List[torch.Tensor]] = None,
    ) -> List[Instances]:
        pred_instances: List[Instances] = []
        for idx, gt in enumerate(targets):
            inst = Instances(gt.image_size)
            inst.pred_boxes = gt.gt_boxes
            if gt.has("gt_classes"):
                inst.pred_classes = gt.gt_classes
            if scores is not None and idx < len(scores):
                inst.scores = scores[idx]
            else:
                inst.scores = gt.gt_boxes.tensor.new_ones((len(gt),))
            pred_instances.append(inst)
        return pred_instances

    def _build_gt_pred_instances_from_logits(
        self, targets: List[Instances], outputs: FastRCNNOutputs
    ) -> List[Instances]:
        """
        Build instances with GT boxes but predicted classes/scores from logits.
        This keeps GT boxes while using model classification output.
        """
        logits = outputs.pred_class_logits
        probs = F.softmax(logits, dim=-1)
        logits_per_image = torch.split(logits, outputs.num_preds_per_image, dim=0)
        probs_per_image = torch.split(probs, outputs.num_preds_per_image, dim=0)
        pred_instances: List[Instances] = []
        for logits_i, probs_i, gt in zip(logits_per_image, probs_per_image, targets):
            inst = Instances(gt.image_size)
            inst.pred_boxes = gt.gt_boxes
            if probs_i.numel() == 0:
                inst.pred_classes = gt.gt_boxes.tensor.new_zeros((len(gt),), dtype=torch.int64)
                inst.scores = gt.gt_boxes.tensor.new_zeros((len(gt),))
                inst.pred_class_logits = logits_i
                inst.pred_class_scores = probs_i
                pred_instances.append(inst)
                continue
            class_probs = probs_i[:, : self.num_classes]
            scores, pred_classes = torch.max(class_probs, dim=1)
            inst.pred_classes = pred_classes
            inst.scores = scores
            inst.pred_class_logits = logits_i[:,:-1]
            inst.pred_class_scores = probs_i[:,:-1]
            pred_instances.append(inst)
        return pred_instances

    def _get_gt_scores_from_inference(
        self, outputs: FastRCNNOutputs, targets: List[Instances]
    ) -> List[torch.Tensor]:
        det_instances, det_indices = outputs.inference(
            score_thresh=0.0,
            nms_thresh=1.0,
            topk_per_image=-1,
        )
        gt_scores: List[torch.Tensor] = []
        for det_inst, det_inds, gt in zip(det_instances, det_indices, targets):
            if len(gt) == 0:
                gt_scores.append(gt.gt_boxes.tensor.new_zeros((0,)))
                continue
            if not det_inst.has("pred_class_scores"):
                gt_scores.append(gt.gt_boxes.tensor.new_ones((len(gt),)))
                continue
            score_map = {}
            for row, proposal_idx in enumerate(det_inds.tolist()):
                if proposal_idx not in score_map:
                    score_map[proposal_idx] = det_inst.pred_class_scores[row]
            scores = []
            gt_classes = gt.gt_classes if gt.has("gt_classes") else None
            for gt_idx in range(len(gt)):
                if gt_classes is None:
                    scores.append(det_inst.pred_class_scores.new_tensor(0.0))
                    continue
                cls = int(gt_classes[gt_idx])
                score_vec = score_map.get(gt_idx)
                if score_vec is None or cls < 0 or cls >= score_vec.numel():
                    scores.append(det_inst.pred_class_scores.new_tensor(0.0))
                else:
                    scores.append(score_vec[cls])
            gt_scores.append(torch.stack(scores))
        return gt_scores

    def forward(
        self,
        images,
        features,
        proposals,
        targets=None,
        batched_inputs: Optional[List[Dict[str, object]]] = None,
    ):
        """
        See :class:`ROIHeads.forward`.
        """
        use_gt_boxes = (
            (not self.training)
            and self.attr_enabled
            and getattr(self.attr_cfg, "EVAL_USE_GT_BOXES", False)
            and targets is not None
        )
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        elif use_gt_boxes:
            proposals = self._build_gt_proposals(targets)
        self._cache_bg_filter_context(images, proposals, batched_inputs, targets)

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        
        attr_embeddings = None
        if self.attr_enabled and not self._attribute_warmup_active():
            if self.training:
                if self.attr_pool:
                    attr_input = decouple_layer(feature_pooled, self.attr_backbone_scale)
                else:
                    attr_input = decouple_layer(box_features, self.attr_backbone_scale)
                
                attr_embeddings = self.attribute_head(attr_input)
            else:
                if self.attr_pool:
                    attr_input = feature_pooled
                else:
                    attr_input = box_features
        
        feature_pooled_s = F.relu(self.fc_s(feature_pooled))
        feature_pooled_l = F.relu(self.fc_l(feature_pooled))
        
        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            feature_pooled_s, feature_pooled_l
        )

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )

        attr_losses = {}
        if self.training and self.attr_enabled and not self._attribute_warmup_active():
            attr_losses, attr_targets = self._attribute_forward(attr_embeddings, outputs)
            
        if self.training:          
            
            if self.memory:
                with torch.no_grad():
                    gt_classes = outputs.gt_classes
                    pad_size = self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE \
                        * self.cfg.SOLVER.IMS_PER_BATCH // torch.distributed.get_world_size()
                    if self.cfg.DATASETS.TWO_STREAM:
                        pad_size *= 2
                    feature_pooled_pad_s = feature_pooled_s.new_full((
                        pad_size, feature_pooled_s.size(1)), -1)
                    feature_pooled_pad_s[: feature_pooled_s.size(0)] = feature_pooled_s
                    feature_pooled_pad_l = feature_pooled_l.new_full((
                        pad_size, feature_pooled_l.size(1)), -1)
                    feature_pooled_pad_l[: feature_pooled_l.size(0)] = feature_pooled_l
                    gt_classes_pad = gt_classes.new_full((pad_size,), -1)
                    gt_classes_pad[: gt_classes.size(0)] = gt_classes
                    
                    self.update_memory(feature_pooled_pad_s.detach(), feature_pooled_pad_l.detach(), gt_classes_pad)

            losses = outputs.losses()

            storage = get_event_storage()
            if int(storage.iter) >= self.warmup_distill:

                if self.semantic:
                    gt_classes = outputs.gt_classes
                    bg_class_ind = pred_class_logits.shape[1] - 1
                    true_cases = (gt_classes >= 0) & (gt_classes < bg_class_ind)
                    
                    gt_prototype_classes_s,  gt_prototype_classes_l= self.predict_prototype(feature_pooled_s, feature_pooled_l, gt_classes)
                    gt_prototype_classes_s = gt_prototype_classes_s.detach()
                    gt_prototype_classes_l = gt_prototype_classes_l.detach()
                
                    losses = outputs.losses()
                    loss_kld = F.kl_div(F.log_softmax(pred_class_logits[true_cases], dim=1),
                        gt_prototype_classes_s[true_cases], reduction='batchmean')
                    loss_reg_disitll = outputs.smooth_l1_loss_distill(gt_prototype_classes_l)
                    losses.update({'loss_kld': loss_kld * 0.1, 'loss_reg_disitll': loss_reg_disitll * 0.7})
                    losses['loss_cls'] = losses['loss_cls'] * 1.0
                    losses['loss_box_reg'] = losses['loss_box_reg'] * 0.7
                    
                if self.augmentation:
                    new_features, new_classes = self.generate_features(gt_classes)
                    if len(new_features) == 0:
                        loss_cls_score_aug = feature_pooled_pad_s.new_full((1,), 0).mean()
                    else:
                        pred_class_logits_aug, _ = self.box_predictor(
                            new_features, new_features
                        )
                        loss_cls_score_aug = F.cross_entropy(
                            pred_class_logits_aug, new_classes, reduction="mean"
                        )
                    losses.update({"loss_cls_score_aug": loss_cls_score_aug * 0.1})
                    
            if self.attr_enabled:
                losses.update(attr_losses)
                
            return [], losses
        else:            
            pred_instances, kept_indices = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            return pred_instances, {}

    def _attribute_warmup_active(self) -> bool:
        if not self.training:
            return False
        if self.attr_warmup_iters <= 0:
            return False
        storage = get_event_storage()
        return int(storage.iter) < self.attr_warmup_iters


@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output
