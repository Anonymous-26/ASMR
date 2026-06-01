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

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


logger = logging.getLogger(__name__)

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
from .attr_prob_class import AttributeProbClassInference
from .attr_monitor import AttributeMonitor
from .bg_filter_monitor import (
    BackgroundFilterTensorboardLogger,
    build_background_filter_payload,
    filter_background_filter_payload,
    select_background_filter_images_by_action_diversity,
)

class AMBR(nn.Module):
    
    def __init__(self, cfg, novel_index, base_index, num_classes, feature_dim):
        super().__init__()
        self.attr_cfg = cfg.MODEL.ATTRIBUTE.clone()
        self.attr_enabled = self.attr_cfg.ENABLED
        self.attr_loss_weight = self.attr_cfg.LOSS_WEIGHT
        self.attr_backward_scale = self.attr_cfg.BACKWARD_SCALE
        self.attr_contrastive_weight = self.attr_cfg.CONTRASTIVE_WEIGHT
        self.attr_temperature = self.attr_cfg.TEMPERATURE
        self.attr_cluster_loss_weight = self.attr_cfg.CLUSTER_LOSS_WEIGHT
        self.bg_threshold = self.attr_cfg.BG_THRESHOLD
        self.pseudo_threshold = self.attr_cfg.PSEUDO_THRESHOLD
        self.bg_penalty_weight = self.attr_cfg.BG_SUPPRESSION_WEIGHT
        self.attr_warmup_iters = self.attr_cfg.WARMUP_ITERS
        self.attr_monitor: Optional[AttributeMonitor] = None
        self.novel_class_indices = list(novel_index)
        self.base_index = base_index
        self.num_classes = num_classes
        self.attr_prototypes_ready = False
        self.latest_attribute_state = {}
        self.is_warmuped = False

        if not self.attr_enabled:
            self.hypergraph_reasoner = None
            self.prototype_bank = None
            return None

        self.attr_pool = cfg.MODEL.ATTRIBUTE.POOLED
        self.attr_with_center_norm = cfg.MODEL.ATTRIBUTE.WITH_CENTER_NORM
        self.attribute_head = AttributeEmbeddingHead(
            feature_dim,
            int(self.attr_cfg.HIDDEN_DIM),
            self.attr_cfg.EMBEDDING_DIM,
            self.attr_pool,
            self.attr_with_center_norm,
            self.attr_backward_scale
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
            
        reasoner_flags = { # ablation
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
            
        self.attr_bilinear = None
        self.attr_bilinear_cfg = self.attr_cfg.BILINEAR
        self.attr_bilinear_enabled = bool(self.attr_bilinear_cfg and self.attr_bilinear_cfg.ENABLED)
        if self.attr_bilinear_enabled:
            self.attr_bilinear = AttributeBilinearMatcher(
                dim=self.attr_cfg.EMBEDDING_DIM,
                proj_dim=self.attr_bilinear_cfg.PROJ_DIM,
                normalize=self.attr_bilinear_cfg.NORMALIZE,
            )

        self.attr_cluster_projector = None
        self.attr_cluster_proj_cfg = self.attr_cfg.CLUSTER_PROJECTOR
        self.attr_cluster_proj_enabled = bool(self.attr_cluster_proj_cfg and self.attr_cluster_proj_cfg.ENABLED)
        if self.attr_cluster_proj_enabled:
            self.attr_cluster_projector = AttributeClusterProjector(
                dim=self.attr_cfg.EMBEDDING_DIM,
                hidden_dim=self.attr_cluster_proj_cfg.HIDDEN_DIM,
            )
            self.attr_cluster_proj_norm = self.attr_cluster_proj_cfg.NORMALIZE

        self.register_buffer(
            "attr_class_prototypes",
            torch.zeros(self.num_classes + 1, self.attr_cfg.EMBEDDING_DIM),
        )

        if self.attr_cfg.FREEZE_ALIGN_BILINEAR and self.attr_bilinear is not None:
            for p in self.attr_bilinear.parameters():
                p.requires_grad = False
            print("freeze attr_bilinear parameters")
        if self.attr_cfg.FREEZE_ALIGN_CLUSTER_PROJECTOR and self.attr_cluster_projector is not None:
            for p in self.attr_cluster_projector.parameters():
                p.requires_grad = False
            print("freeze attr_cluster_projector parameters")
            
        self.margin_threshold = self.attr_cfg.MARGIN_THRE
        
        bg_filter_monitor_cfg = self.attr_cfg.BG_FILTER_MONITOR
        self._bg_filter_writer: Optional[SummaryWriter] = None
        self.bg_filter_monitor = None
        self.bg_filter_log_period = 0
        self._bg_filter_last_step = -1
        self._bg_filter_context = None
        self.bg_filter_min_action_types = 2
        if bg_filter_monitor_cfg and bg_filter_monitor_cfg.ENABLED:
            self.bg_filter_min_action_types = bg_filter_monitor_cfg.MIN_ACTION_TYPES
            check_every_iter = bg_filter_monitor_cfg.CHECK_EVERY_ITER 
            self.bg_filter_log_period = 1 if check_every_iter else int(bg_filter_monitor_cfg.LOG_PERIOD)
            writer = self.attr_monitor.writer if self.attr_monitor is not None else None
            if writer is None and (comm.is_main_process() or comm.get_world_size() > 1):
                log_dir = os.path.join(str(cfg.OUTPUT_DIR), str(bg_filter_monitor_cfg.LOG_DIR))
                if comm.get_world_size() > 1:
                    log_dir = os.path.join(log_dir, f"rank{comm.get_rank()}")
                self._bg_filter_writer = SummaryWriter(log_dir=log_dir)
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
    
    def _attribute_forward(self, box_features: torch.Tensor, outputs: FastRCNNOutputs):
        attr_embeddings = self.attribute_head(box_features)
        if self.prototype_bank is None:
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
        bg_mask = gt_classes == self.num_classes
        if not torch.any(bg_mask):
            return None, gt_classes
        gt_classes_before = gt_classes.clone()
        bg_indices = torch.nonzero(bg_mask, as_tuple=False).squeeze(1)
        valid_indices = [
            idx for idx in self.novel_class_indices 
            if idx < class_prototypes.shape[0] and idx != self.num_classes
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
    
    def _attribute_warmup_active(self, is_training, training_iter):
        if self.is_warmuped:
            return True
        if not is_training or self.attr_warmup_iters <= 0:
            return False
        if int(training_iter.iter) > self.attr_warmup_iters:
            self.is_warmuped = True
        return  self.is_warmuped
    
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
