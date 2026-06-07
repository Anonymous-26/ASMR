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
    BoundedSemanticResidualAdapter,
    AttributeClusterProjector,
    AttributeGraphReasoner,
    AttributeHypergraphReasoner,
    AttributeIdentityReasoner,
    AttributeIncidenceReasoner,
    AttributePrototypeBank,
    VisualConditionedAttributeAttention,
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
        self.agr_cfg = self.attr_cfg.AGR
        self.agr_detector_loss_enabled = bool(self.agr_cfg.DETECTOR_LOSS_ENABLED)
        self.agr_replace_detector_loss = bool(self.agr_cfg.REPLACE_DETECTOR_LOSS)
        self.agr_ignore_in_detector = bool(self.agr_cfg.IGNORE_IN_DETECTOR)
        self.agr_pseudo_loss_weight = float(self.agr_cfg.PSEUDO_LOSS_WEIGHT)
        self.agr_pseudo_weight_by_confidence = bool(
            self.agr_cfg.PSEUDO_WEIGHT_BY_CONFIDENCE
        )
        self.agr_normalize_by_all_rois = bool(self.agr_cfg.NORMALIZE_BY_ALL_ROIS)
        self.agr_max_pseudo_per_batch = int(self.agr_cfg.MAX_PSEUDO_PER_BATCH)
        self.agr_visual_quality_enabled = bool(self.agr_cfg.VISUAL_QUALITY_ENABLED)
        self.agr_max_visual_bg_prob = float(self.agr_cfg.MAX_VISUAL_BG_PROB)
        self.agr_min_visual_novel_prob = float(self.agr_cfg.MIN_VISUAL_NOVEL_PROB)
        self.agr_min_visual_semantic_target_prob = float(
            self.agr_cfg.MIN_VISUAL_SEMANTIC_TARGET_PROB
        )
        self.agr_require_visual_semantic_agreement = bool(
            self.agr_cfg.REQUIRE_VISUAL_SEMANTIC_AGREEMENT
        )
        self.agr_soft_reg_enabled = bool(self.agr_cfg.SOFT_REG_ENABLED)
        self.agr_soft_reg_weight = float(self.agr_cfg.SOFT_REG_WEIGHT)
        self.agr_soft_reg_margin = float(self.agr_cfg.SOFT_REG_MARGIN)
        self.agr_soft_reg_weight_by_confidence = bool(
            self.agr_cfg.SOFT_REG_WEIGHT_BY_CONFIDENCE
        )
        self.agr_soft_reg_suppress_hard_labels = bool(
            self.agr_cfg.SOFT_REG_SUPPRESS_HARD_LABELS
        )
        self.subs_cfg = self.attr_cfg.SUBS
        self.subs_enabled = bool(self.subs_cfg.ENABLED)
        self.subs_topk = int(self.subs_cfg.TOPK)
        self.subs_risk_norm = str(self.subs_cfg.RISK_NORM).lower()
        self.subs_suppress_strength = float(self.subs_cfg.SUPPRESS_STRENGTH)
        self.subs_min_bg_weight = float(self.subs_cfg.MIN_BG_WEIGHT)
        self.subs_detach_risk = bool(self.subs_cfg.DETACH_RISK)
        self.subs_apply_stage = str(self.subs_cfg.APPLY_STAGE).lower()
        self.attr_warmup_iters = self.attr_cfg.WARMUP_ITERS
        self.attention_cfg = self.attr_cfg.ATTENTION
        self.attention_enabled = bool(self.attention_cfg.ENABLED)
        self.residual_adapter_cfg = self.attr_cfg.RESIDUAL_ADAPTER
        self.residual_adapter_enabled = bool(self.residual_adapter_cfg.ENABLED)
        self.visual_proto_cfg = self.attr_cfg.VISUAL_PROTOTYPE
        self.visual_proto_enabled = bool(self.visual_proto_cfg.ENABLED)
        self.visual_proto_momentum = float(self.visual_proto_cfg.MOMENTUM)
        self.visual_proto_max_weight = float(self.visual_proto_cfg.MAX_WEIGHT)
        self.visual_proto_count_tau = float(self.visual_proto_cfg.COUNT_TAU)
        self.visual_proto_min_updates = int(self.visual_proto_cfg.MIN_UPDATES)
        self.visual_proto_mix_dynamic = bool(self.visual_proto_cfg.MIX_DYNAMIC)
        self.visual_proto_mix_static = bool(self.visual_proto_cfg.MIX_STATIC)
        self.visual_proto_mix_loss = bool(self.visual_proto_cfg.MIX_LOSS)
        self.visual_proto_mix_inference = bool(self.visual_proto_cfg.MIX_INFERENCE)
        self.visual_proto_context_from_mixed = bool(
            self.visual_proto_cfg.SEMANTIC_CONTEXT_FROM_MIXED
        )
        self.attr_monitor: Optional[AttributeMonitor] = None
        self.novel_class_indices = list(novel_index)
        self.base_index = base_index
        self.num_classes = num_classes
        self.bg_class_index = num_classes
        self.attr_prototypes_ready = False
        self.latest_attribute_state = {}
        self.latest_agr_detector_state = None
        self.latest_agr_soft_loss = None
        self.latest_subs_weights = None
        self.latest_subs_raw_risk = None
        self.latest_subs_rel_risk = None
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
        match_dim = (
            int(self.attr_bilinear_cfg.PROJ_DIM)
            if self.attr_bilinear_enabled and int(self.attr_bilinear_cfg.PROJ_DIM) > 0
            else int(self.attr_cfg.EMBEDDING_DIM)
        )
        self.visual_attribute_attention = None
        if self.attention_enabled:
            self.visual_attribute_attention = VisualConditionedAttributeAttention(
                dim=match_dim,
                hidden_dim=int(self.attention_cfg.HIDDEN_DIM),
                temperature=float(self.attention_cfg.TEMPERATURE),
            )
        self.semantic_residual_adapter = None
        if self.residual_adapter_enabled:
            if self.visual_attribute_attention is None:
                raise ValueError(
                    "RESIDUAL_ADAPTER.ENABLED requires ATTENTION.ENABLED."
                )
            self.semantic_residual_adapter = BoundedSemanticResidualAdapter(
                semantic_dim=match_dim,
                visual_dim=feature_dim,
                hidden_dim=int(self.residual_adapter_cfg.HIDDEN_DIM),
                gate_hidden_dim=int(self.residual_adapter_cfg.GATE_HIDDEN_DIM),
                max_scale=float(self.residual_adapter_cfg.MAX_SCALE),
                detach_gate_inputs=bool(
                    self.residual_adapter_cfg.DETACH_GATE_INPUTS
                ),
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
        self.register_buffer(
            "visual_ema_class_prototypes",
            torch.zeros(self.num_classes + 1, match_dim),
        )
        self.register_buffer(
            "visual_ema_class_counts",
            torch.zeros(self.num_classes + 1),
        )
        self.register_buffer(
            "visual_ema_class_updates",
            torch.zeros(self.num_classes + 1),
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
        self.bg_filter_match_iou = 0.5
        if bg_filter_monitor_cfg and bg_filter_monitor_cfg.ENABLED:
            self.bg_filter_min_action_types = bg_filter_monitor_cfg.MIN_ACTION_TYPES
            self.bg_filter_match_iou = float(bg_filter_monitor_cfg.MATCH_IOU)
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
        if self.visual_attribute_attention is not None:
            params += list(self.visual_attribute_attention.parameters())
        if self.semantic_residual_adapter is not None:
            params += list(self.semantic_residual_adapter.parameters())
        return [p for p in params if p.requires_grad]
    
    def _build_semantic_state(self, box_features: torch.Tensor):
        attr_embeddings = self.attribute_head(box_features)
        if self.prototype_bank is None:
            return None
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
        full_class_prototypes = class_prototypes
        class_prototypes = self._align_prototypes(class_prototypes)
        (
            attr_embeddings,
            cluster_embeddings,
            class_prototypes,
        ) = self._project_attr_and_keys(
            attr_embeddings, cluster_embeddings, class_prototypes
        )
        if self.attr_bilinear is not None:
            full_class_prototypes = self.attr_bilinear.project_key(
                full_class_prototypes
            )
        self._update_prototype_cache(class_prototypes)
        dynamic_prototypes = None
        attention = None
        semantic_context = None
        if self.visual_attribute_attention is not None:
            aligned_incidence = self._align_incidence_to_current_classes(incidence)
            dynamic_prototypes, attention = self.visual_attribute_attention(
                attr_embeddings, cluster_embeddings, aligned_incidence
            )
            semantic_context = torch.matmul(attention, cluster_embeddings)
            self._log_visual_attribute_attention(attention)
        mixed_class_prototypes, mixed_dynamic_prototypes = (
            self._mix_with_visual_prototypes(class_prototypes, dynamic_prototypes)
        )
        state_class_prototypes = (
            mixed_class_prototypes if self.visual_proto_mix_loss else class_prototypes
        )
        state_dynamic_prototypes = (
            mixed_dynamic_prototypes
            if self.visual_proto_mix_loss
            else dynamic_prototypes
        )
        if (
            self.visual_proto_enabled
            and self.visual_proto_context_from_mixed
            and mixed_dynamic_prototypes is not None
        ):
            class_logits = self._compute_class_logits(
                attr_embeddings, mixed_class_prototypes, mixed_dynamic_prototypes
            )[:, : self.num_classes]
            top_classes = class_logits.argmax(dim=1)
            row_indices = torch.arange(
                mixed_dynamic_prototypes.shape[0],
                device=mixed_dynamic_prototypes.device,
            )
            semantic_context = mixed_dynamic_prototypes[row_indices, top_classes]
        return (
            attr_embeddings,
            cluster_embeddings,
            incidence,
            state_class_prototypes,
            state_dynamic_prototypes,
            attention,
            semantic_context,
            mixed_class_prototypes,
            mixed_dynamic_prototypes,
            full_class_prototypes,
        )

    def enhance_visual_features(
        self,
        box_features: torch.Tensor,
        visual_features: torch.Tensor,
        visual_logits: torch.Tensor,
    ):
        semantic_state = self._build_semantic_state(box_features)
        if self.semantic_residual_adapter is None or semantic_state is None:
            return visual_features, semantic_state
        semantic_context = semantic_state[6]
        enhanced, scales, residual, entropy, margin = self.semantic_residual_adapter(
            visual_features, semantic_context, visual_logits
        )
        self._log_semantic_residual(scales, residual, entropy, margin)
        return enhanced, semantic_state

    @staticmethod
    def _log_semantic_residual(
        scales: torch.Tensor,
        residual: torch.Tensor,
        entropy: torch.Tensor,
        margin: torch.Tensor,
    ) -> None:
        try:
            storage = get_event_storage()
        except Exception:
            return
        storage.put_scalar("attr_residual/scale_mean", float(scales.mean().item()))
        storage.put_scalar("attr_residual/scale_min", float(scales.min().item()))
        storage.put_scalar("attr_residual/scale_max", float(scales.max().item()))
        storage.put_scalar(
            "attr_residual/residual_norm_mean",
            float(residual.detach().norm(dim=1).mean().item()),
        )
        storage.put_scalar(
            "attr_residual/visual_entropy_mean", float(entropy.mean().item())
        )
        storage.put_scalar(
            "attr_residual/visual_margin_mean", float(margin.mean().item())
        )

    def _compute_detector_agr_cls_loss(
        self, outputs: FastRCNNOutputs
    ) -> Optional[torch.Tensor]:
        if not self.agr_detector_loss_enabled:
            return None
        state = self.latest_agr_detector_state
        if not state:
            return None
        targets = state.get("targets")
        weights = state.get("weights")
        if targets is None or weights is None:
            return None
        if targets.numel() != outputs.gt_classes.numel():
            return None
        targets = targets.to(device=outputs.pred_class_logits.device, dtype=torch.long)
        weights = weights.to(
            device=outputs.pred_class_logits.device,
            dtype=outputs.pred_class_logits.dtype,
        )
        losses = F.cross_entropy(
            outputs.pred_class_logits, targets, ignore_index=-1, reduction="none"
        )
        valid = (targets != -1) & (weights > 0)
        if not torch.any(valid):
            return outputs.pred_class_logits.new_tensor(0.0)
        weighted = losses * weights
        if self.agr_normalize_by_all_rois:
            denom = float(max(targets.numel(), 1))
        else:
            denom = float(max(valid.sum().item(), 1))
        loss = weighted.sum() / denom
        try:
            storage = get_event_storage()
            storage.put_scalar("agr_detector/loss_cls", float(loss.detach().item()))
            storage.put_scalar(
                "agr_detector/num_ignored", float(state.get("num_ignored", 0))
            )
            storage.put_scalar(
                "agr_detector/num_pseudo", float(state.get("num_pseudo", 0))
            )
            storage.put_scalar(
                "agr_detector/pseudo_weight_mean",
                float(state.get("pseudo_weight_mean", 0.0)),
            )
        except Exception:
            pass
        return loss

    def _valid_subs_novel_indices(
        self, class_prototypes: torch.Tensor
    ) -> List[int]:
        return [
            int(idx)
            for idx in self.novel_class_indices
            if 0 <= int(idx) < class_prototypes.shape[0] - 1
        ]

    def _subs_current_stage(self) -> str:
        if self.num_classes in (15, 60):
            return "base"
        if self.num_classes in (20, 80):
            return "novel"
        return (
            "novel"
            if any(0 <= int(idx) < self.num_classes for idx in self.novel_class_indices)
            else "base"
        )

    def _subs_stage_active(self, class_prototypes: torch.Tensor) -> bool:
        if not self.subs_enabled:
            return False
        stage = self._subs_current_stage()
        if self.subs_apply_stage not in ("all", "base", "novel"):
            raise ValueError(
                "MODEL.ATTRIBUTE.SUBS.APPLY_STAGE must be one of "
                "{'base', 'novel', 'all'}, got {}".format(self.subs_apply_stage)
            )
        if self.subs_apply_stage == "all":
            return True
        return self.subs_apply_stage == stage

    @staticmethod
    def _tensor_percentile(values: torch.Tensor, q: float) -> Optional[torch.Tensor]:
        if values is None or values.numel() == 0:
            return None
        flat = values.reshape(-1).sort().values
        index = int(round((flat.numel() - 1) * float(q)))
        index = max(0, min(index, flat.numel() - 1))
        return flat[index]

    def _compute_subs_bg_weights(
        self,
        attr_embeddings: torch.Tensor,
        gt_classes: Optional[torch.Tensor],
        class_prototypes: torch.Tensor,
        dynamic_prototypes: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        self.latest_subs_weights = None
        self.latest_subs_raw_risk = None
        self.latest_subs_rel_risk = None
        if (
            not self.subs_enabled
            or gt_classes is None
            or attr_embeddings is None
            or class_prototypes is None
        ):
            return None
        if not self._subs_stage_active(class_prototypes):
            return None
        if attr_embeddings.shape[0] != gt_classes.numel():
            return None
        bg_mask = gt_classes == self.bg_class_index
        if not torch.any(bg_mask):
            return None
        valid_novel_indices = self._valid_subs_novel_indices(class_prototypes)
        if not valid_novel_indices:
            self._log_subs_stats(
                None,
                None,
                None,
                bg_mask,
                active=False,
                num_valid_novel=0,
            )
            return None

        if dynamic_prototypes is None:
            class_logits = self._compute_class_logits(
                attr_embeddings, class_prototypes, None
            )
        else:
            class_logits = self._compute_class_logits(
                attr_embeddings, class_prototypes, dynamic_prototypes
            )
        class_logits = class_logits[:, : class_prototypes.shape[0] - 1]
        attr_probs = F.softmax(class_logits, dim=1)
        novel_indices = torch.as_tensor(
            valid_novel_indices, device=attr_embeddings.device, dtype=torch.long
        )
        novel_probs = attr_probs[:, novel_indices]
        topk = min(max(self.subs_topk, 1), novel_probs.shape[1])
        raw_risk = novel_probs.topk(k=topk, dim=1).values.sum(dim=1).clamp(0.0, 1.0)
        bg_raw_risk = raw_risk[bg_mask]
        if self.subs_risk_norm == "none":
            rel_risk = raw_risk
        elif self.subs_risk_norm == "bg_relative":
            bg_mean = bg_raw_risk.mean()
            bg_max = bg_raw_risk.max()
            denom = (bg_max - bg_mean).clamp(min=1e-6)
            rel_risk = ((raw_risk - bg_mean) / denom).clamp(0.0, 1.0)
        else:
            raise ValueError(
                "MODEL.ATTRIBUTE.SUBS.RISK_NORM must be one of "
                "{'bg_relative', 'none'}, got {}".format(self.subs_risk_norm)
            )
        if self.subs_detach_risk:
            risk_for_weight = rel_risk.detach()
        else:
            risk_for_weight = rel_risk

        weights = attr_embeddings.new_ones(gt_classes.shape)
        bg_weights = (1.0 - self.subs_suppress_strength * risk_for_weight).clamp(
            min=self.subs_min_bg_weight, max=1.0
        )
        weights[bg_mask] = bg_weights[bg_mask]
        self.latest_subs_weights = weights
        self.latest_subs_raw_risk = raw_risk.detach()
        self.latest_subs_rel_risk = rel_risk.detach()
        self._log_subs_stats(
            weights.detach(),
            raw_risk.detach(),
            rel_risk.detach(),
            bg_mask,
            active=True,
            num_valid_novel=len(valid_novel_indices),
        )
        return weights

    def _log_subs_stats(
        self,
        weights: Optional[torch.Tensor],
        raw_risk: Optional[torch.Tensor],
        rel_risk: Optional[torch.Tensor],
        bg_mask: torch.Tensor,
        active: bool,
        num_valid_novel: int = 0,
    ) -> None:
        try:
            storage = get_event_storage()
        except Exception:
            return
        storage.put_scalar("subs/active", float(active))
        storage.put_scalar(
            "subs/stage", 0.0 if self._subs_current_stage() == "base" else 1.0
        )
        storage.put_scalar("subs/num_bg", float(bg_mask.sum().item()))
        storage.put_scalar("subs/num_valid_novel", float(num_valid_novel))
        if (
            not active
            or weights is None
            or raw_risk is None
            or rel_risk is None
            or not torch.any(bg_mask)
        ):
            return
        bg_weights = weights[bg_mask]
        bg_raw_risk = raw_risk[bg_mask]
        bg_rel_risk = rel_risk[bg_mask]
        storage.put_scalar("subs/raw_risk_bg_mean", float(bg_raw_risk.mean().item()))
        storage.put_scalar("subs/raw_risk_bg_max", float(bg_raw_risk.max().item()))
        storage.put_scalar("subs/rel_risk_bg_mean", float(bg_rel_risk.mean().item()))
        storage.put_scalar("subs/rel_risk_bg_max", float(bg_rel_risk.max().item()))
        storage.put_scalar("subs/risk_bg_mean", float(bg_raw_risk.mean().item()))
        storage.put_scalar("subs/risk_bg_max", float(bg_raw_risk.max().item()))
        storage.put_scalar("subs/bg_weight_mean", float(bg_weights.mean().item()))
        storage.put_scalar("subs/bg_weight_min", float(bg_weights.min().item()))
        bg_weight_p10 = self._tensor_percentile(bg_weights, 0.10)
        bg_weight_p50 = self._tensor_percentile(bg_weights, 0.50)
        raw_risk_p90 = self._tensor_percentile(bg_raw_risk, 0.90)
        rel_risk_p90 = self._tensor_percentile(bg_rel_risk, 0.90)
        if bg_weight_p10 is not None:
            storage.put_scalar("subs/bg_weight_p10", float(bg_weight_p10.item()))
        if bg_weight_p50 is not None:
            storage.put_scalar("subs/bg_weight_p50", float(bg_weight_p50.item()))
        if raw_risk_p90 is not None:
            storage.put_scalar("subs/raw_risk_bg_p90", float(raw_risk_p90.item()))
        if rel_risk_p90 is not None:
            storage.put_scalar("subs/rel_risk_bg_p90", float(rel_risk_p90.item()))
        suppressed_ratio = (bg_weights < 0.99).float().mean()
        storage.put_scalar("subs/suppressed_bg_ratio", float(suppressed_ratio.item()))
        storage.put_scalar(
            "subs/ratio_bg_downweighted",
            float(suppressed_ratio.item()),
        )

    def _compute_subs_cls_loss(
        self, outputs: FastRCNNOutputs
    ) -> Optional[torch.Tensor]:
        if not self.subs_enabled:
            return None
        weights = self.latest_subs_weights
        if weights is None or weights.numel() != outputs.gt_classes.numel():
            return None
        targets = outputs.gt_classes.to(
            device=outputs.pred_class_logits.device, dtype=torch.long
        )
        weights = weights.to(
            device=outputs.pred_class_logits.device,
            dtype=outputs.pred_class_logits.dtype,
        )
        raw_losses = F.cross_entropy(
            outputs.pred_class_logits, targets, ignore_index=-1, reduction="none"
        )
        valid = targets != -1
        if not torch.any(valid):
            return outputs.pred_class_logits.new_tensor(0.0)
        weights = weights * valid.to(dtype=weights.dtype)
        denom = weights.sum().clamp(min=1.0)
        loss = (raw_losses * weights).sum() / denom
        try:
            storage = get_event_storage()
            storage.put_scalar("subs/loss_cls", float(loss.detach().item()))
            storage.put_scalar("subs/loss_denom", float(denom.detach().item()))
        except Exception:
            pass
        return loss

    def _attribute_forward(
        self,
        box_features: torch.Tensor,
        outputs: FastRCNNOutputs,
        semantic_state=None,
    ):
        self.latest_subs_weights = None
        self.latest_subs_raw_risk = None
        self.latest_subs_rel_risk = None
        if semantic_state is None:
            semantic_state = self._build_semantic_state(box_features)
        if semantic_state is None:
            return {}, outputs.gt_classes
        (
            attr_embeddings,
            cluster_embeddings,
            incidence,
            class_prototypes,
            dynamic_prototypes,
            attention,
            _,
            _,
            _,
            *extra_state,
        ) = semantic_state
        self._log_weakly_supervised_attribute_metrics(
            attr_embeddings,
            cluster_embeddings,
            incidence,
            class_prototypes,
            dynamic_prototypes,
            attention,
            outputs.gt_classes,
        )
        attr_targets = outputs.gt_classes
        if attr_targets is not None:
            attr_targets = attr_targets.clone()
        subs_class_prototypes = class_prototypes
        subs_dynamic_prototypes = dynamic_prototypes
        if self._subs_current_stage() == "base" and extra_state:
            subs_class_prototypes = extra_state[0]
            subs_dynamic_prototypes = None
        self._compute_subs_bg_weights(
            attr_embeddings,
            outputs.gt_classes,
            subs_class_prototypes,
            subs_dynamic_prototypes,
        )
        penalty, attr_targets = self._apply_background_filtering(
            attr_embeddings, attr_targets, class_prototypes, outputs
        )
        losses = self._compute_attribute_losses(
            attr_embeddings, attr_targets, class_prototypes, dynamic_prototypes
        )
        if self.agr_soft_reg_enabled:
            losses["loss_attr_agr_soft"] = (
                self.latest_agr_soft_loss
                if self.latest_agr_soft_loss is not None
                else attr_embeddings.new_tensor(0.0)
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
        self._update_visual_prototypes(attr_embeddings, outputs.gt_classes)
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

    def _align_incidence_to_current_classes(
        self, incidence: torch.Tensor
    ) -> torch.Tensor:
        expected = self.num_classes + 1
        if incidence.shape[1] == expected:
            return incidence
        aligned = incidence.new_zeros((incidence.shape[0], expected))
        aligned[:, : self.num_classes] = incidence[:, self.base_index]
        aligned[:, self.num_classes] = incidence[:, -1]
        return aligned

    @staticmethod
    def _compute_class_logits(
        attr_embeddings: torch.Tensor,
        class_prototypes: torch.Tensor,
        dynamic_prototypes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if dynamic_prototypes is None:
            return torch.matmul(attr_embeddings, class_prototypes.t())
        return torch.einsum("nd,ncd->nc", attr_embeddings, dynamic_prototypes)

    def _visual_prototype_weights(
        self,
        device: torch.device,
        dtype: torch.dtype,
        num_rows: Optional[int] = None,
    ) -> torch.Tensor:
        if not self.visual_proto_enabled:
            count = self.num_classes + 1 if num_rows is None else int(num_rows)
            return torch.zeros(count, device=device, dtype=dtype)
        counts = self.visual_ema_class_counts.to(device=device, dtype=dtype)
        updates = self.visual_ema_class_updates.to(device=device, dtype=dtype)
        if num_rows is not None:
            counts = counts[:num_rows]
            updates = updates[:num_rows]
        weights = counts / (counts + max(self.visual_proto_count_tau, 1e-6))
        weights = weights * max(self.visual_proto_max_weight, 0.0)
        weights = torch.where(
            updates >= float(max(self.visual_proto_min_updates, 0)),
            weights,
            torch.zeros_like(weights),
        )
        if weights.numel() > self.num_classes:
            weights[self.num_classes :] = 0.0
        return weights.clamp(min=0.0, max=1.0)

    def _mix_with_visual_prototypes(
        self,
        class_prototypes: torch.Tensor,
        dynamic_prototypes: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.visual_proto_enabled:
            return class_prototypes, dynamic_prototypes
        visual = self.visual_ema_class_prototypes.to(
            device=class_prototypes.device, dtype=class_prototypes.dtype
        )
        count = min(class_prototypes.shape[0], visual.shape[0])
        weights = self._visual_prototype_weights(
            class_prototypes.device, class_prototypes.dtype, count
        )
        mixed_static = class_prototypes
        if self.visual_proto_mix_static and count > 0:
            mixed_static = class_prototypes.clone()
            static_part = (
                (1.0 - weights[:count, None]) * class_prototypes[:count]
                + weights[:count, None] * visual[:count]
            )
            mixed_static[:count] = F.normalize(static_part, dim=-1)

        mixed_dynamic = dynamic_prototypes
        if (
            self.visual_proto_mix_dynamic
            and dynamic_prototypes is not None
            and count > 0
        ):
            mixed_dynamic = dynamic_prototypes.clone()
            dynamic_part = (
                (1.0 - weights[None, :count, None])
                * dynamic_prototypes[:, :count]
                + weights[None, :count, None] * visual[None, :count]
            )
            mixed_dynamic[:, :count] = F.normalize(dynamic_part, dim=-1)
        return mixed_static, mixed_dynamic

    @torch.no_grad()
    def _update_visual_prototypes(
        self,
        attr_embeddings: torch.Tensor,
        gt_classes: Optional[torch.Tensor],
    ) -> None:
        if not self.visual_proto_enabled or gt_classes is None:
            return
        foreground = (gt_classes >= 0) & (gt_classes < self.num_classes)
        if not torch.any(foreground):
            self._log_visual_prototype_stats()
            return
        embeddings = F.normalize(attr_embeddings.detach()[foreground], dim=-1)
        labels = gt_classes.detach()[foreground]
        class_sums = embeddings.new_zeros(
            (self.num_classes, embeddings.shape[1])
        )
        class_counts = embeddings.new_zeros(self.num_classes)
        class_sums.index_add_(0, labels, embeddings)
        class_counts.index_add_(
            0, labels, embeddings.new_ones(labels.shape[0])
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(class_sums)
            dist.all_reduce(class_counts)
        momentum = min(max(self.visual_proto_momentum, 0.0), 0.9999)
        updated_classes = torch.nonzero(
            class_counts > 0, as_tuple=False
        ).squeeze(1)
        for class_index in updated_classes:
            cls = int(class_index.item())
            class_mean = F.normalize(
                class_sums[cls : cls + 1] / class_counts[cls].clamp(min=1.0),
                dim=-1,
            ).squeeze(0)
            if self.visual_ema_class_updates[cls] <= 0:
                updated = class_mean
            else:
                updated = (
                    momentum * self.visual_ema_class_prototypes[cls]
                    + (1.0 - momentum) * class_mean
                )
            self.visual_ema_class_prototypes[cls] = F.normalize(
                updated, dim=0
            )
            self.visual_ema_class_counts[cls] += float(class_counts[cls].item())
            self.visual_ema_class_updates[cls] += 1.0
        self._log_visual_prototype_stats()

    @torch.no_grad()
    def _log_visual_prototype_stats(self) -> None:
        if not self.visual_proto_enabled:
            return
        try:
            storage = get_event_storage()
        except Exception:
            return
        weights = self._visual_prototype_weights(
            self.visual_ema_class_prototypes.device,
            self.visual_ema_class_prototypes.dtype,
        )
        valid = self.visual_ema_class_updates[: self.num_classes] > 0
        storage.put_scalar(
            "attr_visual_proto/num_updated_classes", float(valid.sum().item())
        )
        if torch.any(valid):
            storage.put_scalar(
                "attr_visual_proto/count_mean",
                float(self.visual_ema_class_counts[: self.num_classes][valid].mean().item()),
            )
            storage.put_scalar(
                "attr_visual_proto/weight_mean",
                float(weights[: self.num_classes][valid].mean().item()),
            )
            storage.put_scalar(
                "attr_visual_proto/weight_max",
                float(weights[: self.num_classes][valid].max().item()),
            )
            if (
                self.attr_class_prototypes.shape[1]
                == self.visual_ema_class_prototypes.shape[1]
            ):
                text = F.normalize(
                    self.attr_class_prototypes[: self.num_classes][valid], dim=-1
                )
                visual = F.normalize(
                    self.visual_ema_class_prototypes[: self.num_classes][valid],
                    dim=-1,
                )
                storage.put_scalar(
                    "attr_visual_proto/text_visual_cosine",
                    float((text * visual).sum(dim=1).mean().item()),
                )

    def _log_visual_attribute_attention(self, attention: torch.Tensor) -> None:
        if attention.numel() == 0:
            return
        try:
            storage = get_event_storage()
        except Exception:
            return
        detached = attention.detach()
        entropy = -(
            detached * detached.clamp(min=1e-8).log()
        ).sum(dim=1) / np.log(max(detached.shape[1], 2))
        storage.put_scalar("attr_attention/entropy_mean", float(entropy.mean().item()))
        storage.put_scalar(
            "attr_attention/top1_response_mean",
            float(detached.max(dim=1).values.mean().item()),
        )
        if detached.shape[0] > 1:
            centered = detached - detached.mean(dim=0, keepdim=True)
            storage.put_scalar(
                "attr_attention/roi_response_std_mean",
                float(centered.square().mean(dim=0).sqrt().mean().item()),
            )

    def _build_attribute_responses(
        self,
        attr_embeddings: torch.Tensor,
        cluster_embeddings: torch.Tensor,
        attention: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention is not None:
            return attention
        attr_embeddings = F.normalize(attr_embeddings, dim=-1)
        cluster_embeddings = F.normalize(cluster_embeddings, dim=-1)
        logits = torch.matmul(attr_embeddings, cluster_embeddings.t())
        return F.softmax(logits / self.attr_temperature, dim=1)

    @torch.no_grad()
    def _attribute_inference(self, box_features: torch.Tensor, semantic_state=None):
        if semantic_state is None:
            semantic_state = self._build_semantic_state(box_features)
        if semantic_state is None:
            return None, None
        (
            attr_embeddings,
            cluster_embeddings,
            _,
            class_prototypes,
            dynamic_prototypes,
            attention,
            _,
            mixed_class_prototypes,
            mixed_dynamic_prototypes,
            *_,
        ) = semantic_state
        if self.visual_proto_enabled and self.visual_proto_mix_inference:
            class_prototypes = mixed_class_prototypes
            dynamic_prototypes = mixed_dynamic_prototypes
        class_logits = self._compute_class_logits(
            attr_embeddings, class_prototypes, dynamic_prototypes
        )
        class_probs = F.softmax(class_logits / self.attr_temperature, dim=1)
        cluster_probs = self._build_attribute_responses(
            attr_embeddings, cluster_embeddings, attention
        )
        return class_probs, cluster_probs

    @torch.no_grad()
    def _log_weakly_supervised_attribute_metrics(
        self,
        attr_embeddings: torch.Tensor,
        cluster_embeddings: torch.Tensor,
        incidence: torch.Tensor,
        class_prototypes: torch.Tensor,
        dynamic_prototypes: Optional[torch.Tensor],
        attention: Optional[torch.Tensor],
        gt_classes: Optional[torch.Tensor],
    ) -> None:
        if gt_classes is None:
            return
        foreground = (gt_classes >= 0) & (gt_classes < self.num_classes)
        if not torch.any(foreground):
            return
        try:
            storage = get_event_storage()
        except Exception:
            return

        responses = self._build_attribute_responses(
            attr_embeddings, cluster_embeddings, attention
        )[foreground]
        labels = gt_classes[foreground]
        aligned_incidence = self._align_incidence_to_current_classes(incidence)
        allowed = aligned_incidence[:, labels].t() > 0
        storage.put_scalar("attr_eval/num_fg_rois", float(labels.numel()))

        num_attributes = responses.shape[1]
        for requested_k in (1, 3, 5):
            k = min(requested_k, num_attributes)
            if k <= 0:
                continue
            topk = responses.topk(k, dim=1).indices
            hits = allowed.gather(1, topk)
            storage.put_scalar(
                "attr_eval/weak_precision_at_{}".format(requested_k),
                float(hits.float().mean().item()),
            )
            storage.put_scalar(
                "attr_eval/weak_any_hit_at_{}".format(requested_k),
                float(hits.any(dim=1).float().mean().item()),
            )

        entropy = -(
            responses * responses.clamp(min=1e-8).log()
        ).sum(dim=1)
        storage.put_scalar(
            "attr_eval/effective_num_attributes",
            float(entropy.exp().mean().item()),
        )

        class_logits = self._compute_class_logits(
            attr_embeddings[foreground],
            class_prototypes,
            dynamic_prototypes[foreground]
            if dynamic_prototypes is not None
            else None,
        )[:, : self.num_classes]
        storage.put_scalar(
            "attr_eval/class_top1_accuracy",
            float((class_logits.argmax(dim=1) == labels).float().mean().item()),
        )
        topk = min(5, class_logits.shape[1])
        storage.put_scalar(
            "attr_eval/class_top5_accuracy",
            float(
                (class_logits.topk(topk, dim=1).indices == labels[:, None])
                .any(dim=1)
                .float()
                .mean()
                .item()
            ),
        )

        if labels.numel() < 2:
            return
        normalized = F.normalize(responses, dim=1)
        similarities = torch.matmul(normalized, normalized.t())
        pair_mask = torch.triu(
            torch.ones_like(similarities, dtype=torch.bool), diagonal=1
        )
        same_class = labels[:, None] == labels[None, :]
        intra_mask = pair_mask & same_class
        inter_mask = pair_mask & (~same_class)
        storage.put_scalar("attr_eval/num_intra_pairs", float(intra_mask.sum().item()))
        storage.put_scalar("attr_eval/num_inter_pairs", float(inter_mask.sum().item()))
        if torch.any(intra_mask):
            intra_similarity = similarities[intra_mask].mean()
            storage.put_scalar(
                "attr_eval/intra_class_similarity", float(intra_similarity.item())
            )
        else:
            intra_similarity = None
        if torch.any(inter_mask):
            inter_similarity = similarities[inter_mask].mean()
            storage.put_scalar(
                "attr_eval/inter_class_similarity", float(inter_similarity.item())
            )
        else:
            inter_similarity = None
        if intra_similarity is not None and inter_similarity is not None:
            storage.put_scalar(
                "attr_eval/class_separation_gap",
                float((intra_similarity - inter_similarity).item()),
            )

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
        dynamic_prototypes: Optional[torch.Tensor] = None,
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
        valid_classes = gt_classes[valid_mask]
        targets = class_prototypes[valid_classes]
        if dynamic_prototypes is not None:
            dynamic_valid = dynamic_prototypes[valid_mask]
            row_indices = torch.arange(dynamic_valid.shape[0], device=preds.device)
            targets = dynamic_valid[row_indices, valid_classes]
        proto_loss = 1.0 - F.cosine_similarity(preds, targets, dim=-1)
        losses["loss_attr_proto"] = proto_loss.mean() * self.attr_loss_weight

        if self.attr_contrastive_weight > 0:
            logits = self._compute_class_logits(
                preds,
                class_prototypes,
                dynamic_prototypes[valid_mask] if dynamic_prototypes is not None else None,
            ) / self.attr_temperature
            contrastive = F.cross_entropy(logits, valid_classes, reduction="mean")
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
        outputs: Optional[FastRCNNOutputs] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        self.latest_agr_detector_state = None
        self.latest_agr_soft_loss = (
            attr_embeddings.new_tensor(0.0)
            if self.agr_soft_reg_enabled
            else None
        )
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
        visual_bg_prob = None
        visual_novel_prob = None
        visual_semantic_target_prob = None
        visual_semantic_agree = None
        visual_quality_mask = torch.ones_like(max_sim, dtype=torch.bool)
        semantic_pseudo_mask = pseudo_mask
        detector_targets = gt_classes.clone()
        if self.agr_replace_detector_loss:
            detector_weights = attr_embeddings.new_ones(gt_classes.shape)
        else:
            detector_weights = attr_embeddings.new_zeros(gt_classes.shape)
        
        if torch.any(suppress_mask):
            suppressed = bg_indices[suppress_mask]
            gt_classes[suppressed] = -1
            if self.agr_replace_detector_loss or self.agr_ignore_in_detector:
                detector_targets[suppressed] = -1
                detector_weights[suppressed] = 0.0
            if self.bg_penalty_weight > 0:
                penalty = (
                    (max_sim[suppress_mask] - self.bg_threshold)
                    .relu()
                    .mean()
                    * self.bg_penalty_weight
                )

        if self.pseudo_threshold > 0:
            semantic_pseudo_mask = (max_sim > self.pseudo_threshold) & is_discriminative
            pseudo_mask = semantic_pseudo_mask.clone()
            if (
                self.agr_visual_quality_enabled
                and outputs is not None
                and outputs.pred_class_logits is not None
                and outputs.pred_class_logits.shape[0] == gt_classes.shape[0]
                and outputs.pred_class_logits.shape[1] > self.num_classes
            ):
                with torch.no_grad():
                    probs = F.softmax(outputs.pred_class_logits.detach(), dim=1)
                    bg_probs_all = probs[:, self.bg_class_index]
                    novel_probs_all = probs[:, novel_indices]
                    visual_novel_prob, visual_novel_local_idx = novel_probs_all.max(
                        dim=1
                    )
                    visual_bg_prob = bg_probs_all[bg_indices]
                    visual_novel_prob = visual_novel_prob[bg_indices]
                    semantic_targets = novel_indices[max_idx]
                    visual_semantic_target_prob = probs[
                        bg_indices, semantic_targets
                    ]
                    visual_top_novel_class = novel_indices[
                        visual_novel_local_idx[bg_indices]
                    ]
                    visual_semantic_agree = visual_top_novel_class == semantic_targets

                    if 0.0 < self.agr_max_visual_bg_prob < 1.0:
                        visual_quality_mask = visual_quality_mask & (
                            visual_bg_prob < self.agr_max_visual_bg_prob
                        )
                    if self.agr_min_visual_novel_prob > 0.0:
                        visual_quality_mask = visual_quality_mask & (
                            visual_novel_prob > self.agr_min_visual_novel_prob
                        )
                    if self.agr_min_visual_semantic_target_prob > 0.0:
                        visual_quality_mask = visual_quality_mask & (
                            visual_semantic_target_prob
                            > self.agr_min_visual_semantic_target_prob
                        )
                    if self.agr_require_visual_semantic_agreement:
                        visual_quality_mask = visual_quality_mask & visual_semantic_agree
                pseudo_mask = pseudo_mask & visual_quality_mask
            else:
                semantic_pseudo_mask = pseudo_mask
            max_pseudo = max(self.agr_max_pseudo_per_batch, 0)
            if max_pseudo > 0 and int(pseudo_mask.sum().item()) > max_pseudo:
                pseudo_scores = max_sim.masked_fill(~pseudo_mask, float("-inf"))
                selected = torch.topk(
                    pseudo_scores, k=max_pseudo, dim=0
                ).indices
                capped_mask = torch.zeros_like(pseudo_mask)
                capped_mask[selected] = True
                pseudo_mask = capped_mask
            if torch.any(pseudo_mask):
                pseudo = bg_indices[pseudo_mask]
                pseudo_targets = novel_indices[max_idx[pseudo_mask]]
                if self.agr_soft_reg_enabled:
                    self.latest_agr_soft_loss = self._compute_agr_soft_regularization(
                        bg_embeddings=bg_embeddings,
                        pseudo_mask=pseudo_mask,
                        pseudo_targets=pseudo_targets,
                        novel_indices=novel_indices,
                        novel_prototypes=novel_prototypes,
                        max_sim=max_sim,
                    )
                    if self.agr_soft_reg_suppress_hard_labels:
                        gt_classes[pseudo] = -1
                else:
                    gt_classes[pseudo] = pseudo_targets
                    detector_targets[pseudo] = pseudo_targets
                    if self.agr_pseudo_weight_by_confidence:
                        pseudo_weights = (
                            (max_sim[pseudo_mask] - self.pseudo_threshold)
                            / max(1.0 - self.pseudo_threshold, 1e-6)
                        ).clamp(min=0.0, max=1.0)
                        detector_weights[pseudo] = (
                            pseudo_weights * self.agr_pseudo_loss_weight
                        )
                    else:
                        detector_weights[pseudo] = self.agr_pseudo_loss_weight

        pseudo_weight_mean = 0.0
        if torch.any(pseudo_mask):
            pseudo_indices = bg_indices[pseudo_mask]
            pseudo_weight_mean = float(
                detector_weights[pseudo_indices].detach().mean().item()
            )
        self.latest_agr_detector_state = {
            "targets": detector_targets.detach(),
            "weights": detector_weights.detach(),
            "num_ignored": int((suppress_mask & (~pseudo_mask)).sum().item()),
            "num_pseudo": int(pseudo_mask.sum().item()),
            "pseudo_weight_mean": pseudo_weight_mean,
        }
        self._log_agr_quality_stats(
            semantic_pseudo_mask=semantic_pseudo_mask,
            pseudo_mask=pseudo_mask,
            visual_quality_mask=visual_quality_mask,
            visual_bg_prob=visual_bg_prob,
            visual_novel_prob=visual_novel_prob,
            visual_semantic_target_prob=visual_semantic_target_prob,
            visual_semantic_agree=visual_semantic_agree,
        )

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

    def _compute_agr_soft_regularization(
        self,
        *,
        bg_embeddings: torch.Tensor,
        pseudo_mask: torch.Tensor,
        pseudo_targets: torch.Tensor,
        novel_indices: torch.Tensor,
        novel_prototypes: torch.Tensor,
        max_sim: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not self.agr_soft_reg_enabled
            or self.agr_soft_reg_weight <= 0
            or not torch.any(pseudo_mask)
        ):
            return bg_embeddings.new_tensor(0.0)
        preds = bg_embeddings[pseudo_mask]
        target_local = (pseudo_targets[:, None] == novel_indices[None, :]).float().argmax(dim=1)
        target_prototypes = novel_prototypes[target_local]
        pos_loss = 1.0 - F.cosine_similarity(preds, target_prototypes, dim=-1)

        all_sims = torch.matmul(preds, novel_prototypes.t())
        row_indices = torch.arange(all_sims.shape[0], device=all_sims.device)
        pos_sims = all_sims[row_indices, target_local]
        neg_sims = all_sims.masked_fill(
            F.one_hot(target_local, num_classes=all_sims.shape[1]).bool(),
            float("-inf"),
        )
        hard_neg = neg_sims.max(dim=1).values
        margin_loss = (
            self.agr_soft_reg_margin + hard_neg - pos_sims
        ).clamp(min=0.0)
        loss = pos_loss + margin_loss
        if self.agr_soft_reg_weight_by_confidence:
            weights = (
                (max_sim[pseudo_mask] - self.pseudo_threshold)
                / max(1.0 - self.pseudo_threshold, 1e-6)
            ).clamp(min=0.0, max=1.0)
            loss = loss * weights
            denom = weights.sum().clamp(min=1.0)
            loss = loss.sum() / denom
        else:
            loss = loss.mean()
        loss = loss * self.agr_soft_reg_weight
        try:
            storage = get_event_storage()
            storage.put_scalar("agr_soft/loss", float(loss.detach().item()))
            storage.put_scalar("agr_soft/num_rois", float(pseudo_mask.sum().item()))
        except Exception:
            pass
        return loss

    def _log_agr_quality_stats(
        self,
        *,
        semantic_pseudo_mask: torch.Tensor,
        pseudo_mask: torch.Tensor,
        visual_quality_mask: torch.Tensor,
        visual_bg_prob: Optional[torch.Tensor],
        visual_novel_prob: Optional[torch.Tensor],
        visual_semantic_target_prob: Optional[torch.Tensor],
        visual_semantic_agree: Optional[torch.Tensor],
    ) -> None:
        try:
            storage = get_event_storage()
            semantic_count = int(semantic_pseudo_mask.sum().item())
            final_count = int(pseudo_mask.sum().item())
            storage.put_scalar(
                "agr_quality/num_semantic_pseudo_raw", float(semantic_count)
            )
            storage.put_scalar("agr_quality/num_pseudo_final", float(final_count))
            storage.put_scalar(
                "agr_quality/num_rejected_by_visual",
                float(max(semantic_count - final_count, 0)),
            )
            if visual_bg_prob is None:
                return
            if semantic_count > 0:
                rejected = semantic_pseudo_mask & (~visual_quality_mask)
                storage.put_scalar(
                    "agr_quality/visual_reject_ratio",
                    float(rejected.sum().item()) / float(max(semantic_count, 1)),
                )
            storage.put_scalar(
                "agr_quality/visual_bg_prob_mean",
                float(visual_bg_prob.detach().mean().item()),
            )
            storage.put_scalar(
                "agr_quality/visual_novel_prob_mean",
                float(visual_novel_prob.detach().mean().item()),
            )
            storage.put_scalar(
                "agr_quality/visual_semantic_target_prob_mean",
                float(visual_semantic_target_prob.detach().mean().item()),
            )
            if visual_semantic_agree is not None:
                storage.put_scalar(
                    "agr_quality/visual_semantic_agreement_ratio",
                    float(visual_semantic_agree.float().mean().item()),
                )
        except Exception:
            pass
    
    def _attribute_warmup_active(self, training_iter):
        if self.is_warmuped:
            return True
        if self.attr_warmup_iters <= 0:
            self.is_warmuped = True
            return True
        if int(training_iter.iter) >= self.attr_warmup_iters:
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
            "gt_classes": [
                inst.gt_classes.detach()
                if inst is not None and inst.has("gt_classes")
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
            gt_classes=context.get("gt_classes"),
            num_preds_per_image=context.get("num_preds_per_image", []),
            bg_class_index=self.bg_class_index,
            background_class_name=str(
                getattr(self.attr_cfg, "BACKGROUND_CLASS", "background")
            ),
            class_names=class_names,
            match_iou=self.bg_filter_match_iou,
            file_names=context.get("file_names"),
        )
        if not self._should_log_bg_filter(step):
            return
        stats_payload = payload
        selected_image_indices = self._get_mixed_action_bg_filter_indices(payload)
        payload = filter_background_filter_payload(payload, selected_image_indices)
        self.bg_filter_monitor.log(
            step=step,
            payload=payload,
            stats_payload=stats_payload,
            images_tensor=context.get("images_tensor"),
            image_sizes=context.get("image_sizes"),
        )
