from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class AttributeFusion(nn.Module):
    """
    Fuse detector and attribute signals with a PoE-style log-probability rule.
    Designed to be inference-only and independent from attribute training logic.
    """

    def __init__(
        self,
        cfg,
        num_classes: int,
        base_indices: Sequence[int],
        novel_indices: Sequence[int],
        ppr_module: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        fusion_cfg = cfg.MODEL.ATTRIBUTE.FUSION
        self.enabled = bool(fusion_cfg.ENABLED)
        self.num_classes = int(num_classes)
        valid = range(self.num_classes)
        self.base_indices = {int(i) for i in base_indices if int(i) in valid}
        self.novel_indices = {int(i) for i in novel_indices if int(i) in valid}
        self.beta_base = float(fusion_cfg.BETA_BASE)
        self.beta_novel = float(fusion_cfg.BETA_NOVEL)
        self.gamma_base = float(fusion_cfg.GAMMA_BASE)
        self.gamma_novel = float(fusion_cfg.GAMMA_NOVEL)
        self.delta = float(fusion_cfg.DELTA)
        self.prior = str(fusion_cfg.PRIOR)
        self.use_gating = bool(fusion_cfg.USE_GATING)
        self.cluster_map = str(fusion_cfg.CLUSTER_MAP)
        self.use_su = bool(fusion_cfg.USE_SHARED_UNIQUE)
        self.su_base_shared = float(fusion_cfg.SU_BASE_SHARED)
        self.su_base_unique = float(fusion_cfg.SU_BASE_UNIQUE)
        self.su_novel_shared = float(fusion_cfg.SU_NOVEL_SHARED)
        self.su_novel_unique = float(fusion_cfg.SU_NOVEL_UNIQUE)
        self.eps = float(fusion_cfg.EPS)
        self.mode = str(getattr(fusion_cfg, "MODE", "poe"))
        self.ppr = ppr_module

    @staticmethod
    def _align_mask_columns(mask: torch.Tensor, target_cols: int) -> torch.Tensor:
        if mask.shape[1] == target_cols:
            return mask
        if mask.shape[1] > target_cols:
            return mask[:, :target_cols]
        pad = mask.new_zeros((mask.shape[0], target_cols - mask.shape[1]))
        return torch.cat([mask, pad], dim=1)

    def _class_weight_vectors(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        beta = torch.full((self.num_classes,), self.beta_base, device=device)
        gamma = torch.full((self.num_classes,), self.gamma_base, device=device)
        if self.novel_indices:
            novel = torch.as_tensor(sorted(self.novel_indices), device=device, dtype=torch.long)
            beta[novel] = self.beta_novel
            gamma[novel] = self.gamma_novel
        return beta, gamma

    def _su_weight_matrix(
        self,
        incidence: torch.Tensor,
        shared_mask: Optional[torch.Tensor],
        unique_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        device = incidence.device
        dtype = incidence.dtype
        num_clusters, num_classes = incidence.shape
        w_s = torch.full((num_classes,), self.su_base_shared, device=device, dtype=dtype)
        w_u = torch.full((num_classes,), self.su_base_unique, device=device, dtype=dtype)
        if self.novel_indices:
            novel = torch.as_tensor(sorted(self.novel_indices), device=device, dtype=torch.long)
            novel = novel[(novel >= 0) & (novel < num_classes)]
            w_s[novel] = float(self.su_novel_shared)
            w_u[novel] = float(self.su_novel_unique)
        if num_classes == self.num_classes + 1:
            w_s[-1] = 1.0
            w_u[-1] = 1.0
        weights = torch.zeros((num_clusters, num_classes), device=device, dtype=dtype)
        if shared_mask is not None:
            weights += shared_mask.to(device=device, dtype=dtype) * w_s
        if unique_mask is not None:
            weights += unique_mask.to(device=device, dtype=dtype) * w_u
        fill = (weights == 0) & (incidence > 0)
        if torch.any(fill):
            weights = weights + fill.to(dtype=dtype) * w_s
        return incidence * weights

    def map_cluster_to_class(
        self,
        cluster_probs: torch.Tensor,
        incidence: torch.Tensor,
        shared_mask: Optional[torch.Tensor],
        unique_mask: Optional[torch.Tensor],
        keep_background: bool = False,
    ) -> torch.Tensor:
        if cluster_probs is None or incidence is None:
            return torch.empty(0, device=cluster_probs.device if cluster_probs is not None else None)
        if incidence.numel() == 0:
            return torch.zeros((cluster_probs.shape[0], self.num_classes), device=cluster_probs.device)
        if shared_mask is not None:
            shared_mask = self._align_mask_columns(shared_mask, incidence.shape[1])
        if unique_mask is not None:
            unique_mask = self._align_mask_columns(unique_mask, incidence.shape[1])
        H = incidence
        if self.use_su:
            H = self._su_weight_matrix(incidence, shared_mask, unique_mask)
        if self.cluster_map == "ppr" and self.ppr is not None:
            probs = self.ppr(cluster_probs, H)
        else:
            probs = cluster_probs @ H
            row_sum = probs.sum(dim=1, keepdim=True).clamp(min=self.eps)
            probs = probs / row_sum
        if not keep_background and probs.shape[1] > self.num_classes:
            probs = probs[:, : self.num_classes]
        return probs

    def _align_probs(self, probs: torch.Tensor, expected: int) -> torch.Tensor:
        if probs.shape[1] == expected:
            return probs
        if probs.shape[1] > expected:
            return probs[:, :expected]
        pad = probs.new_zeros((probs.shape[0], expected - probs.shape[1]))
        return torch.cat([probs, pad], dim=1)

    def _prior(self, device: torch.device, num_classes: int) -> Optional[torch.Tensor]:
        if self.delta <= 0:
            return None
        if self.prior == "uniform":
            return torch.full((num_classes,), 1.0 / max(1, num_classes), device=device)
        return None

    def fuse(
        self,
        p_det: torch.Tensor,
        p_attr: torch.Tensor,
        p_cluster: torch.Tensor,
    ) -> torch.Tensor:
        if p_det is None or p_attr is None or p_cluster is None:
            return torch.empty(0, device=p_det.device if p_det is not None else None)
        expected = p_det.shape[1]
        p_attr = self._align_probs(p_attr, expected)
        p_cluster = self._align_probs(p_cluster, expected)
        device = p_det.device
        beta_vec, gamma_vec = self._class_weight_vectors(device)
        # beta = torch.cat([beta_vec, beta_vec.new_zeros(1)], dim=0)
        # gamma = torch.cat([gamma_vec, gamma_vec.new_zeros(1)], dim=0)
        beta = beta_vec
        gamma = gamma_vec
        if self.use_gating:
            g_attr = p_attr.max(dim=1, keepdim=True).values
            g_cluster = p_cluster.max(dim=1, keepdim=True).values
            beta = beta.unsqueeze(0) * g_attr
            gamma = gamma.unsqueeze(0) * g_cluster
        else:
            beta = beta.unsqueeze(0)
            gamma = gamma.unsqueeze(0)
        
        if self.mode == "additive":
            # Additive fusion: p = p_det + beta * p_attr + gamma * p_cluster
            # Note: p_det is already a probability distribution (or close to it)
            prob = p_det + beta * p_attr + gamma * p_cluster
            
            prior = self._prior(device, expected)
            if prior is not None:
                # Subtract prior? For additive, maybe we just ignore delta or treat it differently.
                # Here we stick to a simple subtraction of scaled prior if delta > 0, 
                # but ensuring non-negativity.
                if self.delta > 0:
                    prob = prob - float(self.delta) * prior
                    prob = prob.clamp(min=self.eps)

            # Re-normalize
            row_sum = prob.sum(dim=1, keepdim=True).clamp(min=self.eps)
            return prob / row_sum

        # Default: PoE (Product of Experts)
        logit = (p_det + self.eps).log()
        logit = logit + beta * (p_attr + self.eps).log()
        logit = logit + gamma * (p_cluster + self.eps).log()
        prior = self._prior(device, expected)
        if prior is not None:
            logit = logit - float(self.delta) * (prior + self.eps).log()
        return F.softmax(logit, dim=1)
