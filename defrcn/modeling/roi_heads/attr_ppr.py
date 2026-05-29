"""PPR-based attribute-to-class inference and probability fusion utilities."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

Tensor = torch.Tensor


def _as_tensor(value: Union[Tensor, Sequence, float], device: torch.device, dtype: torch.dtype) -> Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _normalize_rows(matrix: Tensor, eps: float) -> Tensor:
    row_sum = matrix.sum(dim=1, keepdim=True).clamp(min=eps)
    return matrix / row_sum


def _ensure_2d(x: Tensor) -> Tensor:
    if x.dim() == 1:
        return x.unsqueeze(0)
    return x


def _apply_unique_boost(
    weights: Tensor,
    unique_attrs: Optional[Union[Tensor, Dict[int, Sequence[int]], Sequence[Sequence[int]]]],
    boost: float,
) -> Tensor:
    if unique_attrs is None or boost <= 1.0:
        return weights
    w = weights.clone()
    if isinstance(unique_attrs, torch.Tensor):
        mask = unique_attrs
        if mask.dim() != 2:
            return w
        if mask.shape == w.shape:
            w = w * (1.0 + (boost - 1.0) * mask.to(dtype=w.dtype, device=w.device))
            return w
        if mask.t().shape == w.shape:
            mask_t = mask.t()
            w = w * (1.0 + (boost - 1.0) * mask_t.to(dtype=w.dtype, device=w.device))
            return w
        return w
    if isinstance(unique_attrs, dict):
        for cls_idx, attr_indices in unique_attrs.items():
            if cls_idx < 0 or cls_idx >= w.shape[1]:
                continue
            idx = torch.as_tensor(list(attr_indices), device=w.device, dtype=torch.long)
            if idx.numel() == 0:
                continue
            idx = idx.clamp(min=0, max=w.shape[0] - 1)
            w[idx, cls_idx] = w[idx, cls_idx] * boost
        return w
    if isinstance(unique_attrs, (list, tuple)):
        for cls_idx, attr_indices in enumerate(unique_attrs):
            if cls_idx < 0 or cls_idx >= w.shape[1]:
                continue
            if not isinstance(attr_indices, Iterable):
                continue
            idx = torch.as_tensor(list(attr_indices), device=w.device, dtype=torch.long)
            if idx.numel() == 0:
                continue
            idx = idx.clamp(min=0, max=w.shape[0] - 1)
            w[idx, cls_idx] = w[idx, cls_idx] * boost
        return w
    return w


class PPRAttributeInference(nn.Module):
    """
    Personalized PageRank (PPR) inference over a bipartite attribute-class graph.

    Inputs:
      - a_hat: attribute probabilities, shape [A] or [N, A]
      - W: attribute-to-class weights, shape [A, C], non-negative
      - attribute_reliability (optional): shape [A] or [N, A], values in [0, 1]
      - unique_attrs (optional): list/dict/mask describing per-class unique attributes

    Output:
      - p_attr: class probabilities, shape [N, C], non-negative and sum to 1 per row
    """

    def __init__(
        self,
        alpha: float = 0.85,
        max_iter: int = 50,
        tol: float = 1e-6,
        eps: float = 1e-12,
        unique_edge_boost: float = 1.0,
        shared_penalty_gamma: float = 0.0,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.eps = float(eps)
        self.unique_edge_boost = float(unique_edge_boost)
        self.shared_penalty_gamma = float(shared_penalty_gamma)

    def forward(
        self,
        a_hat: Tensor,
        W: Tensor,
        attribute_reliability: Optional[Tensor] = None,
        unique_attrs: Optional[Union[Tensor, Dict[int, Sequence[int]], Sequence[Sequence[int]]]] = None,
    ) -> Tensor:
        device = a_hat.device
        dtype = a_hat.dtype

        W = _as_tensor(W, device=device, dtype=dtype)
        W = W.clamp_min(0.0)
        W = _apply_unique_boost(W, unique_attrs, self.unique_edge_boost)

        if self.shared_penalty_gamma > 0:
            deg = (W > 0).to(dtype=dtype).sum(dim=1).clamp(min=1.0)
            penalty = deg.pow(-self.shared_penalty_gamma).unsqueeze(1)
            W = W * penalty

        a_hat = _ensure_2d(a_hat)
        if attribute_reliability is not None:
            rel = _ensure_2d(_as_tensor(attribute_reliability, device=device, dtype=dtype))
            if rel.shape != a_hat.shape:
                rel = rel.expand_as(a_hat)
            a_hat = a_hat * rel

        a_sum = a_hat.sum(dim=1, keepdim=True)
        a_uniform = torch.full_like(a_hat, 1.0 / max(1, a_hat.shape[1]))
        a_hat = torch.where(a_sum > self.eps, a_hat, a_uniform)
        a_hat = a_hat / a_hat.sum(dim=1, keepdim=True).clamp(min=self.eps)

        num_attr, num_cls = W.shape
        p_a2c = _normalize_rows(W, self.eps)
        p_c2a = _normalize_rows(W.t(), self.eps)

        top = torch.cat([torch.zeros((num_attr, num_attr), device=device, dtype=dtype), p_a2c], dim=1)
        bottom = torch.cat([p_c2a, torch.zeros((num_cls, num_cls), device=device, dtype=dtype)], dim=1)
        P = torch.cat([top, bottom], dim=0)
        P = _normalize_rows(P, self.eps)

        v = torch.zeros((a_hat.shape[0], num_attr + num_cls), device=device, dtype=dtype)
        v[:, :num_attr] = a_hat
        v = v / v.sum(dim=1, keepdim=True).clamp(min=self.eps)

        p = v.clone()
        for _ in range(self.max_iter):
            p_next = (1.0 - self.alpha) * v + self.alpha * (p @ P)
            delta = (p_next - p).abs().sum(dim=1).max()
            p = p_next
            if float(delta.item()) < self.tol:
                break

        p_cls = p[:, num_attr:]
        p_cls = p_cls.clamp_min(0.0)
        p_cls = p_cls / p_cls.sum(dim=1, keepdim=True).clamp(min=self.eps)
        return p_cls


def fuse_probs_proben(
    p_det: Tensor,
    p_attr: Tensor,
    pi: Optional[Tensor] = None,
    alpha_det: float = 1.0,
    beta_attr: float = 1.0,
    gamma: Optional[float] = 1.0,
    eps: float = 1e-12,
) -> Tensor:
    """
    Product-of-experts fusion on probability simplex.

    Args:
      p_det: detector probabilities, shape [C] or [N, C]
      p_attr: attribute probabilities, shape [C] or [N, C]
      pi: optional class prior, shape [C] (uniform if None)
      alpha_det: detector weight
      beta_attr: attribute weight
      gamma: prior exponent; if None, use (alpha_det + beta_attr - 1)
      eps: numerical stability

    Returns:
      p_fuse: fused probabilities, same shape as inputs
    """
    p_det = _ensure_2d(p_det)
    p_attr = _ensure_2d(p_attr)
    if p_det.shape != p_attr.shape:
        raise ValueError("p_det and p_attr must have the same shape.")
    device = p_det.device
    dtype = p_det.dtype
    if pi is None:
        pi = torch.full((p_det.shape[1],), 1.0 / max(1, p_det.shape[1]), device=device, dtype=dtype)
    pi = _as_tensor(pi, device=device, dtype=dtype)
    pi = pi / pi.sum().clamp(min=eps)
    if gamma is None:
        gamma_val = float(alpha_det + beta_attr - 1.0)
    else:
        gamma_val = float(gamma)

    logit = (
        float(alpha_det) * (p_det + eps).log()
        + float(beta_attr) * (p_attr + eps).log()
        - float(gamma_val) * (pi + eps).log()
    )
    p_fuse = F.softmax(logit, dim=-1)
    return p_fuse



