"""Attribute-to-class probability inference utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

Tensor = torch.Tensor


def _load_prob_matrix(path: Path) -> Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Prob class matrix not found at {path}")
    if path.suffix in {".npz", ".npy"}:
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.lib.npyio.NpzFile):
            for key in ("p_c_given_a", "prob_class", "P", "matrix"):
                if key in data:
                    return torch.from_numpy(data[key].astype(np.float32))
            first_key = list(data.keys())[0]
            return torch.from_numpy(data[first_key].astype(np.float32))
        return torch.from_numpy(np.asarray(data, dtype=np.float32))
    if path.suffix in {".json"}:
        import json

        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if isinstance(payload, dict):
            for key in ("p_c_given_a", "prob_class", "P", "matrix"):
                if key in payload:
                    return torch.as_tensor(payload[key], dtype=torch.float32)
        return torch.as_tensor(payload, dtype=torch.float32)
    raise ValueError(f"Unsupported prob class matrix format: {path}")


class AttributeProbClassInference(nn.Module):
    """
    Compute class probabilities from attribute probabilities with a fixed prior:

      P(c | ROI) = sum_j P(c | a_j) * P(a_j | ROI)

    Inputs:
      - attr_probs: shape [N, A], P(a_j | ROI)
      - prob_c_given_a: shape [C, A]
    Output:
      - class_probs: shape [N, C]
    """

    def __init__(
        self,
        prob_path: Optional[str],
        eps: float = 1e-12,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.normalize = bool(normalize)
        self.prob_path = Path(prob_path) if prob_path else None
        self.register_buffer("prob_c_given_a", torch.empty(0))
        if self.prob_path:
            self._load()

    def _load(self) -> None:
        if self.prob_path is None:
            return
        matrix = _load_prob_matrix(self.prob_path)
        self.prob_c_given_a = matrix

    def forward(self, attr_probs: Tensor, prob_c_given_a: Optional[Tensor] = None) -> Tensor:
        if prob_c_given_a is None:
            if self.prob_c_given_a.numel() == 0:
                raise ValueError("prob_c_given_a is empty; check PROB_CLASS.PATH.")
            prob = self.prob_c_given_a
        else:
            prob = prob_c_given_a
        prob = prob.to(device=attr_probs.device, dtype=attr_probs.dtype)
        if prob.dim() != 2:
            raise ValueError("prob_c_given_a must be 2D [C, A].")
        if attr_probs.dim() != 2:
            raise ValueError("attr_probs must be 2D [N, A].")
        if prob.shape[1] != attr_probs.shape[1]:
            raise ValueError(
                f"attr_probs has {attr_probs.shape[1]} attrs but prob matrix has {prob.shape[1]}"
            )
        class_probs = attr_probs @ prob.t()
        class_probs = class_probs.clamp_min(0.0)
        if self.normalize:
            class_probs = class_probs / class_probs.sum(dim=1, keepdim=True).clamp(
                min=self.eps
            )
        return class_probs
