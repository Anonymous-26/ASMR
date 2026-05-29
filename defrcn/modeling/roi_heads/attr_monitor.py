from __future__ import annotations

from typing import Optional

import torch
from detectron2.utils import comm
from torch.utils.tensorboard import SummaryWriter


class AttributeMonitor:
    def __init__(
        self,
        log_dir: str,
        log_period: int = 50,
        max_gate_log: int = 50,
        log_images: bool = True,
        log_hist: bool = True,
    ) -> None:
        self.log_period = max(int(log_period), 1)
        self.max_gate_log = max(int(max_gate_log), 0)
        self.log_images = bool(log_images)
        self.log_hist = bool(log_hist)
        self.writer: Optional[SummaryWriter] = None
        self._last_step = -1

        if comm.is_main_process():
            self.writer = SummaryWriter(log_dir=log_dir)

    def _should_log(self, step: Optional[int]) -> bool:
        if self.writer is None:
            return False
        if step is None:
            return False
        if step == self._last_step:
            return False
        if step % self.log_period != 0:
            return False
        self._last_step = step
        return True

    @staticmethod
    def _to_grayscale_image(matrix: torch.Tensor) -> torch.Tensor:
        img = matrix.detach().float()
        if img.numel() == 0:
            return img.new_zeros((1, 1, 1))
        min_val = float(img.min().item())
        max_val = float(img.max().item())
        denom = max(max_val - min_val, 1e-12)
        img = (img - min_val) / denom
        return img.unsqueeze(0)

    @staticmethod
    def _prototype_similarity(class_prototypes: torch.Tensor) -> torch.Tensor:
        if class_prototypes is None or class_prototypes.numel() == 0:
            return torch.zeros((1, 1))
        proto = torch.nn.functional.normalize(class_prototypes, dim=-1)
        return torch.matmul(proto, proto.t())

    def log(
        self,
        step: Optional[int],
        class_prototypes: Optional[torch.Tensor],
    ) -> None:
        if not self._should_log(step):
            return

        if class_prototypes is not None:
            proto_cpu = class_prototypes.detach().float().cpu()
            if self.log_images:
                sim = self._prototype_similarity(proto_cpu)
                self.writer.add_image(
                    "attr_su/prototype_similarity",
                    self._to_grayscale_image(sim),
                    step,
                )

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
