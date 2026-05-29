from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def build_background_filter_payload(
    *,
    gt_classes_before: torch.Tensor,
    gt_classes_after: torch.Tensor,
    bg_indices: torch.Tensor,
    novel_indices: torch.Tensor,
    max_sim: torch.Tensor,
    max_idx: torch.Tensor,
    margin: torch.Tensor,
    suppress_mask: torch.Tensor,
    pseudo_mask: torch.Tensor,
    proposal_boxes: Optional[torch.Tensor],
    gt_boxes: Optional[Sequence[Optional[torch.Tensor]]],
    num_preds_per_image: Sequence[int],
    bg_class_index: int,
    background_class_name: str,
    class_names: Sequence[str],
    file_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    total_rois = int(gt_classes_before.numel())
    num_bg_candidates = int(bg_indices.numel())
    num_raw_ignored = int(suppress_mask.sum().item())
    ignored_mask = gt_classes_after[bg_indices] == -1
    pseudo_final_mask = (gt_classes_after[bg_indices] >= 0) & (
        gt_classes_after[bg_indices] != bg_class_index
    )
    num_ignored = int(ignored_mask.sum().item())
    num_pseudo = int(pseudo_final_mask.sum().item())

    denom_all = max(total_rois, 1)
    denom_bg = max(num_bg_candidates, 1)
    stats = {
        "num_input_rois": total_rois,
        "num_bg_candidates": num_bg_candidates,
        "num_raw_ignored": num_raw_ignored,
        "num_ignored": num_ignored,
        "num_pseudo": num_pseudo,
        "ratio_bg_candidates_over_all": float(num_bg_candidates) / float(denom_all),
        "ratio_raw_ignored_over_bg": float(num_raw_ignored) / float(denom_bg),
        "ratio_ignored_over_all": float(num_ignored) / float(denom_all),
        "ratio_pseudo_over_all": float(num_pseudo) / float(denom_all),
        "ratio_ignored_over_bg": float(num_ignored) / float(denom_bg),
        "ratio_pseudo_over_bg": float(num_pseudo) / float(denom_bg),
    }

    offsets: List[Tuple[int, int]] = []
    start = 0
    for count in num_preds_per_image:
        end = start + int(count)
        offsets.append((start, end))
        start = end

    def decode_label(class_index: int) -> str:
        if class_index == -1:
            return "ignore"
        if class_index == bg_class_index:
            return background_class_name
        if 0 <= class_index < len(class_names):
            return str(class_names[class_index])
        return str(class_index)

    per_image: List[Dict[str, object]] = []
    if proposal_boxes is None or not offsets:
        return {"stats": stats, "per_image": per_image}

    suppress_mask_cpu = suppress_mask.detach().cpu()
    pseudo_mask_cpu = pseudo_mask.detach().cpu()
    bg_indices_cpu = bg_indices.detach().cpu()
    gt_classes_after_cpu = gt_classes_after.detach().cpu()
    max_sim_cpu = max_sim.detach().cpu()
    max_idx_cpu = max_idx.detach().cpu()
    margin_cpu = margin.detach().cpu()
    novel_indices_cpu = novel_indices.detach().cpu()
    boxes_cpu = proposal_boxes.detach().cpu()

    records_by_image: Dict[int, List[Dict[str, object]]] = {}
    for local_idx, global_idx_t in enumerate(bg_indices_cpu):
        global_idx = int(global_idx_t.item())
        image_index = 0
        roi_index_in_image = global_idx
        for idx, (start_idx, end_idx) in enumerate(offsets):
            if start_idx <= global_idx < end_idx:
                image_index = idx
                roi_index_in_image = global_idx - start_idx
                break

        final_class = int(gt_classes_after_cpu[global_idx].item())
        if final_class == -1:
            action = "suppress_to_ignore"
        elif final_class != bg_class_index:
            action = "pseudo_to_novel"
        else:
            action = "keep_bg"

        top1_class = int(novel_indices_cpu[int(max_idx_cpu[local_idx].item())].item())
        record = {
            "image_index": image_index,
            "roi_index_in_image": roi_index_in_image,
            "global_roi_index": global_idx,
            "box_xyxy": boxes_cpu[global_idx].tolist(),
            "label_before_name": background_class_name,
            "label_after_name": decode_label(final_class),
            "action": action,
            "top1_class_name": decode_label(top1_class),
            "max_sim": float(max_sim_cpu[local_idx].item()),
            "margin": float(margin_cpu[local_idx].item()),
            "passed_bg_threshold": bool(suppress_mask_cpu[local_idx].item()),
            "passed_pseudo_threshold": bool(pseudo_mask_cpu[local_idx].item()),
        }
        records_by_image.setdefault(image_index, []).append(record)

    for image_index, records in sorted(records_by_image.items()):
        per_image.append(
            {
                "image_index": image_index,
                "file_name": (
                    str(file_names[image_index])
                    if file_names is not None and image_index < len(file_names)
                    else ""
                ),
                "gt_boxes_xyxy": (
                    gt_boxes[image_index].detach().cpu().tolist()
                    if gt_boxes is not None
                    and image_index < len(gt_boxes)
                    and gt_boxes[image_index] is not None
                    else []
                ),
                "num_input_rois": int(offsets[image_index][1] - offsets[image_index][0]),
                "num_candidates": len(records),
                "num_changed": sum(
                    1 for record in records if record["action"] != "keep_bg"
                ),
                "records": records,
            }
        )

    return {"stats": stats, "per_image": per_image}


def filter_background_filter_payload(
    payload: Optional[Dict[str, object]],
    image_indices: Sequence[int],
) -> Optional[Dict[str, object]]:
    if not payload:
        return payload
    keep = {int(idx) for idx in image_indices}
    if not keep:
        return None

    per_image_all = payload.get("per_image", [])
    per_image = [
        entry for entry in per_image_all if int(entry.get("image_index", -1)) in keep
    ]
    if not per_image:
        return None

    total_rois = sum(int(entry.get("num_input_rois", 0)) for entry in per_image)
    num_bg_candidates = sum(int(entry.get("num_candidates", 0)) for entry in per_image)
    all_records = [
        record
        for entry in per_image
        for record in entry.get("records", [])
    ]
    num_raw_ignored = sum(
        int(bool(record.get("passed_bg_threshold", False))) for record in all_records
    )
    num_ignored = sum(
        int(record.get("action") == "suppress_to_ignore") for record in all_records
    )
    num_pseudo = sum(
        int(record.get("action") == "pseudo_to_novel") for record in all_records
    )
    denom_all = max(total_rois, 1)
    denom_bg = max(num_bg_candidates, 1)
    stats = {
        "num_input_rois": total_rois,
        "num_bg_candidates": num_bg_candidates,
        "num_raw_ignored": num_raw_ignored,
        "num_ignored": num_ignored,
        "num_pseudo": num_pseudo,
        "ratio_bg_candidates_over_all": float(num_bg_candidates) / float(denom_all),
        "ratio_raw_ignored_over_bg": float(num_raw_ignored) / float(denom_bg),
        "ratio_ignored_over_all": float(num_ignored) / float(denom_all),
        "ratio_pseudo_over_all": float(num_pseudo) / float(denom_all),
        "ratio_ignored_over_bg": float(num_ignored) / float(denom_bg),
        "ratio_pseudo_over_bg": float(num_pseudo) / float(denom_bg),
    }
    return {"stats": stats, "per_image": per_image}


def select_background_filter_images_by_action_diversity(
    payload: Optional[Dict[str, object]],
    min_action_types: int,
) -> List[int]:
    if not payload:
        return []
    if min_action_types <= 1:
        return [int(entry.get("image_index", -1)) for entry in payload.get("per_image", [])]

    selected: List[int] = []
    for entry in payload.get("per_image", []):
        action_types = {
            str(record.get("action", ""))
            for record in entry.get("records", [])
            if str(record.get("action", ""))
        }
        if len(action_types) >= int(min_action_types):
            selected.append(int(entry.get("image_index", -1)))
    return selected


class BackgroundFilterTensorboardLogger:
    NUM_PANELS = 3

    def __init__(
        self,
        *,
        writer,
        pixel_mean: Sequence[float],
        pixel_std: Sequence[float],
        input_format: str = "RGB",
        max_images: int = 2,
        max_rois: int = 20,
    ) -> None:
        self.writer = writer
        self.max_images = max(int(max_images), 1)
        self.max_rois = max(int(max_rois), 1)
        self.pixel_mean = torch.as_tensor(pixel_mean, dtype=torch.float32).view(-1, 1, 1)
        self.pixel_std = torch.as_tensor(pixel_std, dtype=torch.float32).view(-1, 1, 1)
        self.input_format = str(input_format).upper()

    @staticmethod
    def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    @staticmethod
    def _sanitize_tag(value: str) -> str:
        value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
        return value.strip("._") or "image"

    @staticmethod
    def _color_for_action(action: str) -> Tuple[int, int, int]:
        if action == "suppress_to_ignore":
            return (149, 165, 166)
        if action == "pseudo_to_novel":
            return (231, 76, 60)
        return (52, 152, 219)

    @staticmethod
    def _sort_records(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
        def key_fn(record: Dict[str, object]) -> Tuple[int, float, float]:
            return (
                int(record.get("roi_index_in_image", 0)),
                float(record.get("max_sim", 0.0)),
                float(record.get("margin", 0.0)),
            )

        return sorted(records, key=key_fn)

    def _sample_records(
        self,
        records: List[Dict[str, object]],
        image_index: int,
        step: Optional[int],
    ) -> Tuple[List[Dict[str, object]], int]:
        # Keep the image-level selection logic unchanged, but only visualize keep_bg
        # proposals so the panels are less cluttered.
        keep_records = self._sort_records(
            [
                record for record in records
                if str(record.get("action", "")) == "keep_bg"
            ]
        )
        capacity = max(1, int(np.ceil(self.max_rois * self.NUM_PANELS * 0.25)))
        if len(keep_records) <= capacity:
            return keep_records, self.NUM_PANELS

        seed = (
            int(image_index) * 1000003
            + int(step or 0) * 9176
            + len(records) * 131
        )
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(keep_records), size=capacity, replace=False)
        selected = [keep_records[int(idx)] for idx in chosen.tolist()]
        return self._sort_records(selected), self.NUM_PANELS

    @staticmethod
    def _build_panel_assignments(
        records: List[Dict[str, object]],
        num_panels: int,
    ) -> List[List[Dict[str, object]]]:
        num_panels = max(int(num_panels), 1)
        if not records:
            return [[] for _ in range(num_panels)]
        base = len(records) // num_panels
        remainder = len(records) % num_panels
        assignments: List[List[Dict[str, object]]] = []
        start = 0
        for panel_idx in range(num_panels):
            count = base + (1 if panel_idx < remainder else 0)
            end = start + count
            assignments.append(records[start:end])
            start = end
        return assignments

    def _denormalize_image(
        self,
        image_tensor: torch.Tensor,
        image_size: Sequence[int],
    ) -> Image.Image:
        height, width = int(image_size[0]), int(image_size[1])
        image = image_tensor.detach().cpu()[:, :height, :width].float()
        image = image * self.pixel_std + self.pixel_mean
        if self.input_format == "BGR" and image.shape[0] == 3:
            image = image[[2, 1, 0], :, :]
        image = image.clamp(0, 255).permute(1, 2, 0).numpy().astype(np.uint8)
        return Image.fromarray(image)

    def _draw_records(
        self,
        image: Image.Image,
        records: List[Dict[str, object]],
        gt_boxes: Optional[Sequence[Sequence[float]]] = None,
    ) -> Image.Image:
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        if gt_boxes:
            for gt_box in gt_boxes:
                x0, y0, x1, y1 = [int(round(v)) for v in gt_box]
                draw.rectangle([x0, y0, x1, y1], outline=(241, 196, 15), width=2)
        for record in records:
            x0, y0, x1, y1 = [int(round(v)) for v in record["box_xyxy"]]
            color = self._color_for_action(str(record["action"]))
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        return canvas

    def _compose_panel(
        self,
        *,
        step: Optional[int],
        image_index: int,
        image_tensor: torch.Tensor,
        image_size: Sequence[int],
        records: List[Dict[str, object]],
        gt_boxes: Optional[Sequence[Sequence[float]]] = None,
    ) -> Optional[torch.Tensor]:
        if not records:
            return None
        records, num_panels = self._sample_records(records, image_index, step)
        panel_assignments = self._build_panel_assignments(records, num_panels)
        base_image = self._denormalize_image(image_tensor, image_size)
        width, height = base_image.size
        panel = Image.new(
            "RGB", (width * num_panels, height), color=(255, 255, 255)
        )
        for panel_idx, chunk in enumerate(panel_assignments):
            sub_image = self._draw_records(base_image, chunk, gt_boxes=gt_boxes)
            panel.paste(sub_image, (width * panel_idx, 0))
        return self._pil_to_tensor(panel)

    def log(
        self,
        *,
        step: Optional[int],
        payload: Optional[Dict[str, object]],
        images_tensor: Optional[torch.Tensor],
        image_sizes: Optional[Sequence[Sequence[int]]],
    ) -> None:
        if self.writer is None or step is None or not payload:
            return
        stats = payload.get("stats")
        if stats:
            for key, value in stats.items():
                prefix = "bg_filter/count" if key.startswith("num_") else "bg_filter/ratio"
                self.writer.add_scalar(f"{prefix}/{key}", float(value), step)

        if images_tensor is None or image_sizes is None:
            return

        ranked = sorted(
            payload.get("per_image", []),
            key=lambda item: (
                -int(item.get("num_changed", 0)),
                -int(item.get("num_candidates", 0)),
                int(item.get("image_index", 0)),
            ),
        )
        for entry in ranked[: self.max_images]:
            image_index = int(entry["image_index"])
            if image_index >= len(image_sizes) or image_index >= images_tensor.shape[0]:
                continue
            panel = self._compose_panel(
                step=step,
                image_index=image_index,
                image_tensor=images_tensor[image_index],
                image_size=image_sizes[image_index],
                records=entry.get("records", []),
                gt_boxes=entry.get("gt_boxes_xyxy", []),
            )
            if panel is not None:
                file_name = str(entry.get("file_name", "") or "")
                image_tag = f"image_{image_index}"
                if file_name:
                    image_tag = self._sanitize_tag(
                        os.path.splitext(os.path.basename(file_name))[0]
                    )
                self.writer.add_image(f"bg_filter/{image_tag}/iter_{step}", panel, step)
