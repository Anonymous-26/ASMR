import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils import comm

from defrcn.modeling.roi_heads.attr_modules import AttributePrototypeBank

from .evaluator import DatasetEvaluator

logger = logging.getLogger(__name__)


def _collect_class_names(cfg, dataset_name: Optional[str]) -> Optional[List[str]]:
    attr_cfg = cfg.MODEL.ATTRIBUTE
    if attr_cfg.CLASS_NAMES:
        return list(attr_cfg.CLASS_NAMES)
    if dataset_name:
        metadata = MetadataCatalog.get(dataset_name)
        thing_classes = getattr(metadata, "thing_classes", None)
        if thing_classes:
            return list(thing_classes)
    return None


def _pairwise_ious(
    gt_instances: Instances, pred_instances: Instances
) -> torch.Tensor:
    h_gt, w_gt = gt_instances.image_size
    h_pred, w_pred = pred_instances.image_size
    if (h_gt, w_gt) == (h_pred, w_pred):
        return pairwise_iou(gt_instances.gt_boxes, pred_instances.pred_boxes)
    scale_y = float(h_gt) / float(h_pred)
    scale_x = float(w_gt) / float(w_pred)
    pred_boxes_tensor = pred_instances.pred_boxes.tensor.clone()
    pred_boxes_tensor[:, 0] *= scale_x
    pred_boxes_tensor[:, 2] *= scale_x
    pred_boxes_tensor[:, 1] *= scale_y
    pred_boxes_tensor[:, 3] *= scale_y
    scaled_pred_boxes = Boxes(pred_boxes_tensor)
    return pairwise_iou(gt_instances.gt_boxes, scaled_pred_boxes)


def _match_predictions(
    gt_instances: Instances,
    pred_instances: Instances,
    matching_iou: float,
) -> List[Tuple[int, int, float]]:
    if len(gt_instances) == 0 or len(pred_instances) == 0:
        return []
    if not gt_instances.has("gt_boxes") or not pred_instances.has("pred_boxes"):
        return []
    ious = _pairwise_ious(gt_instances, pred_instances)
    pred_scores = (
        pred_instances.scores
        if pred_instances.has("scores")
        else torch.zeros(len(pred_instances))
    )
    used = torch.zeros(len(pred_instances), dtype=torch.bool)
    matches: List[Tuple[int, int, float]] = []

    for gt_idx in range(len(gt_instances)):
        iou_row = ious[gt_idx]
        candidate_mask = (iou_row >= matching_iou) & (~used)
        if not torch.any(candidate_mask):
            continue
        candidate_inds = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)
        best_idx = candidate_inds[0]
        best_iou = iou_row[best_idx]
        best_score = pred_scores[best_idx]
        for idx in candidate_inds[1:]:
            iou_val = iou_row[idx]
            score = pred_scores[idx]
            if iou_val > best_iou or (iou_val == best_iou and score > best_score):
                best_iou = iou_val
                best_score = score
                best_idx = idx
        used[best_idx] = True
        matches.append((gt_idx, int(best_idx), float(best_iou)))
    return matches


class AttributeClusterStatsEvaluator(DatasetEvaluator):
    """
    Collect attribute-cluster usage statistics by matching predictions to GT boxes.
    """

    def __init__(
        self,
        cfg,
        output_path: str,
        dataset_name: Optional[str] = None,
        matching_iou: Optional[float] = None,
    ) -> None:
        self.cfg = cfg
        attr_cfg = cfg.MODEL.ATTRIBUTE
        if not attr_cfg.ENABLED:
            raise ValueError("AttributeClusterStatsEvaluator requires MODEL.ATTRIBUTE.ENABLED = True")
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.matching_iou = matching_iou if matching_iou is not None else attr_cfg.EVAL_MATCH_IOU
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        class_names = _collect_class_names(cfg, dataset_name)

        prototype_bank = AttributePrototypeBank(
            super_attr_path=attr_cfg.SUPER_ATTR_PATH,
            class_names=class_names,
            background_name=attr_cfg.BACKGROUND_CLASS,
        )
        cluster_state = prototype_bank.get_cluster_state(torch.device("cpu"))
        self.num_clusters = int(cluster_state["incidence"].shape[0])
        self.reset()

    def reset(self) -> None:
        device = torch.device("cpu")
        self.cluster_class_sum = torch.zeros(
            (self.num_clusters, self.num_classes),
            dtype=torch.float64,
            device=device,
        )
        self.class_count = torch.zeros(self.num_classes, dtype=torch.int64, device=device)
        self.match_iou_hist = torch.zeros(10, dtype=torch.int64, device=device)
        self.match_iou_sum = 0.0
        self.match_iou_count = 0

    def process(self, inputs: Sequence[Dict], outputs: Sequence[Dict]) -> None:
        if not isinstance(inputs, Sequence):
            inputs = [inputs]
        if not isinstance(outputs, Sequence):
            outputs = [outputs]

        for input_record, output_record in zip(inputs, outputs):
            gt_instances = input_record.get("instances")
            pred_instances = output_record.get("instances")
            if gt_instances is None or pred_instances is None:
                continue

            gt_instances = gt_instances.to("cpu")
            pred_instances = pred_instances.to("cpu")
            matches = _match_predictions(gt_instances, pred_instances, self.matching_iou)
            if not matches:
                continue
            if not pred_instances.has("attr_cluster_probs"):
                continue
            attr_cluster_probs = pred_instances.attr_cluster_probs
            if attr_cluster_probs.ndim != 2 or attr_cluster_probs.shape[1] != self.num_clusters:
                continue

            for gt_idx, pred_idx, match_iou in matches:
                if attr_cluster_probs.shape[0] <= pred_idx:
                    continue
                gt_class = int(gt_instances.gt_classes[gt_idx])
                if gt_class < 0 or gt_class >= self.num_classes:
                    continue
                probs = attr_cluster_probs[pred_idx].to(torch.float64)
                self.cluster_class_sum[:, gt_class] += probs
                self.class_count[gt_class] += 1
                self._record_match_iou(match_iou)

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        results: Dict[str, Dict[str, float]] = {}
        local_sums = self.cluster_class_sum.cpu().numpy()
        local_counts = self.class_count.cpu().numpy()
        local_hist = self.match_iou_hist.cpu().numpy()
        local_iou_sum = np.array([self.match_iou_sum], dtype=np.float64)
        local_iou_count = np.array([self.match_iou_count], dtype=np.int64)
        all_sums = comm.gather(local_sums, dst=0)
        all_counts = comm.gather(local_counts, dst=0)
        all_hist = comm.gather(local_hist, dst=0)
        all_iou_sum = comm.gather(local_iou_sum, dst=0)
        all_iou_count = comm.gather(local_iou_count, dst=0)
        if not comm.is_main_process():
            return results

        total_sums = np.sum(np.stack(all_sums, axis=0), axis=0)
        total_counts = np.sum(np.stack(all_counts, axis=0), axis=0)
        total_hist = np.sum(np.stack(all_hist, axis=0), axis=0)
        total_iou_sum = float(np.sum(np.stack(all_iou_sum, axis=0)))
        total_iou_count = int(np.sum(np.stack(all_iou_count, axis=0)))
        total_rois = int(total_counts.sum())
        if total_rois <= 0:
            logger.warning("AttributeClusterStatsEvaluator collected no samples.")
            return results

        eps = 1e-12
        cluster_sum = torch.from_numpy(total_sums).to(torch.float64)
        class_count = torch.from_numpy(total_counts).to(torch.float64).clamp(min=1.0)
        p_k_c = cluster_sum / class_count
        row_sum = p_k_c.sum(dim=1, keepdim=True).clamp(min=eps)
        p_c_k = p_k_c / row_sum
        safe_probs = p_c_k.clamp(min=eps)
        entropy = -(p_c_k * safe_probs.log()).sum(dim=1)

        np.savez(
            self.output_path,
            p_k_c=p_k_c.cpu().numpy().astype(np.float32),
            p_c_k=p_c_k.cpu().numpy().astype(np.float32),
            entropy=entropy.cpu().numpy().astype(np.float32),
            cluster_class_sum=total_sums.astype(np.float32),
            class_count=total_counts.astype(np.int64),
            match_iou_hist=total_hist.astype(np.int64),
            match_iou_sum=np.array(total_iou_sum, dtype=np.float64),
            match_iou_count=np.array(total_iou_count, dtype=np.int64),
        )
        avg_match_iou = total_iou_sum / max(1, total_iou_count)
        results["attr_cluster_stats"] = {
            "num_rois": float(total_rois),
            "match_iou_avg": float(avg_match_iou),
            "match_iou_count": float(total_iou_count),
        }
        return results

    def _record_match_iou(self, iou_value: float) -> None:
        iou = max(0.0, min(1.0, float(iou_value)))
        self.match_iou_sum += iou
        self.match_iou_count += 1
        num_bins = int(self.match_iou_hist.numel())
        if num_bins <= 0:
            return
        bin_idx = min(int(iou * num_bins), num_bins - 1)
        self.match_iou_hist[bin_idx] += 1


class AttributeEvaluator(DatasetEvaluator):
    """
    Evaluate attribute head outputs on GT-aligned instances.

    Metrics:
      - det_class_top{1,2,3}_acc from det_class_probs vs gt class.
      - det_class_top{1,2,3}_rec from det_class_probs vs gt class.
      - attr_class_top{1,2,3}_acc from attr_probs vs gt class.
      - attr_class_top{1,2,3}_rec from attr_probs vs gt class.
      - attr_cluster_top{1,2,3}_acc from attr_cluster_probs vs GT clusters.
    """

    def __init__(
        self,
        cfg,
        dataset_name: Optional[str] = None,
        matching_iou: Optional[float] = None,
    ) -> None:
        self.cfg = cfg
        attr_cfg = cfg.MODEL.ATTRIBUTE
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.matching_iou = matching_iou if matching_iou is not None else attr_cfg.EVAL_MATCH_IOU
        self.class_topk_values = (1, 2, 3)
        self.cluster_topk_values = (1, 2, 3)

        class_names = _collect_class_names(cfg, dataset_name)
        prototype_bank = AttributePrototypeBank(
            super_attr_path=attr_cfg.SUPER_ATTR_PATH,
            class_names=class_names,
            background_name=attr_cfg.BACKGROUND_CLASS,
        )
        cluster_state = prototype_bank.get_cluster_state(torch.device("cpu"))
        incidence = cluster_state["incidence"]
        self.cluster_category = (incidence > 0.0).to(torch.bool)
        self.num_clusters = int(self.cluster_category.shape[0])

        self.reset()

    def reset(self) -> None:
        self.det_class_total = 0
        self.det_class_correct = {k: 0 for k in self.class_topk_values}
        self.det_class_base_total = 0
        self.det_class_base_correct = {k: 0 for k in self.class_topk_values}
        self.det_class_novel_total = 0
        self.det_class_novel_correct = {k: 0 for k in self.class_topk_values}
        self.det_class_rec_total = 0
        self.det_class_rec_correct = {k: 0 for k in self.class_topk_values}
        self.det_class_rec_base_total = 0
        self.det_class_rec_base_correct = {k: 0 for k in self.class_topk_values}
        self.det_class_rec_novel_total = 0
        self.det_class_rec_novel_correct = {k: 0 for k in self.class_topk_values}
        self.attr_class_total = 0
        self.attr_class_correct = {k: 0 for k in self.class_topk_values}
        self.attr_class_base_total = 0
        self.attr_class_base_correct = {k: 0 for k in self.class_topk_values}
        self.attr_class_novel_total = 0
        self.attr_class_novel_correct = {k: 0 for k in self.class_topk_values}
        self.attr_class_rec_total = 0
        self.attr_class_rec_correct = {k: 0 for k in self.class_topk_values}
        self.attr_class_rec_base_total = 0
        self.attr_class_rec_base_correct = {k: 0 for k in self.class_topk_values}
        self.attr_class_rec_novel_total = 0
        self.attr_class_rec_novel_correct = {k: 0 for k in self.class_topk_values}
        self.attr_cluster_total = 0
        self.attr_cluster_correct = {k: 0 for k in self.cluster_topk_values}
        self.attr_cluster_base_total = 0
        self.attr_cluster_base_correct = {k: 0 for k in self.cluster_topk_values}
        self.attr_cluster_novel_total = 0
        self.attr_cluster_novel_correct = {k: 0 for k in self.cluster_topk_values}
        self.per_attr_total = np.zeros(self.num_clusters, dtype=np.int64)
        self.per_attr_correct = {
            k: np.zeros(self.num_clusters, dtype=np.int64)
            for k in self.cluster_topk_values
        }

    def process(self, inputs: Sequence[Dict], outputs: Sequence[Dict]) -> None:
        if not isinstance(inputs, Sequence):
            inputs = [inputs]
        if not isinstance(outputs, Sequence):
            outputs = [outputs]

        for input_record, output_record in zip(inputs, outputs):
            gt_instances = input_record.get("instances")
            pred_instances = output_record.get("instances")
            if gt_instances is None or pred_instances is None:
                continue

            gt_instances = gt_instances.to("cpu")
            pred_instances = pred_instances.to("cpu")
            matches = _match_predictions(gt_instances, pred_instances, self.matching_iou)
            if not matches:
                matches = []

            gt_classes = gt_instances.gt_classes
            valid_gt_mask = (gt_classes >= 0) & (gt_classes < self.num_classes)
            if torch.any(valid_gt_mask):
                for gt_class in gt_classes[valid_gt_mask].tolist():
                    is_base = gt_class < 15
                    self.det_class_rec_total += 1
                    self.attr_class_rec_total += 1
                    if is_base:
                        self.det_class_rec_base_total += 1
                        self.attr_class_rec_base_total += 1
                    else:
                        self.det_class_rec_novel_total += 1
                        self.attr_class_rec_novel_total += 1

            det_class_probs = (
                pred_instances.det_class_probs.detach().cpu()
                if pred_instances.has("det_class_probs")
                else None
            )
            attr_probs = (
                pred_instances.attr_probs.detach().cpu()
                if pred_instances.has("attr_probs")
                else None
            )
            attr_cluster_probs = (
                pred_instances.attr_cluster_probs.detach().cpu()
                if pred_instances.has("attr_cluster_probs")
                else None
            )

            for gt_idx, pred_idx, _match_iou in matches:
                gt_class = int(gt_classes[gt_idx])
                if gt_class < 0 or gt_class >= self.num_classes:
                    continue

                if det_class_probs is not None and det_class_probs.shape[0] > pred_idx:
                    class_probs = det_class_probs[pred_idx]
                    self.det_class_total += 1
                    is_base = gt_class < 15
                    if is_base:
                        self.det_class_base_total += 1
                    else:
                        self.det_class_novel_total += 1
                    for topk in self.class_topk_values:
                        limit = min(topk, class_probs.shape[0])
                        if limit <= 0:
                            continue
                        topk_indices = torch.topk(class_probs, k=limit).indices
                        if int(gt_class) in topk_indices.tolist():
                            self.det_class_correct[topk] += 1
                            if is_base:
                                self.det_class_base_correct[topk] += 1
                            else:
                                self.det_class_novel_correct[topk] += 1
                            self.det_class_rec_correct[topk] += 1
                            if is_base:
                                self.det_class_rec_base_correct[topk] += 1
                            else:
                                self.det_class_rec_novel_correct[topk] += 1

                if attr_probs is not None and attr_probs.shape[0] > pred_idx:
                    class_probs = attr_probs[pred_idx]
                    self.attr_class_total += 1
                    is_base = gt_class < 15
                    if is_base:
                        self.attr_class_base_total += 1
                    else:
                        self.attr_class_novel_total += 1
                    for topk in self.class_topk_values:
                        limit = min(topk, class_probs.shape[0])
                        if limit <= 0:
                            continue
                        topk_indices = torch.topk(class_probs, k=limit).indices
                        if int(gt_class) in topk_indices.tolist():
                            self.attr_class_correct[topk] += 1
                            if is_base:
                                self.attr_class_base_correct[topk] += 1
                            else:
                                self.attr_class_novel_correct[topk] += 1
                            self.attr_class_rec_correct[topk] += 1
                            if is_base:
                                self.attr_class_rec_base_correct[topk] += 1
                            else:
                                self.attr_class_rec_novel_correct[topk] += 1

                if (
                    attr_cluster_probs is not None
                    and attr_cluster_probs.shape[0] > pred_idx
                    and self.num_clusters > 0
                ):
                    if gt_class >= self.cluster_category.shape[1]:
                        continue
                    gt_attr = self.cluster_category[:, gt_class]
                    if not gt_attr.any():
                        continue
                    cluster_scores = attr_cluster_probs[pred_idx]
                    if cluster_scores.shape[0] != gt_attr.shape[0]:
                        continue
                    gt_bool = gt_attr.to(torch.bool)
                    self.per_attr_total += gt_bool.to(torch.int64).cpu().numpy()
                    self.attr_cluster_total += 1
                    is_base = gt_class < 15
                    if is_base:
                        self.attr_cluster_base_total += 1
                    else:
                        self.attr_cluster_novel_total += 1
                    order = torch.argsort(cluster_scores, descending=True)
                    for topk in self.cluster_topk_values:
                        limit = min(topk, order.numel())
                        if limit <= 0:
                            continue
                        topk_idx = order[:limit]
                        strict_hit = bool(gt_bool[topk_idx].all().item())
                        if strict_hit:
                            self.attr_cluster_correct[topk] += 1
                            if is_base:
                                self.attr_cluster_base_correct[topk] += 1
                            else:
                                self.attr_cluster_novel_correct[topk] += 1
                        if strict_hit:
                            hit_mask = torch.zeros_like(gt_bool, dtype=torch.bool)
                            hit_mask[topk_idx] = True
                            per_hit = (gt_bool & hit_mask).to(torch.int64).cpu().numpy()
                            self.per_attr_correct[topk] += per_hit

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        metrics: Dict[str, float] = {}
        local_counts = np.array(
            [
                self.det_class_total,
                self.det_class_correct[1],
                self.det_class_correct[2],
                self.det_class_correct[3],
                self.det_class_base_total,
                self.det_class_base_correct[1],
                self.det_class_base_correct[2],
                self.det_class_base_correct[3],
                self.det_class_novel_total,
                self.det_class_novel_correct[1],
                self.det_class_novel_correct[2],
                self.det_class_novel_correct[3],
                self.det_class_rec_total,
                self.det_class_rec_correct[1],
                self.det_class_rec_correct[2],
                self.det_class_rec_correct[3],
                self.det_class_rec_base_total,
                self.det_class_rec_base_correct[1],
                self.det_class_rec_base_correct[2],
                self.det_class_rec_base_correct[3],
                self.det_class_rec_novel_total,
                self.det_class_rec_novel_correct[1],
                self.det_class_rec_novel_correct[2],
                self.det_class_rec_novel_correct[3],
                self.attr_class_total,
                self.attr_class_correct[1],
                self.attr_class_correct[2],
                self.attr_class_correct[3],
                self.attr_class_base_total,
                self.attr_class_base_correct[1],
                self.attr_class_base_correct[2],
                self.attr_class_base_correct[3],
                self.attr_class_novel_total,
                self.attr_class_novel_correct[1],
                self.attr_class_novel_correct[2],
                self.attr_class_novel_correct[3],
                self.attr_class_rec_total,
                self.attr_class_rec_correct[1],
                self.attr_class_rec_correct[2],
                self.attr_class_rec_correct[3],
                self.attr_class_rec_base_total,
                self.attr_class_rec_base_correct[1],
                self.attr_class_rec_base_correct[2],
                self.attr_class_rec_base_correct[3],
                self.attr_class_rec_novel_total,
                self.attr_class_rec_novel_correct[1],
                self.attr_class_rec_novel_correct[2],
                self.attr_class_rec_novel_correct[3],
                self.attr_cluster_total,
                self.attr_cluster_correct[1],
                self.attr_cluster_correct[2],
                self.attr_cluster_correct[3],
                self.attr_cluster_base_total,
                self.attr_cluster_base_correct[1],
                self.attr_cluster_base_correct[2],
                self.attr_cluster_base_correct[3],
                self.attr_cluster_novel_total,
                self.attr_cluster_novel_correct[1],
                self.attr_cluster_novel_correct[2],
                self.attr_cluster_novel_correct[3],
            ],
            dtype=np.int64,
        )
        per_attr_payload = [
            self.per_attr_total,
            self.per_attr_correct[1],
            self.per_attr_correct[2],
            self.per_attr_correct[3],
        ]
        all_counts = comm.gather(local_counts, dst=0)
        all_per_attr = comm.gather(per_attr_payload, dst=0)
        if not comm.is_main_process():
            return {}
        totals = np.sum(np.stack(all_counts, axis=0), axis=0)
        (
            det_class_total,
            det_class_c1,
            det_class_c2,
            det_class_c3,
            det_class_base_total,
            det_class_base_c1,
            det_class_base_c2,
            det_class_base_c3,
            det_class_novel_total,
            det_class_novel_c1,
            det_class_novel_c2,
            det_class_novel_c3,
            det_class_rec_total,
            det_class_rec_c1,
            det_class_rec_c2,
            det_class_rec_c3,
            det_class_rec_base_total,
            det_class_rec_base_c1,
            det_class_rec_base_c2,
            det_class_rec_base_c3,
            det_class_rec_novel_total,
            det_class_rec_novel_c1,
            det_class_rec_novel_c2,
            det_class_rec_novel_c3,
            class_total,
            class_c1,
            class_c2,
            class_c3,
            class_base_total,
            class_base_c1,
            class_base_c2,
            class_base_c3,
            class_novel_total,
            class_novel_c1,
            class_novel_c2,
            class_novel_c3,
            class_rec_total,
            class_rec_c1,
            class_rec_c2,
            class_rec_c3,
            class_rec_base_total,
            class_rec_base_c1,
            class_rec_base_c2,
            class_rec_base_c3,
            class_rec_novel_total,
            class_rec_novel_c1,
            class_rec_novel_c2,
            class_rec_novel_c3,
            cluster_total,
            cluster_c1,
            cluster_c2,
            cluster_c3,
            cluster_base_total,
            cluster_base_c1,
            cluster_base_c2,
            cluster_base_c3,
            cluster_novel_total,
            cluster_novel_c1,
            cluster_novel_c2,
            cluster_novel_c3,
        ) = totals.tolist()
        per_attr_total = np.sum(np.stack([x[0] for x in all_per_attr], axis=0), axis=0)
        per_attr_c1 = np.sum(np.stack([x[1] for x in all_per_attr], axis=0), axis=0)
        per_attr_c2 = np.sum(np.stack([x[2] for x in all_per_attr], axis=0), axis=0)
        per_attr_c3 = np.sum(np.stack([x[3] for x in all_per_attr], axis=0), axis=0)

        if det_class_total > 0:
            metrics["det_class_top1_acc"] = float(det_class_c1 / max(1, det_class_total))
            metrics["det_class_top2_acc"] = float(det_class_c2 / max(1, det_class_total))
            metrics["det_class_top3_acc"] = float(det_class_c3 / max(1, det_class_total))
        if det_class_rec_total > 0:
            metrics["det_class_top1_rec"] = float(det_class_rec_c1 / max(1, det_class_rec_total))
            metrics["det_class_top2_rec"] = float(det_class_rec_c2 / max(1, det_class_rec_total))
            metrics["det_class_top3_rec"] = float(det_class_rec_c3 / max(1, det_class_rec_total))
        if det_class_base_total > 0:
            metrics["det_class_base_top1_acc"] = float(
                det_class_base_c1 / max(1, det_class_base_total)
            )
            metrics["det_class_base_top2_acc"] = float(
                det_class_base_c2 / max(1, det_class_base_total)
            )
            metrics["det_class_base_top3_acc"] = float(
                det_class_base_c3 / max(1, det_class_base_total)
            )
        if det_class_rec_base_total > 0:
            metrics["det_class_base_top1_rec"] = float(
                det_class_rec_base_c1 / max(1, det_class_rec_base_total)
            )
            metrics["det_class_base_top2_rec"] = float(
                det_class_rec_base_c2 / max(1, det_class_rec_base_total)
            )
            metrics["det_class_base_top3_rec"] = float(
                det_class_rec_base_c3 / max(1, det_class_rec_base_total)
            )
        if det_class_novel_total > 0:
            metrics["det_class_novel_top1_acc"] = float(
                det_class_novel_c1 / max(1, det_class_novel_total)
            )
            metrics["det_class_novel_top2_acc"] = float(
                det_class_novel_c2 / max(1, det_class_novel_total)
            )
            metrics["det_class_novel_top3_acc"] = float(
                det_class_novel_c3 / max(1, det_class_novel_total)
            )
        if det_class_rec_novel_total > 0:
            metrics["det_class_novel_top1_rec"] = float(
                det_class_rec_novel_c1 / max(1, det_class_rec_novel_total)
            )
            metrics["det_class_novel_top2_rec"] = float(
                det_class_rec_novel_c2 / max(1, det_class_rec_novel_total)
            )
            metrics["det_class_novel_top3_rec"] = float(
                det_class_rec_novel_c3 / max(1, det_class_rec_novel_total)
            )

        if class_total > 0:
            metrics["attr_class_top1_acc"] = float(class_c1 / max(1, class_total))
            metrics["attr_class_top2_acc"] = float(class_c2 / max(1, class_total))
            metrics["attr_class_top3_acc"] = float(class_c3 / max(1, class_total))
        if class_rec_total > 0:
            metrics["attr_class_top1_rec"] = float(class_rec_c1 / max(1, class_rec_total))
            metrics["attr_class_top2_rec"] = float(class_rec_c2 / max(1, class_rec_total))
            metrics["attr_class_top3_rec"] = float(class_rec_c3 / max(1, class_rec_total))
        if class_base_total > 0:
            metrics["attr_class_base_top1_acc"] = float(
                class_base_c1 / max(1, class_base_total)
            )
            metrics["attr_class_base_top2_acc"] = float(
                class_base_c2 / max(1, class_base_total)
            )
            metrics["attr_class_base_top3_acc"] = float(
                class_base_c3 / max(1, class_base_total)
            )
        if class_rec_base_total > 0:
            metrics["attr_class_base_top1_rec"] = float(
                class_rec_base_c1 / max(1, class_rec_base_total)
            )
            metrics["attr_class_base_top2_rec"] = float(
                class_rec_base_c2 / max(1, class_rec_base_total)
            )
            metrics["attr_class_base_top3_rec"] = float(
                class_rec_base_c3 / max(1, class_rec_base_total)
            )
        if class_novel_total > 0:
            metrics["attr_class_novel_top1_acc"] = float(
                class_novel_c1 / max(1, class_novel_total)
            )
            metrics["attr_class_novel_top2_acc"] = float(
                class_novel_c2 / max(1, class_novel_total)
            )
            metrics["attr_class_novel_top3_acc"] = float(
                class_novel_c3 / max(1, class_novel_total)
            )
        if class_rec_novel_total > 0:
            metrics["attr_class_novel_top1_rec"] = float(
                class_rec_novel_c1 / max(1, class_rec_novel_total)
            )
            metrics["attr_class_novel_top2_rec"] = float(
                class_rec_novel_c2 / max(1, class_rec_novel_total)
            )
            metrics["attr_class_novel_top3_rec"] = float(
                class_rec_novel_c3 / max(1, class_rec_novel_total)
            )

        if cluster_total > 0:
            metrics["attr_cluster_top1_acc"] = float(cluster_c1 / max(1, cluster_total))
            metrics["attr_cluster_top2_acc"] = float(cluster_c2 / max(1, cluster_total))
            metrics["attr_cluster_top3_acc"] = float(cluster_c3 / max(1, cluster_total))
        if cluster_base_total > 0:
            metrics["attr_cluster_base_top1_acc"] = float(
                cluster_base_c1 / max(1, cluster_base_total)
            )
            metrics["attr_cluster_base_top2_acc"] = float(
                cluster_base_c2 / max(1, cluster_base_total)
            )
            metrics["attr_cluster_base_top3_acc"] = float(
                cluster_base_c3 / max(1, cluster_base_total)
            )
        if cluster_novel_total > 0:
            metrics["attr_cluster_novel_top1_acc"] = float(
                cluster_novel_c1 / max(1, cluster_novel_total)
            )
            metrics["attr_cluster_novel_top2_acc"] = float(
                cluster_novel_c2 / max(1, cluster_novel_total)
            )
            metrics["attr_cluster_novel_top3_acc"] = float(
                cluster_novel_c3 / max(1, cluster_novel_total)
            )

        valid_mask = per_attr_total > 0
        if np.any(valid_mask):
            acc_t1 = np.zeros_like(per_attr_c1, dtype=np.float64)
            acc_t2 = np.zeros_like(per_attr_c2, dtype=np.float64)
            acc_t3 = np.zeros_like(per_attr_c3, dtype=np.float64)
            acc_t1[valid_mask] = per_attr_c1[valid_mask] / per_attr_total[valid_mask]
            acc_t2[valid_mask] = per_attr_c2[valid_mask] / per_attr_total[valid_mask]
            acc_t3[valid_mask] = per_attr_c3[valid_mask] / per_attr_total[valid_mask]
            for idx in range(self.num_clusters):
                if per_attr_total[idx] <= 0:
                    continue
                metrics[f"ac{idx}_t1"] = float(acc_t1[idx])
                metrics[f"ac{idx}_t2"] = float(acc_t2[idx])
                metrics[f"ac{idx}_t3"] = float(acc_t3[idx])

        return {"attr": metrics} if metrics else {}
