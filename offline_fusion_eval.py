#!/usr/bin/env python3
import argparse
import json
from copy import deepcopy
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import logging
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, Instances
from defrcn.evaluation import COCOEvaluator, PascalVOCDetectionEvaluator

"""
python offline_fusion_eval.py \
    --input checkpoints2/voc_best_bg/bg2/mfdc_gfsod_novel1/tfa-like/1shot_seed0/eval_results.pt \
    --iou 0.5 \
    --output output_offline_metrics.json
"""

ALPHA = 0.0
TAU_SAFE = 0.2
TAU_ATTR = 0.9
BASE_IDS = set(range(0, 15))
NOVEL_IDS = set(range(15, 20))

LOGGER = logging.getLogger("offline_fusion_eval")


def load_results(path: str) -> Dict:
    return torch.load(path, map_location="cpu")


def ensure_dataset_registered(dataset_name: str) -> None:
    """
    Ensure detectron2 DatasetCatalog/MetadataCatalog are populated for the dataset.
    This mirrors the registration side effects that happen in train/eval entrypoints.
    """
    if dataset_name in DatasetCatalog.list():
        return
    from defrcn.data import builtin as _builtin  # noqa: F401
    if dataset_name in DatasetCatalog.list():
        return
    raise ValueError(
        f"Dataset '{dataset_name}' is not registered. "
        "Make sure it is defined in defrcn/data/builtin.py or pass --dataset correctly."
    )


def _softmax_from_probs(probs: torch.Tensor, temp: float = 1.0, eps: float = 1e-12) -> torch.Tensor:
    if temp <= 0:
        return probs
    logits = (probs + eps).log() / float(temp)
    return F.softmax(logits, dim=-1)


def _apply_novel_logit_shift(
    probs: torch.Tensor,
    novel_ids: set,
    shift: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    if shift == 0.0 or probs.numel() == 0:
        return probs
    logits = (probs + eps).log()
    if novel_ids:
        novel = torch.as_tensor(sorted(novel_ids), device=logits.device, dtype=torch.long)
        novel = novel[novel < logits.shape[1]]
        if novel.numel() > 0:
            logits[:, novel] = logits[:, novel] + float(shift)
    return F.softmax(logits, dim=-1)


def fusion_calculate(candidate_ids, p_det, p_attr, alpha):
    best_score = -1.0
    best_label = -1
    for cid in candidate_ids:
        score = (1.0 - alpha) * float(p_det[cid]) + alpha * float(p_attr[cid])
        if score > best_score:
            best_score = score
            best_label = int(cid)
    return best_label, best_score


def fuse_record(
    record: Dict,
    num_classes: int,
    alpha: float,
    tau_safe: float,
    tau_attr: float,
    topk: int,
    mode: str,
    rescue: bool,
    attr_temp: float,
    attr_logit_shift: float,
    base_ids: set,
    novel_ids: set,
) -> Dict:
    """
    Asymmetric Bi-directional Correction (offline).

    mode:
      - none: keep original predictions
      - label: change labels only; keep original detector scores (AP ranking)
      - score: change labels and scores using fusion score
    """
    if "pred_class_logits" not in record or "attr_probs" not in record:
        return {
            "pred_boxes": record.get("pred_boxes"),
            "pred_classes": record.get("pred_classes"),
            "scores": record.get("scores"),
            "pred_class_scores": record.get("pred_class_scores"),
        }

    det_logits = record["pred_class_logits"]
    p_det = F.softmax(det_logits, dim=-1)
    p_attr = record["attr_probs"]

    # align class dims (keep foreground only)
    num_classes = min(num_classes, p_det.shape[1], p_attr.shape[1])
    p_det = p_det[:, :num_classes]
    p_attr = p_attr[:, :num_classes]

    # calibrate attr probabilities
    p_attr = _softmax_from_probs(p_attr, temp=attr_temp)
    p_attr = _apply_novel_logit_shift(p_attr, novel_ids, shift=attr_logit_shift)

    orig_scores = record.get("scores")

    new_pred_classes = []
    new_scores = []

    for i in range(p_det.shape[0]):
        curr_p_det = p_det[i]
        curr_p_attr = p_attr[i]

        sorted_indices = torch.argsort(curr_p_det, descending=True)
        det_top1_id = int(sorted_indices[0].item())
        det_top2_id = int(sorted_indices[1].item())

        det_score_top1 = float(curr_p_det[det_top1_id].item())
        det_score_top2 = float(curr_p_det[det_top2_id].item())
        det_margin = det_score_top1 - det_score_top2

        attr_top1_id = int(torch.argmax(curr_p_attr).item())
        attr_conf = float(curr_p_attr[attr_top1_id].item())

        final_label = det_top1_id
        final_score = det_score_top1

        if det_top1_id in novel_ids:
            candidates = sorted_indices[: max(1, int(topk))].tolist()
            final_label, final_score = fusion_calculate(
                candidates, curr_p_det, curr_p_attr, alpha
            )
        elif det_top1_id in base_ids:
            needs_rescue = (
                rescue
                and (det_margin < tau_safe)
                and (attr_top1_id in novel_ids)
                and (attr_conf > tau_attr)
            )
            if needs_rescue:
                candidates = set(sorted_indices[: max(1, int(topk))].tolist())
                if attr_top1_id not in candidates:
                    candidates.add(attr_top1_id)
                final_label, final_score = fusion_calculate(
                    list(candidates), curr_p_det, curr_p_attr, alpha
                )
            else:
                final_label = det_top1_id
                final_score = det_score_top1

        new_pred_classes.append(final_label)
        new_scores.append(final_score)

    device = det_logits.device
    if mode == "none":
        return {
            "pred_boxes": record.get("pred_boxes"),
            "pred_classes": record.get("pred_classes"),
            "scores": record.get("scores"),
            "pred_class_scores": record.get("pred_class_scores"),
        }
    if mode == "label":
        keep_scores = orig_scores if orig_scores is not None else torch.tensor(new_scores, device=device)
        return {
            "pred_boxes": record.get("pred_boxes"),
            "pred_classes": torch.tensor(new_pred_classes, device=device),
            "scores": keep_scores,
            "pred_class_scores": record.get("pred_class_scores"),
        }
    return {
        "pred_boxes": record.get("pred_boxes"),
        "pred_classes": torch.tensor(new_pred_classes, device=device),
        "scores": torch.tensor(new_scores, device=device),
        "pred_class_scores": record.get("pred_class_scores"),
    }


def pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]))
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-12)


def compute_acc(records: List[Dict], iou_thresh: float = 0.5) -> Dict:
    total_gt = 0
    matched_gt = 0
    correct = 0
    for rec in records:
        gt_boxes = rec.get("gt_boxes")
        gt_classes = rec.get("gt_classes")
        pred_boxes = rec.get("fused_boxes")
        pred_classes = rec.get("fused_classes")
        pred_scores = rec.get("fused_scores")
        if gt_boxes is None or gt_classes is None or pred_boxes is None or pred_classes is None:
            continue
        if gt_boxes.numel() == 0:
            continue
        total_gt += int(gt_boxes.shape[0])
        if pred_boxes.numel() == 0:
            continue
        ious = pairwise_iou(gt_boxes, pred_boxes)
        used = torch.zeros(pred_boxes.shape[0], dtype=torch.bool)
        for gt_idx in range(gt_boxes.shape[0]):
            iou_row = ious[gt_idx]
            cand = (iou_row >= iou_thresh) & (~used)
            if not torch.any(cand):
                continue
            cand_idx = torch.nonzero(cand, as_tuple=False).squeeze(1)
            best_idx = cand_idx[iou_row[cand_idx].argmax()]
            used[best_idx] = True
            matched_gt += 1
            if int(pred_classes[best_idx]) == int(gt_classes[gt_idx]):
                correct += 1
    acc = correct / matched_gt if matched_gt > 0 else 0.0
    rec = matched_gt / total_gt if total_gt > 0 else 0.0
    return {"acc": acc, "recall": rec, "matched_gt": matched_gt, "total_gt": total_gt}


def roc_curve(scores: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    if scores.numel() == 0:
        return torch.tensor([]), torch.tensor([]), 0.0
    order = torch.argsort(scores, descending=True)
    scores = scores[order]
    labels = labels[order]
    tp = (labels == 1).float().cumsum(0)
    fp = (labels == 0).float().cumsum(0)
    tpr = tp / tp[-1].clamp(min=1.0)
    fpr = fp / fp[-1].clamp(min=1.0)
    auc = torch.trapz(tpr, fpr).item()
    return fpr, tpr, auc


def compute_roc(records: List[Dict], iou_thresh: float = 0.5) -> Dict:
    all_scores = []
    all_labels = []
    for rec in records:
        gt_boxes = rec.get("gt_boxes")
        gt_classes = rec.get("gt_classes")
        pred_boxes = rec.get("fused_boxes")
        pred_classes = rec.get("fused_classes")
        pred_scores = rec.get("fused_scores")
        if gt_boxes is None or gt_classes is None or pred_boxes is None or pred_classes is None:
            continue
        if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
            continue
        ious = pairwise_iou(gt_boxes, pred_boxes)
        used = torch.zeros(pred_boxes.shape[0], dtype=torch.bool)
        for gt_idx in range(gt_boxes.shape[0]):
            iou_row = ious[gt_idx]
            cand = (iou_row >= iou_thresh) & (~used)
            if not torch.any(cand):
                continue
            cand_idx = torch.nonzero(cand, as_tuple=False).squeeze(1)
            best_idx = cand_idx[iou_row[cand_idx].argmax()]
            used[best_idx] = True
            all_scores.append(pred_scores[best_idx])
            all_labels.append(
                1.0 if int(pred_classes[best_idx]) == int(gt_classes[gt_idx]) else 0.0
            )
    if not all_scores:
        return {"auc": 0.0, "num_samples": 0}
    scores = torch.stack(all_scores)
    labels = torch.tensor(all_labels, dtype=torch.float32)
    fpr, tpr, auc = roc_curve(scores, labels)
    return {"auc": auc, "num_samples": int(labels.numel())}


def compute_ap50_official(
    records: List[Dict],
    dataset_name: str,
    output_dir: str,
) -> Dict:
    meta = MetadataCatalog.get(dataset_name)
    evaluator_type = getattr(meta, "name", "")
    if "coco" in evaluator_type:
        evaluator = COCOEvaluator(dataset_name, False, output_dir)
    elif "voc" in evaluator_type:
        evaluator = PascalVOCDetectionEvaluator(dataset_name)
    else:
        raise ValueError(f"Unsupported evaluator_type '{evaluator_type}' for dataset '{dataset_name}'.")

    evaluator.reset()
    for rec in records:
        height = int(rec.get("height", rec.get("image_shape", (0, 0))[0]))
        width = int(rec.get("width", rec.get("image_shape", (0, 0))[1]))
        image_size = (height, width)

        pred_boxes = rec.get("fused_boxes")
        pred_classes = rec.get("fused_classes")
        pred_scores = rec.get("fused_scores")
        if pred_boxes is None or pred_classes is None or pred_scores is None:
            continue

        inst = Instances(image_size)
        inst.pred_boxes = Boxes(pred_boxes)
        inst.pred_classes = pred_classes
        inst.scores = pred_scores

        inputs = {
            "image_id": rec.get("image_id"),
            "file_name": rec.get("file_name"),
            "height": height,
            "width": width,
        }
        outputs = {"instances": inst}
        evaluator.process([inputs], [outputs])

    results = evaluator.evaluate()
    if results is None:
        return {"ap50": 0.0}
    if "bbox" in results and "AP50" in results["bbox"]:
        return {"ap50": float(results["bbox"]["AP50"]), "full": results}
    if "AP50" in results:
        return {"ap50": float(results["AP50"]), "full": results}
    return {"ap50": 0.0, "full": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to .pt results file")
    parser.add_argument("--output", default="", help="Optional JSON output path")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--dataset", default="", help="Dataset name override (optional)")
    parser.add_argument("--evaluator-output", default="output_offline_eval", help="Evaluator output dir")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Fusion weight alpha")
    parser.add_argument("--tau-safe", type=float, default=TAU_SAFE, help="Detector margin threshold")
    parser.add_argument("--tau-attr", type=float, default=TAU_ATTR, help="Attribute confidence threshold")
    parser.add_argument("--topk", type=int, default=3, help="Top-K candidate size")
    parser.add_argument("--fusion-mode", type=str, default="label", choices=("none", "label", "score"))
    parser.add_argument("--disable-rescue", action="store_true", help="Disable base->novel rescue")
    parser.add_argument("--attr-temp", type=float, default=1.0, help="Temperature scaling for attr probs")
    parser.add_argument("--attr-logit-shift", type=float, default=0.0, help="Logit shift for novel classes")
    args = parser.parse_args()

    payload = load_results(args.input)
    records = payload.get("images", [])
    dataset_name = args.dataset or payload.get("meta", {}).get("dataset_name", "")
    if not dataset_name:
        raise ValueError("Dataset name not found in payload; pass --dataset.")
    ensure_dataset_registered(dataset_name)
    num_classes = int(payload.get("meta", {}).get("num_classes", 0)) or 20
    fused_records = []

    if args.alpha <= 0 and args.fusion_mode != "none":
        LOGGER.warning("alpha<=0: forcing fusion_mode=none to match no-fusion baseline.")
        args.fusion_mode = "none"

    for rec in records:
        record = deepcopy(rec)
        fused = fuse_record(
            record,
            num_classes=num_classes,
            alpha=float(args.alpha),
            tau_safe=float(args.tau_safe),
            tau_attr=float(args.tau_attr),
            topk=int(args.topk),
            mode=str(args.fusion_mode),
            rescue=not bool(args.disable_rescue),
            attr_temp=float(args.attr_temp),
            attr_logit_shift=float(args.attr_logit_shift),
            base_ids=BASE_IDS,
            novel_ids=NOVEL_IDS,
        )
        out = dict(record)
        out["fused_boxes"] = fused.get("pred_boxes")
        out["fused_classes"] = fused.get("pred_classes")
        out["fused_scores"] = fused.get("scores")
        out["fused_class_scores"] = fused.get("pred_class_scores")
        out["orig_scores"] = record.get("scores")
        out["orig_classes"] = record.get("pred_classes")
        fused_records.append(out)

    acc = compute_acc(fused_records, iou_thresh=args.iou)
    roc = compute_roc(fused_records, iou_thresh=args.iou)
    ap = compute_ap50_official(fused_records, dataset_name=dataset_name, output_dir=args.evaluator_output)

    total_preds = 0
    label_changes = 0
    score_pairs = []
    for rec in fused_records:
        orig_cls = rec.get("orig_classes")
        fused_cls = rec.get("fused_classes")
        orig_scores = rec.get("orig_scores")
        fused_scores = rec.get("fused_scores")
        if orig_cls is not None and fused_cls is not None:
            n = min(orig_cls.numel(), fused_cls.numel())
            if n > 0:
                total_preds += int(n)
                label_changes += int((orig_cls[:n] != fused_cls[:n]).sum().item())
        if orig_scores is not None and fused_scores is not None:
            n = min(orig_scores.numel(), fused_scores.numel())
            if n > 1:
                score_pairs.append((orig_scores[:n].float(), fused_scores[:n].float()))

    score_corr = None
    if score_pairs:
        orig_all = torch.cat([p[0] for p in score_pairs], dim=0)
        fused_all = torch.cat([p[1] for p in score_pairs], dim=0)
        # Spearman correlation via rank transform
        def _rank(x: torch.Tensor) -> torch.Tensor:
            return torch.argsort(torch.argsort(x))
        r1 = _rank(orig_all).float()
        r2 = _rank(fused_all).float()
        r1 = r1 - r1.mean()
        r2 = r2 - r2.mean()
        denom = (r1.norm() * r2.norm()).clamp(min=1e-12)
        score_corr = float((r1 * r2).sum().item() / denom.item())

    metrics = {
        "acc": acc,
        "roc": roc,
        "ap50": ap,
        "label_change_ratio": (label_changes / max(1, total_preds)),
        "score_spearman": score_corr,
    }
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    else:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
