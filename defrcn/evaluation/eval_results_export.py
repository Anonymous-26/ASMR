import os
import itertools
from typing import Any, Dict, List, Optional

import torch
from detectron2.utils import comm
from detectron2.data import MetadataCatalog
from detectron2.utils.file_io import PathManager

from .evaluator import DatasetEvaluator


class EvalResultsExporter(DatasetEvaluator):
    """
    Export per-GT RoI inference results for offline analysis when GT boxes are used.
    """

    def __init__(self, cfg, dataset_name: str, output_dir: Optional[str] = None):
        self._cpu_device = torch.device("cpu")
        self._distributed = comm.get_world_size() > 1
        self._dataset_name = dataset_name
        self._output_dir = output_dir if output_dir else str(cfg.OUTPUT_DIR)
        self._eval_results_path = str(cfg.MODEL.ATTRIBUTE.EVAL_RESULTS_PATH)
        self._num_classes = int(cfg.MODEL.ROI_HEADS.NUM_CLASSES)
        self._attr_enabled = bool(cfg.MODEL.ATTRIBUTE.ENABLED)
        self._eval_use_gt = bool(cfg.MODEL.ATTRIBUTE.EVAL_USE_GT_BOXES)
        self._class_names = None
        try:
            meta = MetadataCatalog.get(dataset_name)
            if hasattr(meta, "thing_classes"):
                self._class_names = list(meta.thing_classes)
        except Exception:
            self._class_names = None
        self._predictions: List[Dict[str, Any]] = []
        self._shared_fields: Dict[str, Any] = {}

    def reset(self):
        self._predictions = []
        self._shared_fields = {}

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            if "instances" not in output:
                continue
            def _clone_tensor(x):
                return x.detach().to(self._cpu_device).clone()
            instances = output["instances"].to(self._cpu_device)
            gt_instances = input.get("instances")
            if gt_instances is not None:
                gt_instances = gt_instances.to(self._cpu_device)
            record: Dict[str, Any] = {
                "image_id": input.get("image_id", None),
                "file_name": input.get("file_name", None),
                "image_path": input.get("image_path", input.get("file_name", None)),
                "height": input.get("height", instances.image_size[0]),
                "width": input.get("width", instances.image_size[1]),
                "image_shape": instances.image_size,
            }

            # GT fields (from dataset inputs)
            if gt_instances is not None and gt_instances.has("gt_boxes"):
                record["gt_boxes"] = _clone_tensor(gt_instances.gt_boxes.tensor)
            if gt_instances is not None and gt_instances.has("gt_classes"):
                record["gt_classes"] = _clone_tensor(gt_instances.gt_classes)

            # det branch
            if instances.has("pred_boxes"):
                record["pred_boxes"] = _clone_tensor(instances.pred_boxes.tensor)
            if instances.has("scores"):
                record["scores"] = _clone_tensor(instances.scores)
            if instances.has("pred_classes"):
                record["pred_classes"] = _clone_tensor(instances.pred_classes)
            if instances.has("pred_class_scores"):
                record["pred_class_scores"] = _clone_tensor(instances.pred_class_scores)
            if instances.has("pred_class_logits"):
                record["pred_class_logits"] = _clone_tensor(instances.pred_class_logits)

            # attr branch
            if instances.has("attr_probs"):
                record["attr_probs"] = _clone_tensor(instances.attr_probs)
            if instances.has("attr_cluster_probs"):
                record["attr_cluster_probs"] = _clone_tensor(instances.attr_cluster_probs)
            if instances.has("attr_cluster_scores"):
                record["attr_cluster_scores"] = _clone_tensor(instances.attr_cluster_scores)
            if instances.has("attr_scores"):
                record["attr_scores"] = _clone_tensor(instances.attr_scores)
                record["attr_logits"] = _clone_tensor(instances.attr_scores)
            if instances.has("attr_embeddings"):
                record["attr_embeddings"] = _clone_tensor(instances.attr_embeddings)
            if instances.has("det_class_probs"):
                record["det_class_probs"] = _clone_tensor(instances.det_class_probs)

            # optional cached fields
            if instances.has("roi_feature"):
                record["roi_feature"] = _clone_tensor(instances.roi_feature)
            if not self._shared_fields:
                for key in [
                    "class_prototypes",
                    "cluster_embeddings",
                    "cluster_incidence",
                ]:
                    value = output.get(key)
                    if torch.is_tensor(value):
                        self._shared_fields[key] = _clone_tensor(value)

            self._predictions.append(record)

    def _resolve_output_path(self) -> str:
        if not self._eval_results_path:
            return ""
        if os.path.isabs(self._eval_results_path):
            return self._eval_results_path
        return os.path.join(self._output_dir, self._eval_results_path)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))
            shared_fields = comm.gather(self._shared_fields, dst=0)
            if not comm.is_main_process():
                return {}
            merged_shared: Dict[str, Any] = {}
            for item in shared_fields:
                if item:
                    merged_shared = item
                    break
            self._shared_fields = merged_shared

        if not self._predictions:
            return {}

        output_path = self._resolve_output_path()
        if output_path:
            out_dir = os.path.dirname(output_path)
            if out_dir:
                PathManager.mkdirs(out_dir)
            payload = {
                "meta": {
                    "version": 1,
                    "dataset_name": self._dataset_name,
                    "num_classes": self._num_classes,
                    "class_names": self._class_names,
                    "attr_enabled": self._attr_enabled,
                    "eval_use_gt_boxes": self._eval_use_gt,
                },
                "shared": self._shared_fields,
                "images": self._predictions,
            }
            with PathManager.open(output_path, "wb") as f:
                torch.save(payload, f)

        return {}
