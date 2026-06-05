# import torch
# torch.autograd.set_detect_anomaly(True)

import os
import random
import numpy as np
import torch
from detectron2.utils import comm
from detectron2.engine import launch
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from defrcn.config import get_cfg, set_global_cfg
from defrcn.evaluation import DatasetEvaluators, verify_results
from defrcn.engine import Trainer, TwoSteamTrainer, default_argument_parser, default_setup


def set_deterministic(seed: int, rank: int = 0) -> None:
    """
    固定随机种子并开启确定性算法，尽力保证可复现。
    如需禁用，请在 main 中注释该函数调用。
    """
    base_seed = int(seed)
    rank_seed = base_seed + int(rank)
    os.environ["PYTHONHASHSEED"] = str(rank_seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    # Use cfg.SEED when provided; FIXED_SEED keeps old scripts reproducible.
    seed = cfg.SEED if int(cfg.SEED) >= 0 else int(os.environ.get("FIXED_SEED", 42))
    set_deterministic(seed=seed, rank=comm.get_rank())

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    if cfg.DATASETS.TWO_STREAM:
        trainer = TwoSteamTrainer(cfg)
    else:
        trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
