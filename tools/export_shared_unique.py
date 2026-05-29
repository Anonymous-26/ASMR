#!/usr/bin/env python3
'''
NAME=Av3P_wProjs_woSU
DIR=checkpoints/voc_attrv2_params_1/${NAME}/mfdc_gfsod_novel1/tfa-like/1shot_seed0
python tools/export_shared_unique.py --config-file ${DIR}/config.yaml \
 --weights ${DIR}/model_final.pth \
 --plot-dir visual_results/v2_${NAME}_1shot

'''


from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import os

# Add repo root to path so submodules can be found
sys.path.insert(0, str(Path(__file__).parent.parent))

from defrcn.config import get_cfg
from defrcn.modeling.roi_heads.attr_modules import (
    AttributeHypergraphReasoner,
    AttributePrototypeBank,
)
from defrcn.modeling.roi_heads.attr_shared_unique import (
    build_neighbors_by_confusion,
    build_neighbors_by_prototypes,
    mine_shared_unique,
)


def _load_confusion(path: Optional[str]) -> Optional[torch.Tensor]:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    data = np.load(str(file_path), allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        if "confusion" in data:
            matrix = data["confusion"]
        else:
            first_key = list(data.keys())[0]
            matrix = data[first_key]
    else:
        matrix = data
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        return None
    return torch.from_numpy(matrix)


def _load_super_attr_meta(path: str) -> Tuple[List[str], List[str]]:
    file_path = Path(path)
    if not file_path.exists():
        return [], []
    with np.load(str(file_path), allow_pickle=True) as data:
        cluster_names = data.get("cluster_names", None)
        class_names = data.get("class_names", None)
    if cluster_names is None:
        cluster_names_list: List[str] = []
    else:
        cluster_names_list = [str(name) for name in cluster_names.tolist()]
    if class_names is None:
        class_names_list: List[str] = []
    else:
        class_names_list = [str(name) for name in class_names.tolist()]
    return cluster_names_list, class_names_list


def _load_reasoner_weights(
    reasoner: AttributeHypergraphReasoner, weights_path: Optional[str]
) -> None:
    if not weights_path:
        return
    ckpt_path = Path(weights_path)
    if not ckpt_path.exists():
        return
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    prefix = "roi_heads.hypergraph_reasoner."
    alt_prefix = "module.roi_heads.hypergraph_reasoner."
    filtered = {}
    for key, val in state_dict.items():
        if key.startswith(prefix):
            filtered[key[len(prefix) :]] = val
        elif key.startswith(alt_prefix):
            filtered[key[len(alt_prefix) :]] = val
    if filtered:
        reasoner.load_state_dict(filtered, strict=False)


def _build_class_attribute_graph(
    attr_to_classes: Dict[str, List[str]]
) -> nx.Graph:
    graph = nx.Graph()
    for attr, classes in attr_to_classes.items():
        graph.add_node(attr, type="attribute")
        for cls in classes:
            graph.add_node(cls, type="class")
            graph.add_edge(cls, attr)
    return graph


def _show_network(
    graph: nx.Graph,
    save_path: Path,
    title: str,
    attr_color: str = "lightcoral",
    class_color: str = "skyblue",
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(28, 20))
    node_colors = [
        class_color if graph.nodes[n]["type"] == "class" else attr_color
        for n in graph.nodes()
    ]
    node_sizes = [
        2400 if graph.nodes[n]["type"] == "class" else 1200
        for n in graph.nodes()
    ]
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=node_sizes,
        node_color=node_colors,
        font_size=8,
    )
    plt.title(title)
    plt.savefig(save_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export shared/unique attribute clusters per class."
    )
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--plot-dir", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--weights", default="")
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--t-shared", type=int, default=None)
    parser.add_argument("--top-percent", type=float, default=None)
    parser.add_argument("--neighbor-topn", type=int, default=None)
    parser.add_argument("--neighbor-source", default=None, choices=["prototype", "confusion"])
    parser.add_argument("--confusion-path", default=None)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.output == "" and len(args.plot_dir)>0:
        args.output = os.path.join(args.plot_dir, 'result.txt')
        
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    attr_cfg = cfg.MODEL.ATTRIBUTE
    su_cfg = attr_cfg.SHARED_UNIQUE
    device = torch.device(args.device)
    override_names = list(attr_cfg.CLASS_NAMES) if attr_cfg.CLASS_NAMES else None
    cluster_names, stored_class_names = _load_super_attr_meta(attr_cfg.SUPER_ATTR_PATH)

    prototype_bank = AttributePrototypeBank(
        super_attr_path=attr_cfg.SUPER_ATTR_PATH,
        class_names=override_names,
        background_name=attr_cfg.BACKGROUND_CLASS,
    )
    cluster_state = prototype_bank.get_cluster_state(device)
    cluster_embeddings = cluster_state["embeddings"]
    incidence = cluster_state["incidence"]
    similarity = cluster_state["similarity"]
    class_names = prototype_bank.class_names or stored_class_names

    reasoner = None
    if attr_cfg.HGNN.ENABLED:
        reasoner = AttributeHypergraphReasoner(
            feature_dim=attr_cfg.EMBEDDING_DIM,
            hidden_dim=attr_cfg.HGNN.HIDDEN_DIM,
            num_layers=attr_cfg.HGNN.NUM_LAYERS,
            similarity_weight=attr_cfg.HGNN.SIMILARITY_WEIGHT,
        )
        _load_reasoner_weights(reasoner, args.weights)
        cluster_embeddings = reasoner(cluster_embeddings, incidence, similarity)

    class_prototypes = prototype_bank.build_class_prototypes(
        cluster_embeddings, device
    )
    num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
    if class_prototypes.shape[0] > num_classes:
        class_prototypes = class_prototypes[:num_classes]
    if incidence.shape[1] > num_classes:
        incidence = incidence[:, :num_classes]

    if not cluster_names:
        cluster_names = [f"cluster_{i}" for i in range(int(incidence.shape[0]))]
    if not class_names:
        class_names = [f"class_{i}" for i in range(int(incidence.shape[1]))]

    delta = args.delta if args.delta is not None else su_cfg.DELTA
    epsilon = args.epsilon if args.epsilon is not None else su_cfg.EPSILON
    t_shared = args.t_shared if args.t_shared is not None else su_cfg.T_SHARED
    top_percent = (
        args.top_percent if args.top_percent is not None else su_cfg.TOP_PERCENT
    )
    neighbor_topn = (
        args.neighbor_topn if args.neighbor_topn is not None else su_cfg.NEIGHBOR_TOPN
    )
    neighbor_source = (
        args.neighbor_source if args.neighbor_source is not None else su_cfg.NEIGHBOR_SOURCE
    )
    confusion_path = (
        args.confusion_path if args.confusion_path is not None else su_cfg.CONFUSION_PATH
    )

    if neighbor_source == "confusion":
        confusion = _load_confusion(confusion_path)
        if confusion is None:
            neighbors = build_neighbors_by_prototypes(
                class_prototypes, neighbor_topn
            )
        else:
            neighbors = build_neighbors_by_confusion(
                confusion.to(device), neighbor_topn
            )
    else:
        neighbors = build_neighbors_by_prototypes(class_prototypes, neighbor_topn)

    result = mine_shared_unique(
        incidence,
        float(delta),
        float(epsilon),
        neighbors,
        t_shared=int(t_shared) if t_shared and t_shared > 0 else None,
        top_percent=float(top_percent) if not t_shared or t_shared <= 0 else None,
    )

    unique_map: Dict[str, List[int]] = {}
    unique_name_map: Dict[str, List[str]] = {}
    for idx, name in enumerate(class_names[: result.u_mask.shape[0]]):
        unique_map[name] = result.k_uniq[idx].tolist()
        unique_name_map[name] = [
            cluster_names[int(k)] for k in result.k_uniq[idx].tolist()
        ]

    payload = {
        "shared_clusters": result.k_shared.tolist(),
        "shared_cluster_names": [cluster_names[int(k)] for k in result.k_shared.tolist()],
        "unique_clusters": unique_map,
        "unique_cluster_names": unique_name_map,
        "settings": {
            "delta": float(delta),
            "epsilon": float(epsilon),
            "t_shared": int(t_shared) if t_shared is not None else None,
            "top_percent": float(top_percent) if top_percent is not None else None,
            "neighbor_topn": int(neighbor_topn),
            "neighbor_source": str(neighbor_source),
            "confusion_path": confusion_path,
            "weights": args.weights,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    plot_dir = Path(args.plot_dir) if args.plot_dir else out_path.parent / "shared_unique_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    strong = incidence > float(delta)

    shared_attr_to_classes: Dict[str, List[str]] = {}
    shared_set = set(result.k_shared.tolist())
    for k in shared_set:
        attr_name = cluster_names[int(k)]
        class_list = [
            class_names[c]
            for c in range(strong.shape[1])
            if bool(strong[int(k), c])
        ]
        if class_list:
            shared_attr_to_classes[attr_name] = class_list

    unique_attr_to_classes: Dict[str, List[str]] = {}
    for c_idx, cname in enumerate(class_names[: result.u_mask.shape[0]]):
        for k in result.k_uniq[c_idx].tolist():
            attr_name = cluster_names[int(k)]
            unique_attr_to_classes.setdefault(attr_name, []).append(cname)

    if shared_attr_to_classes:
        graph = _build_class_attribute_graph(shared_attr_to_classes)
        _show_network(
            graph,
            plot_dir / "shared_attribute_graph.png",
            "Shared Attribute Graph",
            attr_color="lightcoral",
            class_color="skyblue",
        )

    if unique_attr_to_classes:
        graph = _build_class_attribute_graph(unique_attr_to_classes)
        _show_network(
            graph,
            plot_dir / "unique_attribute_graph.png",
            "Unique Attribute Graph",
            attr_color="lightgreen",
            class_color="skyblue",
        )


if __name__ == "__main__":
    main()
