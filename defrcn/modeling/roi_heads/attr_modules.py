"""
Reusable attribute components (embedding head, prototype bank, HGNN reasoner)
shared by CommonalityROIHeads for modules M2–M5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from defrcn.modeling.meta_arch.gdl import decouple_layer


class AttributeEmbeddingHead(nn.Module):
    """
    属性预测头（M2）：将 ROI 特征投射到属性 embedding 空间。
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, pooled: bool, attr_with_center_norm, attr_backward_scale) -> None:
        super().__init__()
        self.pooled = pooled
        self.attr_backward_scale = attr_backward_scale
        self.attr_with_center_norm = attr_with_center_norm
        if pooled:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            reduced_channels = 128
            self.conv_bottleneck = nn.Sequential(
                nn.Conv2d(input_dim, reduced_channels, kernel_size=3, padding=0, bias=False),
                nn.BatchNorm2d(reduced_channels),
                nn.ReLU(inplace=True)
            )
            input_dim = reduced_channels * 2 * 2
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim),
            )
        if attr_with_center_norm:
            self.center_norm = nn.LayerNorm(output_dim, elementwise_affine=False)
            
    def forward(self, features: torch.Tensor):
        """
        前向计算：先扁平化，再过 MLP，并进行 L2 归一化。
        """
        features = decouple_layer(features, self.attr_backward_scale)
        if self.pooled:
            features = features.mean(dim=[2, 3]) 
            if features.dim() > 2:
                features = torch.flatten(features, start_dim=1)
            embeddings = self.net(features)
        else:
            features = self.conv_bottleneck(features)
            features = torch.flatten(features, start_dim=1) 
            embeddings = self.net(features)
        if self.attr_with_center_norm:
            embeddings = self.center_norm(embeddings)
        return F.normalize(embeddings, dim=-1)


class AttributeBilinearMatcher(nn.Module):
    """
    双线性匹配：用可学习投影增强 e 与 z_k 的匹配能力。

    输入：
      - embeddings: ROI embedding e，shape [N, D]
      - clusters: 簇原型 z_k，shape [K, D]
    输出：
      - projected embeddings / clusters，shape [N, D'] / [K, D']
    """

    def __init__(self, dim: int, proj_dim: int = 0, normalize: bool = True) -> None:
        super().__init__()
        out_dim = int(proj_dim) if proj_dim and proj_dim > 0 else int(dim)
        # out_dim = int(dim)
        self.q_proj = nn.Linear(dim, out_dim, bias=False)
        self.k_proj = nn.Linear(dim, out_dim, bias=False)
        self.normalize = bool(normalize)
        self._init_identity()

    def project_query(self, embeddings: torch.Tensor) -> torch.Tensor:
        proj = self.q_proj(embeddings)
        return F.normalize(proj, dim=-1) if self.normalize else proj

    def project_key(self, clusters: torch.Tensor) -> torch.Tensor:
        proj = self.k_proj(clusters)
        return F.normalize(proj, dim=-1) if self.normalize else proj

    def _init_identity(self) -> None:
        """
        初始化为恒等映射：投影初始等价于输入。
        """
        for module in [self.q_proj, self.k_proj]:
            if isinstance(module, nn.Linear):
                nn.init.eye_(module.weight)


class AttributeClusterProjector(nn.Module):
    """
    可学习簇原型投影：将 z_k 映射到更适合视觉区分的空间。

    输入：
      - clusters: 簇原型 z_k，shape [K, D]
    输出：
      - projected clusters，shape [K, D]
    """

    def __init__(self, dim: int, hidden_dim: int = 0) -> None:
        super().__init__()
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, dim),
            )
        else:
            self.net = nn.Linear(dim, dim)
        self._init_identity()

    def forward(self, clusters: torch.Tensor) -> torch.Tensor:
        return clusters + self.net(clusters)

    def _init_identity(self) -> None:
        """
        初始化为恒等映射：net 输出为 0，使投影初始等价于输入。
        """
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    

class AttributeHypergraphReasoner(nn.Module):
    """
    基于 HGNN 的简单超图推理层（M4）：在 attribute cluster 上迭代传播。
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_layers: int,
        similarity_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.similarity_weight = similarity_weight
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, feature_dim),
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        node_features: torch.Tensor,
        incidence: torch.Tensor,
        similarity: torch.Tensor,
    ):
        """
        构建超图邻接矩阵并应用每一层的线性变换+残差。
        """
        adjacency = self._build_adjacency(incidence, similarity, node_features.device)
        features = node_features
        for layer in self.layers:
            message = adjacency @ features
            residual = features + layer(message)
            features = F.normalize(residual.clone(), dim=-1)
        return features

    def _build_adjacency(
        self, incidence: torch.Tensor, similarity: torch.Tensor, device: torch.device
    ):
        """
        构造超图 adjacency，包括属性-类别 incidence 和超属性相似度两部分。
        """
        node_degree = incidence.sum(dim=1, keepdim=True).clamp(min=1.0)
        edge_degree = incidence.sum(dim=0, keepdim=True).clamp(min=1.0)
        dv_inv_sqrt = node_degree.pow(-0.5)
        de_inv = edge_degree.pow(-1.0).transpose(0, 1)
        hyper_adj = dv_inv_sqrt * (incidence @ (de_inv * incidence.t())) * dv_inv_sqrt.t()
        adjacency = hyper_adj + self.similarity_weight * similarity.to(device)
        adjacency = adjacency + torch.eye(adjacency.size(0), device=device)
        row_sum = adjacency.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return adjacency / row_sum


class AttributeIdentityReasoner(nn.Module):
    """
    消融：不做任何图传播，直接返回原始 cluster embeddings。
    """

    def forward(
        self,
        node_features: torch.Tensor,
        incidence: torch.Tensor,
        similarity: torch.Tensor,
    ) -> torch.Tensor:
        del incidence, similarity
        return node_features


class AttributeGraphReasoner(nn.Module):
    """
    消融：简单图卷积，仅基于 cluster-cluster similarity 做传播。
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, feature_dim),
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        node_features: torch.Tensor,
        incidence: torch.Tensor,
        similarity: torch.Tensor,
    ) -> torch.Tensor:
        del incidence
        adjacency = self._build_adjacency(similarity, node_features.device)
        features = node_features
        for layer in self.layers:
            message = adjacency @ features
            residual = features + layer(message)
            features = F.normalize(residual, dim=-1)
        return features

    def _build_adjacency(
        self, similarity: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        adjacency = similarity.to(device)
        adjacency = adjacency.clamp(min=0.0)
        adjacency = adjacency + torch.eye(adjacency.size(0), device=device)
        row_sum = adjacency.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return adjacency / row_sum


class AttributeIncidenceReasoner(nn.Module):
    """
    消融：仅基于 incidence 做无参数传播，不使用 similarity。
    """

    def __init__(self, num_layers: int = 1) -> None:
        super().__init__()
        self.num_layers = max(int(num_layers), 1)

    def forward(
        self,
        node_features: torch.Tensor,
        incidence: torch.Tensor,
        similarity: torch.Tensor,
    ) -> torch.Tensor:
        del similarity
        adjacency = self._build_adjacency(incidence, node_features.device)
        features = node_features
        for _ in range(self.num_layers):
            features = F.normalize(adjacency @ features, dim=-1)
        return features

    def _build_adjacency(
        self, incidence: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        node_degree = incidence.sum(dim=1, keepdim=True).clamp(min=1.0)
        edge_degree = incidence.sum(dim=0, keepdim=True).clamp(min=1.0)
        dv_inv_sqrt = node_degree.pow(-0.5)
        de_inv = edge_degree.pow(-1.0).transpose(0, 1)
        adjacency = dv_inv_sqrt * (incidence @ (de_inv * incidence.t())) * dv_inv_sqrt.t()
        adjacency = adjacency + torch.eye(adjacency.size(0), device=device)
        row_sum = adjacency.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return adjacency / row_sum


class AttributePrototypeBank:
    """
    属性原型库：加载聚类后的 super-attributes 文件并提供 tensor 接口。
    """

    def __init__(
        self,
        super_attr_path: Optional[str],
        class_names: Optional[List[str]],
        background_name: str,
    ) -> None:
        path = Path(super_attr_path)
        self.super_attr_path = path
        self.class_names_override = list(class_names) if class_names else None
        self.background_name = background_name

        self._loaded = False
        self.cluster_embeddings: Optional[torch.Tensor] = None
        self.cluster_similarity: Optional[torch.Tensor] = None
        self.cluster_category: Optional[torch.Tensor] = None
        self.cluster_shared_mask: Optional[torch.Tensor] = None
        self.cluster_unique_mask: Optional[torch.Tensor] = None
        self.class_names: List[str] = []

    def _load(self) -> None:
        if self._loaded:
            return
        if not self.super_attr_path.exists():
            raise FileNotFoundError(
                f"Super-attribute cache not found at {self.super_attr_path}."
            )
        with np.load(self.super_attr_path, allow_pickle=True) as data:
            self.cluster_embeddings = torch.from_numpy(
                data["cluster_embeddings"].astype(np.float32)
            )
            self.cluster_similarity = torch.from_numpy(
                data["cluster_similarity"].astype(np.float32)
            )
            cluster_category = torch.from_numpy(
                data["cluster_category_matrix"].astype(np.float32)
            )
            shared_mask = data.get("cluster_shared_mask")
            unique_mask = data.get("cluster_unique_mask")
            other_mask = data.get("cluster_other_mask")
            stored_names = [str(name) for name in data["class_names"].tolist()]

        if shared_mask is not None:
            shared_mask = shared_mask.astype(bool)
        if unique_mask is not None:
            unique_mask = unique_mask.astype(bool)
        if other_mask is not None:
            other_mask = other_mask.astype(bool)
        if other_mask is not None: # other 暂时合并到 shared
            if shared_mask is None:
                shared_mask = other_mask
            else:
                shared_mask = shared_mask | other_mask

        if self.class_names_override:
            name_to_index = {name: idx for idx, name in enumerate(stored_names)}
            reorder = []
            for name in self.class_names_override:
                if name not in name_to_index:
                    raise ValueError(
                        f"Class '{name}' missing from super-attribute cache ({stored_names})."
                    )
                reorder.append(name_to_index[name])
            reorder_t = torch.as_tensor(reorder, dtype=torch.long)
            cluster_category = cluster_category[:, reorder_t]
            if shared_mask is not None:
                shared_mask = shared_mask[:, reorder]
            if unique_mask is not None:
                unique_mask = unique_mask[:, reorder]
            self.class_names = list(self.class_names_override)
        else:
            self.class_names = stored_names
        self.cluster_category = cluster_category
        if shared_mask is not None:
            self.cluster_shared_mask = torch.from_numpy(
                shared_mask.astype(np.float32)
            )
        if unique_mask is not None:
            self.cluster_unique_mask = torch.from_numpy(
                unique_mask.astype(np.float32)
            )
            
        self._loaded = True

    def get_cluster_state(self, device: torch.device):
        """
        返回聚类后的簇嵌入、相似度与簇-类别 incidence 三元组。
        """
        self._load()
        assert self.cluster_embeddings is not None
        assert self.cluster_similarity is not None
        assert self.cluster_category is not None
        state = {
            "embeddings": self.cluster_embeddings.to(device),
            "similarity": self.cluster_similarity.to(device),
            "incidence": self.cluster_category.to(device),
        }
        if self.cluster_shared_mask is not None:
            state["shared_mask"] = self.cluster_shared_mask.to(device)
        if self.cluster_unique_mask is not None:
            state["unique_mask"] = self.cluster_unique_mask.to(device)
        return state

    def build_class_prototypes(
        self,
        refined_clusters: torch.Tensor,
        device: torch.device,
        incidence_override: Optional[torch.Tensor] = None,
    ):
        """
        根据 incidence 重新生成类别原型，用于属性对齐 / 推理融合。
        """
        self._load()
        incidence = (
            incidence_override.to(device)
            if incidence_override is not None
            else self.cluster_category.to(device)
        )
        counts = incidence.sum(dim=0, keepdim=True).clamp(min=1.0)
        prototypes = torch.matmul(incidence.t(), refined_clusters) / counts.t()
        return F.normalize(prototypes, dim=-1)
