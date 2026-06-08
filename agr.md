# AGR 模块整理

AGR 在当前代码中属于属性分支的一部分，主要实现位于 `defrcn/modeling/roi_heads/attr_ambr.py`，配置入口位于 `defrcn/config/defaults.py` 的 `MODEL.ATTRIBUTE.AGR`。它的目标是利用属性语义分支发现被检测器采样为背景的 RoI 中可能存在的新类目标，从而对这些 RoI 做忽略、伪标签或软约束处理。

## 1. AGR 包含的内容

当前 AGR 由四部分组成。

### 1.1 基于属性相似度的背景 RoI 筛选

入口函数是 `_apply_background_filtering`。

流程：

1. 取当前 batch 中 `gt_classes == num_classes` 的背景 RoI。
2. 用属性 embedding 与 novel 类原型计算相似度：
   ```python
   similarities = bg_embeddings @ novel_prototypes.T
   ```
3. 取 top-1 novel 相似度 `max_sim` 和 top-1 / top-2 margin。
4. 当满足：
   ```python
   max_sim > MODEL.ATTRIBUTE.BG_THRESHOLD
   margin > MODEL.ATTRIBUTE.MARGIN_THRE
   ```
   时，将该背景 RoI 视为语义可疑背景。
5. 默认处理方式是将这些 RoI 的属性分支 target 设为 `-1`，即忽略其属性损失。

相关全局属性配置：

- `MODEL.ATTRIBUTE.BG_THRESHOLD`
- `MODEL.ATTRIBUTE.PSEUDO_THRESHOLD`
- `MODEL.ATTRIBUTE.BG_SUPPRESSION_WEIGHT`
- `MODEL.ATTRIBUTE.MARGIN_THRE`

注意：如果 `BG_THRESHOLD <= 0`，整个背景过滤逻辑直接关闭。

### 1.2 伪标签生成

当 `PSEUDO_THRESHOLD > 0` 时，AGR 会进一步从语义可疑背景中筛选伪标签：

```python
semantic_pseudo_mask = (max_sim > PSEUDO_THRESHOLD) & is_discriminative
```

伪标签类别来自属性相似度最高的新类：

```python
pseudo_targets = novel_indices[max_idx[pseudo_mask]]
```

如果不开启 soft regularization，则伪标签会直接写入属性分支 target：

```python
gt_classes[pseudo] = pseudo_targets
```

如果同时开启检测器 AGR loss，还会写入 detector target：

```python
detector_targets[pseudo] = pseudo_targets
```

### 1.3 视觉质量过滤

AGR 可以用检测器当前分类 logits 对属性语义伪标签做二次筛选。

配置开关：

```yaml
MODEL.ATTRIBUTE.AGR.VISUAL_QUALITY_ENABLED
```

启用后，候选伪标签需要经过以下可选条件：

- 背景概率不能太高：
  ```python
  visual_bg_prob < MAX_VISUAL_BG_PROB
  ```
- novel 最大概率需要足够高：
  ```python
  visual_novel_prob > MIN_VISUAL_NOVEL_PROB
  ```
- 语义指定类别的视觉概率需要足够高：
  ```python
  visual_semantic_target_prob > MIN_VISUAL_SEMANTIC_TARGET_PROB
  ```
- 可选要求视觉 top novel 类和语义 pseudo 类一致：
  ```python
  REQUIRE_VISUAL_SEMANTIC_AGREEMENT
  ```

这部分的作用是减少纯属性相似度带来的错误伪标签。

### 1.4 检测器分类 loss 接入

入口函数是 `_compute_detector_agr_cls_loss`，在 `roi_heads.py` 中调用。

如果：

```yaml
MODEL.ATTRIBUTE.AGR.DETECTOR_LOSS_ENABLED: True
```

则 AGR 会基于 `detector_targets` 和 `detector_weights` 额外计算检测器分类交叉熵。

两种接入方式：

1. 替换原始 `loss_cls`
   ```yaml
   MODEL.ATTRIBUTE.AGR.REPLACE_DETECTOR_LOSS: True
   ```
2. 作为额外 loss 加入：
   ```python
   losses["loss_cls_agr"] = agr_cls_loss
   ```

如果 `IGNORE_IN_DETECTOR=True`，被 AGR 判断为可疑但未成为最终伪标签的背景 RoI 会在检测器分类 loss 中被忽略。

### 1.5 Soft Regularization

配置开关：

```yaml
MODEL.ATTRIBUTE.AGR.SOFT_REG_ENABLED
```

当开启后，AGR 不一定把候选 RoI 直接硬改成新类标签，而是加入一个属性 embedding 到目标 novel prototype 的软约束。

损失由两部分组成：

1. 正类原型对齐：
   ```python
   1 - cosine(pred_embedding, target_prototype)
   ```
2. hard negative margin：
   ```python
   max(0, SOFT_REG_MARGIN + hard_neg_sim - pos_sim)
   ```

最终乘以：

```yaml
MODEL.ATTRIBUTE.AGR.SOFT_REG_WEIGHT
```

如果：

```yaml
MODEL.ATTRIBUTE.AGR.SOFT_REG_SUPPRESS_HARD_LABELS: True
```

则 soft regularization 会抑制硬伪标签写入，将对应 RoI 的属性 target 设为 `-1`，只保留软约束。

## 2. 可选配置

### 2.1 检测器 loss 相关

| 配置 | 默认值 | 作用 |
|---|---:|---|
| `DETECTOR_LOSS_ENABLED` | `False` | 是否计算 AGR detector 分类 loss |
| `REPLACE_DETECTOR_LOSS` | `False` | 是否用 AGR loss 替换原始 `loss_cls` |
| `IGNORE_IN_DETECTOR` | `False` | 是否在检测器 loss 中忽略被 AGR 抑制的背景 RoI |
| `PSEUDO_LOSS_WEIGHT` | `0.1` | detector 伪标签 loss 权重 |
| `PSEUDO_WEIGHT_BY_CONFIDENCE` | `True` | 是否按属性相似度置信度缩放伪标签权重 |
| `NORMALIZE_BY_ALL_ROIS` | `True` | detector AGR loss 是否按全部 RoI 数归一化 |

### 2.2 伪标签数量与质量控制

| 配置 | 默认值 | 作用 |
|---|---:|---|
| `MAX_PSEUDO_PER_BATCH` | `0` | 每个 batch 最多保留多少伪标签；`0` 表示不限制 |
| `VISUAL_QUALITY_ENABLED` | `True` | 是否使用检测器视觉概率过滤语义伪标签 |
| `MAX_VISUAL_BG_PROB` | `0.95` | 候选 RoI 的背景概率上限 |
| `MIN_VISUAL_NOVEL_PROB` | `0.01` | 候选 RoI 的最大 novel 概率下限 |
| `MIN_VISUAL_SEMANTIC_TARGET_PROB` | `0.0` | 语义目标类别的视觉概率下限 |
| `REQUIRE_VISUAL_SEMANTIC_AGREEMENT` | `False` | 是否要求视觉 top novel 类与属性语义伪标签一致 |

### 2.3 Soft Regularization 相关

| 配置 | 默认值 | 作用 |
|---|---:|---|
| `SOFT_REG_ENABLED` | `False` | 是否启用软正则替代/辅助硬伪标签 |
| `SOFT_REG_WEIGHT` | `0.05` | soft regularization loss 权重 |
| `SOFT_REG_MARGIN` | `0.05` | hard negative margin |
| `SOFT_REG_WEIGHT_BY_CONFIDENCE` | `True` | 是否按属性相似度置信度缩放 soft loss |
| `SOFT_REG_SUPPRESS_HARD_LABELS` | `True` | 启用 soft loss 时是否抑制硬伪标签 |

### 2.4 依赖的非 AGR 配置

| 配置 | 默认值 | 作用 |
|---|---:|---|
| `MODEL.ATTRIBUTE.BG_THRESHOLD` | `0.5` | 背景 RoI 被视为语义可疑的阈值 |
| `MODEL.ATTRIBUTE.PSEUDO_THRESHOLD` | `0.7` | 背景 RoI 被转为伪标签的阈值 |
| `MODEL.ATTRIBUTE.BG_SUPPRESSION_WEIGHT` | `0.0` | 对可疑背景施加额外属性惩罚的权重 |
| `MODEL.ATTRIBUTE.MARGIN_THRE` | `0.1` | top-1 与 top-2 novel 相似度差值阈值 |

## 3. 作用原理

AGR 的核心假设是：小样本检测训练中，一些被采样为背景的 RoI 可能实际上包含未标注的新类目标。如果完全按背景训练，会压制新类特征学习。AGR 使用属性语义空间对背景 RoI 进行再判断：

```text
background RoI visual feature
    -> attribute embedding
    -> similarity to novel semantic prototypes
    -> suspicious background / pseudo novel / ignored background
```

具体作用有三种：

1. **减少错误背景监督**
   当背景 RoI 与某个 novel 类属性原型高度相似时，将其属性 target 设为 `-1`，避免属性分支继续把它当背景学习。

2. **挖掘潜在新类伪标签**
   当相似度超过更高的 `PSEUDO_THRESHOLD`，将其赋值为最相似 novel 类，用于属性分支，必要时也用于检测器分类分支。

3. **用软约束替代硬伪标签**
   当硬伪标签噪声较大时，可以开启 soft regularization，只推动 RoI 属性 embedding 靠近目标 novel prototype，并拉开 hard negative prototype，而不强制改写类别标签。

## 4. 当前实现中的日志指标

AGR 会写入以下主要指标：

### detector AGR

- `agr_detector/loss_cls`
- `agr_detector/num_ignored`
- `agr_detector/num_pseudo`
- `agr_detector/pseudo_weight_mean`

### soft AGR

- `agr_soft/loss`
- `agr_soft/num_rois`

### quality filtering

- `agr_quality/num_semantic_pseudo_raw`
- `agr_quality/num_pseudo_final`
- `agr_quality/num_rejected_by_visual`
- `agr_quality/visual_reject_ratio`
- `agr_quality/visual_bg_prob_mean`
- `agr_quality/visual_novel_prob_mean`
- `agr_quality/visual_semantic_target_prob_mean`
- `agr_quality/visual_semantic_agreement_ratio`

## 5. 使用注意

1. `BG_THRESHOLD <= 0` 会关闭整个 AGR 背景过滤和伪标签流程。
2. `PSEUDO_THRESHOLD <= 0` 时只做背景抑制，不生成伪标签。
3. 如果开启 `DETECTOR_LOSS_ENABLED`，需要明确选择是替换 `loss_cls` 还是作为额外 `loss_cls_agr`。
4. `MAX_PSEUDO_PER_BATCH` 固定上限不一定合理。当前数据观察中 VOC 漏标伪标签较少，强行固定数量容易引入噪声。
5. 当前实验结果显示 AGR/SUBS 类模块收益不稳定，因此更适合作为消融或辅助模块，不建议作为主模型精度提升的核心依赖。
