# Adaptive AGR v2

Adaptive AGR v2 将旧版 AGR 的多阈值规则改成连续质量估计。模块仍位于 `defrcn/modeling/roi_heads/attr_ambr.py`，配置入口是 `MODEL.ATTRIBUTE.AGR`。

## 核心变化

旧版 AGR 依赖：

- `BG_THRESHOLD`
- `PSEUDO_THRESHOLD`
- `MARGIN_THRE`
- `MAX_PSEUDO_PER_BATCH`
- 多个视觉概率阈值

v2 不再使用这些人工规则来决定伪标签质量。旧配置键保留只是为了兼容历史脚本，避免 cfg merge 失败。

v2 只使用一个质量分数：

```text
q = semantic_score * margin_score * visual_score
```

然后根据 `q` 决定 RoI 的训练方式：

```text
q < SOFT_Q      : 保持背景
SOFT_Q <= q    : soft regularization
HARD_Q <= q    : hard pseudo label
```

## 质量分数

### semantic_score

对背景 RoI 的属性 embedding 和 novel 类原型计算 cosine similarity，取 top-1：

```python
max_sim = top1(cos(attr_embedding, novel_prototypes))
```

然后在当前 batch 的背景 RoI 内做标准化：

```python
semantic_score = sigmoid((max_sim - mean(max_sim)) / std(max_sim))
```

### margin_score

使用 top-1 和 top-2 novel 相似度差：

```python
margin = top1_sim - top2_sim
margin_score = sigmoid((margin - mean(margin)) / std(margin))
```

它表示语义预测是否有区分度。

### visual_score

如果检测器 logits 可用，则引入视觉分支置信度：

```python
visual_score = sqrt((1 - visual_bg_prob) * visual_semantic_target_prob)
```

如果视觉 logits 不可用，则 `visual_score = 1`。

## 训练行为

### soft regularization

当 `q >= SOFT_Q` 时，RoI 不直接硬改标签，而是加入 soft regularization：

```python
loss = 1 - cosine(attr_embedding, target_novel_prototype)
     + max(0, margin + hard_negative_sim - positive_sim)
```

每个 RoI 的 loss 按 `q` 加权：

```python
loss = q * loss
```

这样高质量候选贡献更大，低质量候选不会被强行伪标。

### hard pseudo label

当 `q >= HARD_Q` 时，RoI 才会被赋予 hard pseudo label：

```python
gt_classes[roi] = target_novel_class
```

如果开启 detector AGR loss，则 hard pseudo label 也会进入 detector classification loss。

### detector AGR loss

由以下配置控制：

```yaml
MODEL.ATTRIBUTE.AGR.DETECTOR_LOSS_ENABLED
MODEL.ATTRIBUTE.AGR.REPLACE_DETECTOR_LOSS
MODEL.ATTRIBUTE.AGR.IGNORE_IN_DETECTOR
MODEL.ATTRIBUTE.AGR.PSEUDO_LOSS_WEIGHT
MODEL.ATTRIBUTE.AGR.NORMALIZE_BY_ALL_ROIS
```

默认不启用 detector AGR loss。启用后，hard pseudo label 的 detector loss 权重为：

```python
q * PSEUDO_LOSS_WEIGHT
```

## 当前有效配置

| 配置 | 默认值 | 作用 |
|---|---:|---|
| `MODEL.ATTRIBUTE.AGR.ENABLED` | `False` | 显式开启 Adaptive AGR v2 |
| `MODEL.ATTRIBUTE.AGR.SOFT_Q` | `0.20` | 进入 soft regularization 的质量门槛 |
| `MODEL.ATTRIBUTE.AGR.HARD_Q` | `0.55` | 进入 hard pseudo label 的质量门槛 |
| `MODEL.ATTRIBUTE.AGR.SOFT_REG_WEIGHT` | `0.05` | soft regularization 权重 |
| `MODEL.ATTRIBUTE.AGR.SOFT_REG_MARGIN` | `0.05` | hard negative margin |
| `MODEL.ATTRIBUTE.AGR.DETECTOR_LOSS_ENABLED` | `False` | 是否接入 detector AGR loss |
| `MODEL.ATTRIBUTE.AGR.REPLACE_DETECTOR_LOSS` | `False` | 是否替换原始 `loss_cls` |
| `MODEL.ATTRIBUTE.AGR.IGNORE_IN_DETECTOR` | `False` | 是否在 detector loss 中忽略 soft-only RoI |
| `MODEL.ATTRIBUTE.AGR.PSEUDO_LOSS_WEIGHT` | `0.1` | hard pseudo detector loss 权重上限 |
| `MODEL.ATTRIBUTE.AGR.NORMALIZE_BY_ALL_ROIS` | `True` | detector AGR loss 是否按全部 RoI 归一化 |

## 兼容说明

以下旧配置仍保留，但 Adaptive AGR v2 不再使用其规则逻辑：

- `MODEL.ATTRIBUTE.AGR.MAX_PSEUDO_PER_BATCH`
- `MODEL.ATTRIBUTE.AGR.VISUAL_QUALITY_ENABLED`
- `MODEL.ATTRIBUTE.AGR.MAX_VISUAL_BG_PROB`
- `MODEL.ATTRIBUTE.AGR.MIN_VISUAL_NOVEL_PROB`
- `MODEL.ATTRIBUTE.AGR.MIN_VISUAL_SEMANTIC_TARGET_PROB`
- `MODEL.ATTRIBUTE.AGR.REQUIRE_VISUAL_SEMANTIC_AGREEMENT`
- `MODEL.ATTRIBUTE.AGR.PSEUDO_WEIGHT_BY_CONFIDENCE`
- `MODEL.ATTRIBUTE.AGR.SOFT_REG_ENABLED`
- `MODEL.ATTRIBUTE.AGR.SOFT_REG_WEIGHT_BY_CONFIDENCE`
- `MODEL.ATTRIBUTE.AGR.SOFT_REG_SUPPRESS_HARD_LABELS`

`BG_THRESHOLD > 0`、`PSEUDO_THRESHOLD > 0`、旧 `SOFT_REG_ENABLED=True` 或 `DETECTOR_LOSS_ENABLED=True` 都会兼容性地激活 AGR v2，方便旧脚本继续运行；但这些旧阈值不再作为筛选规则参与质量判断。

## 日志指标

新增 v2 指标：

- `agr_v2/quality_mean`
- `agr_v2/quality_max`
- `agr_v2/semantic_score_mean`
- `agr_v2/margin_score_mean`
- `agr_v2/visual_score_mean`
- `agr_v2/num_soft`
- `agr_v2/num_hard`

保留兼容指标：

- `agr_quality/num_semantic_pseudo_raw`
- `agr_quality/num_pseudo_final`
- `agr_quality/visual_bg_prob_mean`
- `agr_quality/visual_novel_prob_mean`
- `agr_quality/visual_semantic_target_prob_mean`
- `agr_quality/visual_semantic_agreement_ratio`
- `agr_soft/loss`
- `agr_soft/num_rois`
- `agr_soft/quality_mean`
- `agr_detector/loss_cls`
- `agr_detector/num_ignored`
- `agr_detector/num_pseudo`
- `agr_detector/pseudo_weight_mean`

## 推荐使用

建议从保守设置开始：

```bash
MODEL.ATTRIBUTE.AGR.ENABLED True
MODEL.ATTRIBUTE.AGR.SOFT_Q 0.20
MODEL.ATTRIBUTE.AGR.HARD_Q 0.55
MODEL.ATTRIBUTE.AGR.SOFT_REG_WEIGHT 0.03
MODEL.ATTRIBUTE.AGR.DETECTOR_LOSS_ENABLED False
MODEL.ATTRIBUTE.BG_THRESHOLD 0.0
MODEL.ATTRIBUTE.PSEUDO_THRESHOLD 0.0
```

如果 `agr_v2/num_hard` 长期为 0，但 `num_soft` 合理，可以先只用 soft regularization。不要急着降低 `HARD_Q`，否则容易重新引入噪声 hard pseudo label。
