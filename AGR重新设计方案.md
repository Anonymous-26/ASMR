# 新增模块：SUBS

## Semantic Uncertain Background Suppression

## 语义不确定背景抑制模块

---

## 1. 模块目标

SUBS 要解决的问题不是“如何挖掘更多 novel 伪样本”，而是：

> 在 FSOD 的 base 预训练和 novel 微调中，部分未标注 novel 目标会被采样为 background ROI。如果这些 ROI 被强制训练为背景，会形成 false-negative background supervision，压制 novel 类学习。

因此，SUBS 的目标是：

> 利用 ASM 属性分支的 top-k novel 语义召回能力，判断一个 background ROI 是否可能不是可靠背景；如果可疑，就降低它作为背景负样本的分类损失权重。

---

# 2. 核心设计原则

## 原则一：不做 hard pseudo label

当前 novel 属性 top1 acc 只有约 0.5–0.6，检测分支 novel top1 也只有约 0.62–0.66。因此无论属性分支还是检测分支，都不适合把 background ROI 直接改成某个 novel 类。

所以 SUBS 明确删除以下逻辑：

```text
background ROI → predicted novel class → hard pseudo label → CE supervision
```

SUBS 只做：

```text
background ROI → semantic novel risk → down-weight background loss
```

---

## 原则二：属性分支只做风险估计，不做类别决策

属性分支 top1 不够可靠，但 top3 接近 0.9，说明它更适合回答：

> 这个 ROI 是否可能属于某个 novel 类？

而不是回答：

> 这个 ROI 具体是哪一个 novel 类？

所以 SUBS 使用属性分支的 top-k novel 响应来估计风险，而不使用属性 top1 生成类别标签。

---

## 原则三：只改 background loss，不改 foreground loss

真实标注的 base / novel foreground ROI 仍然使用原始检测分类损失。

SUBS 只作用于：

```text
gt_classes == num_classes
```

也就是 background ROI。

这样模块不会干扰正常前景监督。

---

## 原则四：连续降权优先于硬忽略

可疑 background 不一定真的是 novel。直接 ignore 可能损失有效背景样本。

因此主方案使用连续权重：

[  
w_i = \max(1-r_i, w_{min})  
]

而不是简单地：

[  
w_i = 0  
]

---

# 3. 确定的最终方案

## 3.1 输入

对于每个 ROI (i)，已有：

- 检测分支 logits：
    

[  
l_i^{det}  
]

- 检测分支概率：
    

[  
p_i^{det}=softmax(l_i^{det})  
]

- 属性分支输出的 novel 类语义概率：
    

[  
p_i^{attr}(c), \quad c \in C_{novel}  
]

如果当前属性分支输出的是相似度，而不是概率，则先对 novel 类相似度做 softmax：

# [  
p_i^{attr}(c)

\frac{\exp(s_{i,c}/\tau)}  
{\sum_{c'\in C_{novel}}\exp(s_{i,c'}/\tau)}  
]

---

## 3.2 语义 novel 风险

对每个 background ROI，取属性分支 novel 类 top-k 响应。

根据你的实验，最终固定：

[  
K=3  
]

[  
T_i = Top3_{c\in C_{novel}}p_i^{attr}(c)  
]

定义 semantic novel risk：

[  
r_i = \sum_{c\in T_i}p_i^{attr}(c)  
]

直观解释：

> 如果一个 background ROI 在属性分支的 novel top3 上有较高总响应，则它可能不是可靠背景。

---

## 3.3 背景损失权重

对 background ROI，定义：

[  
w_i = \max(1-\mathrm{stopgrad}(r_i), w_{min})  
]

推荐默认值：

[  
w_{min}=0.2  
]

其中：

- (r_i) 越高，说明越像 novel，背景损失越小；
    
- (w_{min}) 防止完全忽略背景；
    
- `stopgrad` 防止 detector loss 反向优化属性风险本身，保持属性分支作为稳定辅助信号。
    

---

## 3.4 分类损失

对 foreground ROI：

# [  
\mathcal{L}_{fg}

CE(p_i^{det}, y_i)  
]

对 background ROI：

# [  
\mathcal{L}_{bg}^{SUBS}

w_i \cdot CE(p_i^{det}, bg)  
]

最终分类损失：

# [  
\mathcal{L}_{cls}^{SUBS}

\frac{1}{N}  
\left[  
\sum_{i\in FG} CE(p_i^{det}, y_i)  
+  
\sum_{i\in BG} w_i CE(p_i^{det}, bg)  
\right]  
]

总损失：

# [  
\mathcal{L}

\mathcal{L}_{cls}^{SUBS}  
+  
\mathcal{L}_{box}  
+  
\lambda_{attr}\mathcal{L}_{attr}  
]

主方案不加入伪标签损失，也不加入复杂对比学习损失。

---

# 4. 推荐配置

建议新增配置：

```python
_C.MODEL.ATTRIBUTE.SUBS = CN()
_C.MODEL.ATTRIBUTE.SUBS.ENABLED = False
_C.MODEL.ATTRIBUTE.SUBS.TOPK = 3
_C.MODEL.ATTRIBUTE.SUBS.MIN_BG_WEIGHT = 0.2
_C.MODEL.ATTRIBUTE.SUBS.DETACH_RISK = True
_C.MODEL.ATTRIBUTE.SUBS.APPLY_STAGE = "all"  # "base", "novel", "all"
```

最小必要配置只有三个：

```yaml
MODEL:
  ATTRIBUTE:
    SUBS:
      ENABLED: True
      TOPK: 3
      MIN_BG_WEIGHT: 0.2
```

其中：

- `TOPK=3`：由实验结论固定；
    
- `MIN_BG_WEIGHT=0.2`：唯一建议做轻量消融的参数；
    
- 不需要 `PSEUDO_THRESHOLD`、`MARGIN_THRE`、`VISUAL_QUALITY` 等复杂规则。
    

---

# 5. 模块放置位置

SUBS 应该放在 **ROI Head 的训练分类损失计算阶段**。

当前结构中，`CommonalityROIHeads` 会先获得 ROI features，然后通过 `box_predictor` 输出分类和回归预测；ASM / AMBR 属性分支在训练阶段也会参与 attribute forward。之前 AGR 已经在 `AMBR` 中维护过 detector target / weight 之类的逻辑。因此最自然的实现方式有两种。

---

## 实现方式

## 实现方式：更少侵入

### 在 AMBR 中返回 detector classification weights

AMBR 只负责计算：

```python
subs_bg_weights
```

ROIHeads 拿到这个权重后替换分类 loss。

建议让 AMBR 返回一个 dict：

```python
attr_outputs = self.attribute_branch.attribute_forward(...)
subs_bg_weights = attr_outputs.get("subs_bg_weights", None)
```

然后 ROIHeads 使用该权重计算 detector CE。

这个实现保持 AMBR 仍然是属性相关逻辑中心，更符合当前代码结构。

---

# 6. 具体实现步骤

## Step 1：删除或关闭 AGR hard pseudo-label 逻辑

建议先不物理删除，保留配置但默认关闭：

```yaml
MODEL:
  ATTRIBUTE:
    AGR:
      DETECTOR_LOSS_ENABLED: False
      REPLACE_DETECTOR_LOSS: False
```

并确保不再执行：

```python
gt_classes[pseudo_mask] = pseudo_targets
detector_targets[pseudo_mask] = pseudo_targets
```

SUBS 不改 label。

---

## Step 2：新增 SUBS 配置

在 `defrcn/config/defaults.py` 中加入：

```python
_C.MODEL.ATTRIBUTE.SUBS = CN()
_C.MODEL.ATTRIBUTE.SUBS.ENABLED = False
_C.MODEL.ATTRIBUTE.SUBS.TOPK = 3
_C.MODEL.ATTRIBUTE.SUBS.MIN_BG_WEIGHT = 0.2
_C.MODEL.ATTRIBUTE.SUBS.DETACH_RISK = True
_C.MODEL.ATTRIBUTE.SUBS.APPLY_STAGE = "all"
```

---

## Step 3：在 AMBR 初始化中读取配置

在 `attr_ambr.py` 的 `AMBR.__init__` 中加入：

```python
subs_cfg = attr_cfg.SUBS
self.subs_enabled = subs_cfg.ENABLED
self.subs_topk = subs_cfg.TOPK
self.subs_min_bg_weight = subs_cfg.MIN_BG_WEIGHT
self.subs_detach_risk = subs_cfg.DETACH_RISK
self.subs_apply_stage = subs_cfg.APPLY_STAGE
```

---

## Step 4：计算属性 novel 概率

在 AMBR 中新增函数：

```python
def compute_subs_bg_weights(
    self,
    attr_class_logits_or_probs,
    gt_classes,
):
    """
    Args:
        attr_class_logits_or_probs: Tensor [N, num_classes]
        gt_classes: Tensor [N]
    Returns:
        bg_weights: Tensor [N], only background entries are changed
        risk: Tensor [N]
    """
```

如果输入是 logits：

```python
attr_probs = F.softmax(attr_class_logits_or_probs, dim=1)
```

取 novel 类概率：

```python
novel_probs = attr_probs[:, self.novel_index]
```

取 top3：

```python
topk_probs, _ = novel_probs.topk(
    k=min(self.subs_topk, novel_probs.size(1)),
    dim=1
)
risk = topk_probs.sum(dim=1)
```

注意：如果 novel_probs 是完整 softmax 的一部分，top3 sum 通常不会过高；如果只对 novel 类重新 softmax，top3 sum 可能总是接近 1。  
因此这里有一个关键实现选择。

---

# 7. 关键细节：risk 应该怎么归一化？

这是唯一需要谨慎的地方。

## 推荐主实现：使用全类别 softmax 后的 novel top3 sum

即：

[  
p_i^{attr}=softmax(l_i^{attr})  
]

然后：

[  
r_i=\sum_{c\in Top3(C_{novel})}p_i^{attr}(c)  
]

不要只在 novel 类内部 softmax。

原因：

如果只在 novel 类中 softmax，top3 概率和容易始终偏高，无法判断“像不像 novel”。

全类别 softmax 可以表达：

> 这个 ROI 的属性响应到底集中在 novel 类，还是 base / background 类。

如果属性分支没有 background 类，则可在 base+novel 类上 softmax，用 novel mass 作为风险。

最终推荐：

```python
attr_probs = F.softmax(attr_logits[:, :self.num_classes], dim=1)
novel_probs = attr_probs[:, self.novel_index]
risk = novel_probs.topk(k=3, dim=1).values.sum(dim=1)
```

---

# 8. 背景权重计算

```python
bg_mask = gt_classes == self.num_classes

weights = torch.ones_like(gt_classes, dtype=torch.float32)

if self.subs_detach_risk:
    risk = risk.detach()

bg_weight = 1.0 - risk
bg_weight = torch.clamp(bg_weight, min=self.subs_min_bg_weight, max=1.0)

weights[bg_mask] = bg_weight[bg_mask]
```

返回：

```python
return weights, risk
```

注意：

- foreground 权重始终为 1；
    
- 只有 background ROI 被降权；
    
- 不改变 `gt_classes`；
    
- 不产生 pseudo target。
    

---

# 9. 修改 detector classification loss

当前普通 CE 大概率类似：

```python
loss_cls = F.cross_entropy(pred_class_logits, gt_classes, reduction="mean")
```

改成：

```python
loss_cls_raw = F.cross_entropy(
    pred_class_logits,
    gt_classes,
    reduction="none"
)

if subs_weights is not None:
    loss_cls = (loss_cls_raw * subs_weights).sum() / subs_weights.sum().clamp(min=1.0)
else:
    loss_cls = loss_cls_raw.mean()
```

这里建议用：

[  
\frac{\sum w_i L_i}{\sum w_i}  
]

而不是简单 mean，避免 batch 中可疑背景多时整体 loss scale 下降。

---

# 10. 训练阶段策略

## Base pre-training

建议开启 SUBS。

原因：

base 预训练中可能出现未标注 novel 目标。如果它们被当成 background，会提前压制 novel 相关特征。

但 base 阶段分类器可能没有 novel 类输出。此时有两种选择：

### 情况：属性分支仍有全 20 类 / 80 类原型

可以开启 SUBS，因为属性分支知道 novel 类语义原型。

---

## Novel fine-tuning

必须开启 SUBS。

1-shot 微调中，不完整标注更严重；同一图像中可能存在多个 novel 目标但只有一个被标注，基类数据中也可能混有未标注 novel。因此 SUBS 对 novel fine-tuning 更重要。

---

# 11. 是否需要结合检测分支？

主方案不需要。

原因：

检测分支虽然 top1 更高，但它已经是主分类器，正常参与 CE 训练。SUBS 的作用不是再次用检测分支判断类别，而是引入一个额外的语义风险信号，防止“视觉当前认为背景”的 ROI 被过强压制。

如果使用检测分支进行筛选，可能会出现：

```text
detector 当前认为它是 background → 不降权 → 继续被训练成 background
```

这会削弱 SUBS 的意义。

因此主方案只用属性分支计算 risk。

---

# 12. 可选增强：轻量视觉门控

作为变体，可以加入一个非常简单的视觉门控：

# [  
r_i^{final}

r_i^{attr} \cdot (1 - p_i^{det}(bg))^\gamma  
]

但不推荐主方案使用。

原因：

- 会引入 (\gamma)；
    
- 早期 detector 对 novel 不可靠；
    
- 容易漏掉真正需要抑制的 false-negative background。
    

因此建议只作为 ablation：

```yaml
SUBS.VISUAL_GATE: False
```

---


# 13. 监控指标

为了验证 SUBS 是否按预期工作，建议记录以下指标：

```python
subs/risk_bg_mean
subs/risk_bg_top10_mean
subs/bg_weight_mean
subs/bg_weight_min
subs/bg_weight_p10
subs/num_suppressed_bg
subs/suppressed_bg_ratio
```

其中：

```python
suppressed_bg = bg_weight < 0.99
```

还可以记录：

```python
subs/risk_fg_base_mean
subs/risk_fg_novel_mean
```

理想现象：

- novel foreground 的 `risk` 应该高；
    
- clean background 的 `risk` 应该低；
    
- 一部分 background ROI 被降权；
    
- `suppressed_bg_ratio` 不应过高，建议大致在 5%–30% 之间。
    

如果超过 50%，说明 risk 太泛化，可能需要提高 `MIN_BG_WEIGHT` 或降低属性风险。

---

# 14. 最终设计一句话总结

> **SUBS 不利用属性分支决定 novel 类别，而是利用属性分支的 top-k novel 召回能力估计 background ROI 的语义不确定性，并据此降低其背景分类损失，从而缓解未标注 novel 目标被错误监督为背景的问题。**

这就是当前条件下最必要、最有效、最简洁的最终模块设计。