# R-SUBS：Relative Semantic Uncertain Background Suppression

## 相对语义不确定背景抑制

---

## 1. 最终设计目标

R-SUBS 只解决一个问题：

> FSOD 中部分未标注 novel 目标会被采样成 background ROI，导致检测器被错误监督为“这些区域是背景”。R-SUBS 利用 ASM 属性分支估计 background ROI 的 novel 语义风险，并对高风险 background ROI 的背景分类损失进行降权。

它不做：

```text
hard pseudo label
partial-label loss
det-attr top-k intersection
visual quality filtering
contrastive correction
```

最终模块只做：

```text
background ROI → ASM novel semantic risk → relative calibration → weighted background CE
```

---

# 2. 为什么必须使用 Relative-SUBS

当前观察到：

```text
risk_bg_mean ≈ 0.16
risk_bg_max  ≈ 0.188
bg_weight_min ≈ 0.81
bg_weight_mean ≈ 0.84
```

这说明 raw risk 本身太小，导致抑制力度不足。直接用：

[  
w_i = 1 - r_i  
]

最多只能把背景 loss 降到 0.81，实际作用太弱。

但这不代表 SUBS 错了，而是说明：

> 全类别 softmax 下 novel top3 概率绝对值偏低，应该使用 background ROI 内部的相对风险，而不是 raw risk 的绝对值。

所以最终方案使用 **relative risk calibration**。

---

# 3. 最终公式

## 3.1 Raw semantic novel risk

对每个 ROI (i)，属性分支输出类别 logits：

[  
l_i^{attr}  
]

先在全部 foreground 类上做 softmax：

[  
p_i^{attr} = softmax(l_i^{attr})  
]

取 novel 类概率：

[  
p_i^{novel}(c), \quad c \in C_{novel}  
]

取 novel top3：

[  
T_i = Top3_{c\in C_{novel}}p_i^{novel}(c)  
]

定义 raw risk：

[  
r_i^{raw} = \sum_{c\in T_i} p_i^{novel}(c)  
]

这里固定：

[  
K=3  
]

原因是实验显示 novel top3 接近 0.9，而 top1 不够可靠。

---

## 3.2 Background-relative risk calibration

只在 background ROI 内计算统计量。

设：

[  
B = {i \mid y_i = bg}  
]

计算：

[  
\mu_B = mean(r_i^{raw}), \quad i\in B  
]

[  
m_B = max(r_i^{raw}), \quad i\in B  
]

相对风险：

# [  
r_i^{rel}

clip  
\left(  
\frac{  
r_i^{raw} - \mu_B  
}{  
m_B - \mu_B + \epsilon  
},  
0,  
1  
\right)  
]

解释：

- raw risk 低于平均 background 风险：认为是普通背景；
    
- raw risk 高于平均值越多：越像潜在未标注 novel；
    
- batch 中最可疑 background 的 relative risk 接近 1。
    

---

## 3.3 Background loss weight

背景权重定义为：

[  
w_i =  
1 - \lambda r_i^{rel}  
]

并裁剪：

[  
w_i = clamp(w_i, w_{min}, 1)  
]

最终固定推荐：

```yaml
SUPPRESS_STRENGTH: 0.7
MIN_BG_WEIGHT: 0.3
```

也就是：

[  
w_i = clamp(1 - 0.7r_i^{rel}, 0.3, 1)  
]

这样：

- 普通背景：(w_i \approx 1)
    
- 中等可疑背景：(w_i \approx 0.6\sim0.8)
    
- 最可疑背景：(w_i \approx 0.3)
    

这比当前所有 background 都约 0.84 更合理。

---

## 3.4 Final detector classification loss

Foreground ROI 不变：

[  
\mathcal{L}_{fg}=CE(p_i^{det},y_i)  
]

Background ROI 使用加权 CE：

# [  
\mathcal{L}_{bg}^{R-SUBS}

w_i CE(p_i^{det}, bg)  
]

最终：

# [  
\mathcal{L}_{cls}^{R-SUBS}

\frac{  
\sum_i w_i \cdot CE(p_i^{det}, y_i)  
}{  
\sum_i w_i  
}  
]

其中 foreground ROI 的：

[  
w_i = 1  
]

---

# 4. 最终配置

建议新增或修改配置为：

```yaml
MODEL:
  ATTRIBUTE:
    SUBS:
      ENABLED: True
      TOPK: 3
      RISK_NORM: "bg_relative"
      SUPPRESS_STRENGTH: 0.7
      MIN_BG_WEIGHT: 0.3
      DETACH_RISK: True
      APPLY_STAGE: "novel"
```

如果 base 预训练阶段属性分支能够访问全部 base+novel 原型，则可以改成：

```yaml
APPLY_STAGE: "all"
```

但我建议主实验先用：

```yaml
APPLY_STAGE: "novel"
```

原因是 novel fine-tuning 阶段的不完整标注问题最严重，且分类器已经包含 novel 类，效果更容易稳定体现。

---

# 5. 关键实现选择

## 一：risk 使用全类别 softmax，不使用 novel-only softmax

使用：

```python
attr_probs = F.softmax(attr_logits[:, :num_classes], dim=1)
novel_probs = attr_probs[:, novel_indices]
risk_raw = novel_probs.topk(k=3, dim=1).values.sum(dim=1)
```

不要使用：

```python
novel_probs = F.softmax(attr_logits[:, novel_indices], dim=1)
```

因为 novel-only softmax 会让 top3 sum 天然偏高，无法判断该 ROI 是否真的更像 novel。

---

## 二：只对 background ROI 降权

```python
bg_mask = gt_classes == num_classes
weights = torch.ones_like(gt_classes, dtype=torch.float32)
weights[bg_mask] = bg_weight[bg_mask]
```

foreground 权重必须保持 1。

---

## 三：risk 必须 detach

```python
risk = risk.detach()
```

原因：

> R-SUBS 只把属性分支作为语义风险估计器，不希望 detector classification loss 反向驱动属性分支人为改变 risk。

---

## 四：归一化 classification loss

使用：

```python
loss_cls = (loss_raw * weights).sum() / weights.sum().clamp(min=1.0)
```

不要直接：

```python
(loss_raw * weights).mean()
```

否则当大量 background 被降权时，整体 loss scale 会下降，影响训练稳定性。

# 6. 监控指标

最终必须记录：

```text
subs/raw_risk_bg_mean
subs/raw_risk_bg_max
subs/rel_risk_bg_mean
subs/rel_risk_bg_max
subs/bg_weight_mean
subs/bg_weight_min
subs/bg_weight_p10
subs/suppressed_bg_ratio
```

其中：

```python
suppressed_bg_ratio = ((weights[bg_mask] < 0.99).float().mean())
```

# 最终一句话方案

最终确定方案是：

> **保留 ASM 作为属性语义辅助分支；删除 AGR 的 hard pseudo-label 逻辑；对 background ROI 计算 novel top3 raw semantic risk；在 batch 内 background ROI 上做 relative risk calibration；根据 calibrated risk 对 background classification loss 软降权；不引入伪标签、partial label 或对比学习。**

最终模块名称：

# R-SUBS

**Relative Semantic Uncertain Background Suppression**
