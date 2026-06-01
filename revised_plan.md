• 结合论文和当前代码，我建议转投版本不要简单堆叠三个模块，而是围绕一个统一叙事重构：

  > 静态语义先验只应在视觉证据不足时提供支持；随着视觉证据增强，模型应主动降低语义约束。语义
  > 先验本身也需要根据视觉观测动态校准。

  这可以形成一个新的核心框架：Adaptive Visual-Semantic Manifold Rectification。

  ## 现有实现的关键问题

  当前属性分支在 defrcn/modeling/roi_heads/attr_ambr.py:231 中计算三类固定权重损失：

  loss_attr_proto
  loss_attr_con
  loss_attr_cluster

  这些损失始终约束 RoI embedding 接近静态文本属性原型。即使 30-shot 下视觉分类器已经可靠，约
  束强度也没有降低。这与 VOC Split 2 和 COCO 30-shot 性能退化直接相关。

  AGR 在 defrcn/modeling/roi_heads/attr_ambr.py:453 中依赖三个固定阈值：

  BG_THRESHOLD
  PSEUDO_THRESHOLD
  MARGIN_THRE

  更重要的是，AGR 修改后的 attr_targets 在 defrcn/modeling/roi_heads/roi_heads.py:1005 中没
  有继续传递给视觉分类损失：

  attr_losses, attr_targets = ...
  losses.update(attr_losses)

  也就是说，当前 AGR 主要修正属性分支监督，没有真正修正主检测分类器的监督。这个实现细节很可
  能解释 AGR 只有约 +0.36 AP 的原因。

  ## 1. 自适应语义门控

  ### 推荐设计

  为每个 RoI 学习一个门控系数，而不是按 shot 数量硬编码：

  [
  \alpha_i = \sigma(g([f_i,; u_i,; m_i]))
  ]

  其中：

  - (f_i)：RoI 视觉特征。
  - (u_i)：视觉分类器的不确定性，例如归一化熵。
  - (m_i)：视觉分类 top-1 与 top-2 概率差。
  - (g)：两层 MLP，输出标量。
  - (\alpha_i \in (0,1))：语义分支可信度。

  建议将门控同时用于训练和推理：

  # [
  \mathcal{L}_{sem}

  \frac{1}{N}\sum_i
  \alpha_i
  \left(
  \lambda_p \mathcal{L}{proto}^{(i)}
  +
  \lambda_c \mathcal{L}{con}^{(i)}
  +
  \lambda_a \mathcal{L}_{attr}^{(i)}
  \right)
  ]

  # [
  p_i^{final}

  (1-\alpha_i)p_i^{vis}
  +
  \alpha_i p_i^{sem}
  ]

  其中 (p_i^{sem}) 可由 RoI 属性 embedding 与类别原型的 cosine logits 得到。

  ### 为什么有效

  - 1-shot 下视觉分类熵高、margin 小，模型倾向提高 (\alpha_i)。
  - 30-shot 下视觉证据充分，模型自动减弱静态语义约束。
  - VOC Split 2 中，如果文本聚类偏差导致语义分支与视觉分支冲突，门控可以降低错误先验的影响。

  ### 注意事项

  不能只输入 shot count。否则门控只是手工 schedule 的另一种形式，审稿人仍会质疑泛化能力。

  需要防止门控坍缩为全 0。训练早期可以加入较弱的正则：

  # [
  \mathcal{L}_{gate}

  \max(0,\alpha_{min}-\bar{\alpha})
  ]

  但不建议强制固定目标值。论文应展示不同 shot、不同类别、正确与错误预测样本的 (\alpha) 分
  布。

  ## 2. AGR 升级：动量自适应伪标签

  ### 不建议直接采用 Batch GMM

  GMM 有数学表现力，但当前 FSOD 每个 batch 中有效背景 RoI 的分布波动较大。尤其 1-shot 微调
  时，单 batch 拟合容易不稳定，也会增加实现与复现实验成本。

  更稳妥的主方案是：EMA 动量分布估计 + 自适应分位数阈值。

  对于背景 RoI，先计算：

  [
  s_i = \max_{c \in C_{novel}}
  \cos(e_i, p_c)
  ]

  [
  m_i = s_i^{top1} - s_i^{top2}
  ]

  维护分数和 margin 的 EMA 统计量或直方图：

  # [
  \mathcal{D}_t

  \beta \mathcal{D}{t-1}
  +
  (1-\beta)\mathcal{D}{batch}
  ]

  再动态获得：

  [
  \delta_{bg}^{(t)} = Q_{q_{bg}}(\mathcal{D}_t)
  ]

  [
  \delta_{ps}^{(t)} = Q_{q_{ps}}(\mathcal{D}t), \quad q{ps} > q_{bg}
  ]

  margin 阈值也使用 EMA 分位数，而不是固定 0.2。

  ### 三段式 AGR

  [
  \hat{y}i =
  \begin{cases}
  c_i^*, & s_i \ge \delta{ps}^{(t)} \land m_i \ge \delta_m^{(t)}\
  -1, & s_i \ge \delta_{bg}^{(t)}\
  background, & \text{otherwise}
  \end{cases}
  ]

  建议再引入软权重：

  [
  w_i =
  \operatorname{clip}
  \left(
  \frac{s_i-\delta_{ps}}{1-\delta_{ps}},
  0,1
  \right)
  ]

  伪标签不应与真实标签等权。

  ### 必须修复的监督路径

  动态 AGR 的输出不仅要监督属性分支，还应以受控方式进入主视觉分类损失：

  # [
  \mathcal{L}_{cls}^{AGR}

  \mathcal{L}{cls}^{GT}
  +
  \lambda{ps}
  \sum_{i \in \mathcal{P}}
  w_i \mathcal{L}_{CE}(p_i^{vis}, \hat y_i)
  ]

  其中：

  - exclude 样本从视觉分类背景损失中剔除。
  - pseudo-label 样本以较小权重加入视觉分类器。
  - bbox regression 不使用伪标签，避免错误框回归。

  这是 AGR 从辅助正则升级为真正背景校正器的关键。

  ## 3. 视觉感知的属性消歧

  你的方向正确，但完整 Cross-Attention 容易被质疑复杂度过高。推荐从轻量级、可解释的版本开
  始。

  ### 推荐设计：RoI 条件化属性残差

  当前类别原型是静态聚合：

  [
  p_c =
  \operatorname{Norm}
  \left(
  \frac{1}{|A_c|}
  \sum_{a \in A_c} z_a
  \right)
  ]

  可以改为对每个 RoI 动态重加权：

  # [
  r_{i,a}

  \operatorname{softmax}
  \left(
  \frac{
  (W_q e_i)^\top(W_k z_a)
  }{\sqrt d}
  \right)
  ]

  # [
  p_{i,c}

  \operatorname{Norm}
  \left(
  \sum_a H_{a,c}, r_{i,a},
  (z_a + \Delta z_{i,a})
  \right)
  ]

  其中：

  - (H_{a,c}) 保留原有超图 incidence 结构。
  - (r_{i,a}) 是视觉条件化注意力。
  - (\Delta z_{i,a}) 是轻量残差，可先省略，作为增强消融实验。

  这样，同样是“翅膀”属性：

  - 鸟类 RoI 会提高羽毛、身体轮廓、自然纹理相关属性权重。
  - 飞机 RoI 会提高金属结构、机翼形状、人工边缘相关属性权重。

  ### 更稳妥的实验顺序

  先实现 Cross-Attention Reweighting，不立刻做真正的动态子簇拆分。若有效，再增加每个共享属性
  的 (M) 个可学习视觉子原型：

  [
  z_a^{(1)}, \ldots, z_a^{(M)}
  ]

  由 RoI 注意力选择子原型。这时才可以在论文中强调 sub-cluster manifold。

  ## 推荐的整体方案

  三个模块可以统一为：

  1. Visual-Conditioned Attribute Disambiguation
     用 RoI 条件化注意力修正静态属性原型。
  2. Adaptive Semantic Gating
     根据视觉不确定性控制语义约束和推理融合强度。
  3. Momentum-Adaptive AGR
     根据运行时分布动态划分可靠背景、忽略区域和伪标签区域，并真正反馈到主分类器。

  数据流建议为：

  RoI feature
    ├── visual classifier ────────────────┐
    ├── uncertainty estimator ── gate α ─┤── adaptive fusion
    └── attribute head                    │
          └── visual-conditioned manifold ┘
                └── momentum AGR
                      ├── exclude uncertain negatives
                      └── weighted pseudo-label supervision

  ## 消融实验优先级

  建议按以下顺序推进，避免一次改动太多无法归因：

  ┌──────┬────────────────────────────┬─────────────────────────────────────────┐
  │ 阶段 │ 改动                       │ 目标                                    │
  ├──────┼────────────────────────────┼─────────────────────────────────────────┤
  │ A    │ AGR 输出接入主视觉分类损失 │ 验证当前 AGR 增益小是否源于监督路径不足 │
  │ B    │ EMA 分位数阈值替代固定阈值 │ 消除人工调参质疑                        │
  │ C    │ 自适应语义门控             │ 修复高 shot 倒退                        │
  │ D    │ RoI 条件化属性注意力       │ 回应视觉-语义不对齐                     │
  │ E    │ 属性视觉子原型             │ 仅在 D 有明确收益后增加                 │
  └──────┴────────────────────────────┴─────────────────────────────────────────┘

  核心表格必须覆盖 VOC 三个 Split 的 1/2/3/5/10-shot，以及 COCO 1/2/3/5/10/30-shot。除了
  AP，还应报告：

  - 各 shot 下平均门控值 (\bar{\alpha})。
  - 动态阈值随训练迭代的变化。
  - AGR 生成的 pseudo / excluded / reliable-bg 比例。
  - 伪标签 precision。
  - VOC Split 2 上门控前后的错误类别对。
  - 鸟与飞机等共享属性类别的注意力可视化。

  最先值得实施的是阶段 A 和 B。它们改动较小，却直接回应 AGR 的致命质疑，也能为后续门控提供稳
  定的训练基础。