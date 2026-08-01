---
type: learning-plan
topic: diffusion
status: active
current_gate: G0
started: 2026-07-25
progress_mode: gate-based
---

# Diffusion 学习计划

## 目标与学习方式

本阶段的目标不是直接调用现成 Diffusion 库，而是通过二维 DDPM 实验建立一条完整、可解释的认知链：

> 目标分布 → 前向加噪 → 噪声预测 → 反向采样 → 分布评测 → Diffusion Policy 映射

学习不限定完成天数，只按进度闸门推进。前一关没有通过，不进入后一关。

协作方式：

- 学习者亲手编写和运行代码。
- Codex 负责讲解原理、物理/几何直觉、公式与代码的对应关系。
- 学习者提交代码、图像、实验结果或报错后，Codex 进行检查和反馈。
- 除非学习者明确要求，否则 Codex 不直接编写或修改实现代码。
- 只手写与 Diffusion 机制直接相关的部分；自动微分、优化器、通用神经网络层直接使用 PyTorch。

本阶段暂不要求 CNN、Transformer、图像生成、Flow Matching、Diffusion Policy 训练或完整 ELBO 证明。

## 最终验收

完成本计划后，应当能够：

1. 从公式、代码和几何直觉三个角度解释 DDPM。
2. 独立实现二维双峰分布的前向扩散、噪声预测训练和反向采样。
3. 解释训练时为何只采一个随机时间步，而生成时为何需要迭代。
4. 使用量化指标而不只是“看图不错”判断生成质量。
5. 将二维 DDPM 的每个变量映射到 Diffusion Policy 的动作生成过程。
6. 说明 DDPM 与普通 MSE 行为克隆在多模态输出问题上的差异。

## G0：建立全局图景

### 阅读

主论文：[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

第一遍只读：

- Abstract
- Section 1 Introduction
- Figure 2

暂时跳过详细推导、实验指标和附录。

### 必须回答

1. 前向过程破坏的是什么？
2. 反向模型学习的是什么？
3. 为什么最终可以从高斯噪声生成复杂分布？
4. Diffusion 为什么属于生成模型，而不是普通回归模型？

### 通过标准

能够不看论文，用自己的语言画出并讲清：

```text
真实数据 x_0 → 逐渐加噪 → x_T ≈ 高斯噪声
高斯噪声 x_T → 模型逐步去噪 → 生成样本 x_0
```

## G1：构造目标数据分布

### 学习内容

- 高斯分布的均值、方差和标准差。
- 混合分布与多模态分布。
- batch 维与数据维的区别。
- 为什么可视化是生成实验的第一个正确性检查。

### 实验

生成 4096 个二维双峰高斯样本：

- 左峰中心：`(-2, 0)`
- 右峰中心：`(2, 0)`
- 两峰权重相同
- 每个方向的标准差：`0.3`
- 数据形状：`[4096, 2]`

分别可视化标准差为 `0.3` 和 `1.0` 时的散点分布。

### 产物

- 数据生成代码。
- 两张散点图。
- 一段不超过 200 字的实验解释。

### 通过标准

- 数据形状正确。
- 左右样本比例接近 `1:1`。
- 样本均值接近 `(0, 0)`。
- 能解释多模态、峰内方差和峰间距离分别代表什么。
- 能说明为什么直接用 MSE 回归多个合理目标可能产生“平均结果”。

## G2：前向扩散

### 阅读

回到 DDPM 论文，阅读：

- Section 2
- Section 3.1
- Equation (2)
- Equation (4)
- Algorithm 1

### 学习内容

定义：

$$
\beta_t \in (0,1),\qquad
\alpha_t=1-\beta_t,\qquad
\bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s
$$

前向扩散的闭式采样：

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I)
$$

需要理解：

- `beta` schedule 如何控制每一步破坏数据的速度。
- `alpha_bar` 为什么表示累计保留的信号。
- 为什么两个系数的平方和为 1。
- 为什么可以不经过全部中间步骤，直接得到任意 `x_t`。

### 实验

- 时间步数量 `T=1000`。
- 线性 `beta` schedule：从 `1e-4` 增加到 `2e-2`。
- 可视化 clean、`t=49`、`199`、`499`、`999` 的分布。
- 打印每个时间点的经验均值和协方差。

### 通过标准

- 所有 `x_t` 的形状保持 `[4096, 2]`。
- `beta_t` 递增，`alpha_bar_t` 单调下降。
- 两个峰随时间逐渐消失，而不是突然跳变。
- `t=999` 时样本接近零均值、单位协方差高斯。
- 能逐项解释公式中每个张量的形状和物理意义。

## G3：时间条件噪声预测

### 阅读

阅读 DDPM：

- Section 3.2
- Equation (11)
- Equation (14)
- Algorithm 1 的训练部分

第一遍不要求独立推导完整变分下界。

### 学习内容

模型学习：

$$
\epsilon_\theta(x_t,t)\approx\epsilon
$$

简化训练目标：

$$
\mathcal{L}
=
\mathbb{E}_{x_0,\epsilon,t}
\left[
\left\|
\epsilon-\epsilon_\theta(x_t,t)
\right\|^2
\right]
$$

需要理解：

- 为什么每个训练样本随机选择一个时间步。
- 为什么模型必须同时接收 `x_t` 和 `t`。
- sinusoidal timestep embedding 的作用。
- 为什么标签是自己采样的噪声，不需要人工标注。

### 实验

- 使用二维 MLP 预测噪声。
- 使用 32 维时间编码。
- 每个 batch 为不同样本独立随机采样时间步。
- 先做单批次过拟合，再进行完整训练。
- 记录训练 loss、验证 loss 和恒零噪声预测器的基线。

### 通过标准

- 单批次 loss 可以稳定下降。
- 所有参数都有有限且非零的梯度。
- 正式训练无 `NaN/Inf`。
- 验证 loss 明显优于未训练模型和恒零预测。
- 能解释“预测噪声”和“预测干净样本”的区别。

## G4：反向采样

### 阅读

阅读 DDPM：

- Algorithm 2
- 与反向均值、方差有关的公式

### 学习内容

- 如何由预测噪声估计干净样本。
- 如何构造 $p_\theta(x_{t-1}\mid x_t)$。
- 反向均值决定向哪里去，反向方差保留多少随机性。
- 为什么最后一步不再加入随机噪声。
- 为什么训练可以并行，而反向生成是串行过程。

### 实验

- 从标准高斯噪声开始。
- 完整执行 `T-1 → 0` 的反向采样。
- 保存 `t=999`、`750`、`500`、`250`、`0` 的中间分布。
- 使用真实噪声作为 oracle，单独检查反推公式。

### 通过标准

- oracle 检查能够近似恢复对应的干净样本。
- 反向过程没有形状变化、`NaN` 或错误广播。
- 最后一步不额外加入随机噪声。
- 生成分布从单峰高斯逐渐形成两个峰。
- 能独立解释一次反向更新的每个组成部分。

## G5：可复现与量化评测

### 实验要求

- 先用 seed 0 调通。
- 正式实验使用 seed 0、1、2。
- 每个 seed 生成 4096 个样本。
- 保存训练曲线、前向扩散图、反向轨迹图和最终分布图。
- 保留失败结果，不只保留最好看的结果。

### 量化指标

- 左峰样本比例位于 `40%–60%`。
- 左右生成峰中心到目标中心的距离均不超过 `0.30`。
- 全体生成样本的 y 均值绝对值不超过 `0.15`。
- 两个峰内的平均标准差位于 `0.15–0.55`。
- checkpoint 在新进程中加载后能够再次采样。

### 消融

只改变一个变量：移除时间编码。保持数据、随机种子、网络规模、优化器和训练步数一致。

对比：

- 训练与验证 loss。
- 最终双峰结构。
- 峰中心误差。
- 失败模式。

### 通过标准

- 三个随机种子都能形成清晰双峰。
- 能解释时间编码消融的变化，而不只汇报数值。
- 能区分训练 loss、样本质量和分布覆盖程度。
- 能写出一次失败定位过程。

## G6：映射到 Diffusion Policy

### 阅读

[Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)

本阶段只读：

- Figure 1
- Figure 3
- Section I
- Section II
- Section IV-A：多模态动作
- Section IV-C：动作序列预测

### 对应关系

| 二维 DDPM | Diffusion Policy |
| --- | --- |
| 二维干净样本 $x_0$ | 干净动作序列 $A_t^0$ |
| 高斯噪声 $x_T$ | 随机初始化的动作序列 |
| 时间步 $t$ | 动作去噪迭代步 |
| 二维噪声预测 MLP | 条件 1D U-Net 或 Transformer |
| 无条件生成 | 以图像和机器人状态为条件 |
| 一个二维点 | 一个 action chunk |
| 反向采样 | 逐步生成可执行动作序列 |

### 必须回答

1. 为什么 Diffusion Policy 生成动作序列而不是单步动作？
2. 视觉和机器人状态作为条件时，哪些变量被加噪，哪些不被加噪？
3. 多模态动作在机器人任务中对应什么实际情况？
4. action horizon 和 execution horizon 为什么不同？
5. 迭代去噪为什么会带来控制延迟？

### 通过标准

不看资料，独立画出：

```text
示范动作
→ 随机时间步加噪
→ 结合观察条件预测噪声
→ epsilon MSE
→ 从随机动作序列迭代去噪
→ action chunk
→ 执行部分动作并重新规划
```

通过 G6 后，才制定下一阶段的 Conditional Diffusion、Push-T 和 LeRobot Diffusion Policy 计划。

## 后续论文队列

当前计划完成前不展开：

1. [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)：理解减少推理步数。
2. [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://proceedings.mlr.press/v37/sohl-dickstein15.html)：了解历史起点。
3. Improved DDPM、Score-based SDE、Latent Diffusion：暂不进入当前学习范围。

## 当前进度

- [ ] G0：建立全局图景
- [ ] G1：构造目标数据分布
- [ ] G2：前向扩散
- [ ] G3：时间条件噪声预测
- [ ] G4：反向采样
- [ ] G5：可复现与量化评测
- [ ] G6：映射到 Diffusion Policy

当前任务：完成 DDPM 论文 Abstract、Introduction 和 Figure 2，回答 G0 的四个问题。
