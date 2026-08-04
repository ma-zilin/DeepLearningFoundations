---
type: learning-plan
topic: behavior-cloning
status: active
current_gate: B4
started: 2026-08-02
progress_mode: gate-based
---

# Behavior Cloning 学习地图

## 目标与定位

本阶段只补齐进入 Diffusion Policy 前必需的行为克隆最小闭环：

> 专家轨迹 → 监督学习数据集 → 状态条件策略 → 开环评估 → 闭环 rollout → 分布偏移分析 → Diffusion Policy 映射

Behavior Cloning（BC）把专家示范中的 observation-action 对当作监督学习样本，学习策略：

$$
\pi_\theta(o_t) \approx a_t^*
$$

对于连续动作，最小实验使用均方误差：

$$
\mathcal{L}_{BC}
=
\mathbb{E}_{(o_t,a_t^*)\sim\mathcal{D}}
\left[
\left\|\pi_\theta(o_t)-a_t^*\right\|_2^2
\right]
$$

这里的重点不是做出高性能控制器，而是亲眼看到：**离线 action loss 较低，不保证策略闭环运行可靠。**

本计划不按天数推进。前一关没有形成代码、结果和解释证据，不进入下一关。

## 学习边界

本阶段必须完成：

- 理解 `observation`、`state`、`action`、`episode` 和时间步的关系。
- 用状态输入训练一个小型 MLP BC 策略。
- 按 episode 划分训练集、验证集和测试集。
- 区分开环 action prediction 与闭环 rollout。
- 用受控扰动观察 covariate shift 和误差累积。
- 将普通 BC 映射到 action chunk 与 Diffusion Policy。

本阶段暂不展开：

- 图像输入、CNN 编码器或端到端 visuomotor policy。
- DAgger、GAIL、逆强化学习或强化学习微调。
- Transformer、ACT 或 Diffusion Policy 训练。
- LeRobot、Push-T、真实机器人或复杂仿真平台。
- 追求 benchmark 成绩或搭建通用训练框架。

## 协作方式

- 学习者亲手实现并运行主实验。
- Codex 负责解释原理、物理直觉、数据流和验收证据。
- 学习者提交代码、曲线、rollout 结果或报错后，Codex 进行检查。
- 除非学习者明确要求，否则 Codex 不直接编写实验实现。
- 通用网络层、优化器和自动微分直接使用 PyTorch，不从零重复实现。

## 学习方式与材料

本阶段采用：

> gate-based 小实验为主线，课程片段补充概念，到达对应问题后再阅读论文。

课程和论文不单独形成另一条待办路线。每一关按以下循环推进：

1. 明确本关问题和出口。
2. 阅读本关所需的最小原始材料。
3. 学习者用自己的话复述关键概念和数据流。
4. 学习者亲手完成最小实现或受控实验。
5. 提交代码、输出、曲线或失败轨迹作为证据。
6. 检查物理含义、关键权衡和常见陷阱，达到通过标准后再进入下一关。

材料优先级为：

> 本学习计划的当前 gate → 当前实验暴露的问题 → 对应课程片段 → 延伸论文与理论

### 课程材料

选择 Berkeley CS 185/285 的 Behavior Cloning 部分作为辅助材料，不完整学习整门强化学习课程：

- B0：[`Lecture 2: Behavioral Cloning`](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-2.pdf) 中专家示范、监督学习和连续动作策略部分。
- B4–B5：Lecture 2 的 distribution shift 部分，以及 [`BC Distributional Shift`](https://rail.eecs.berkeley.edu/deeprlcourse/static/sections/section-2-2.pdf) 的问题设定和结论。完整 regret bound 证明不是通过本计划的必要条件。
- B6：[`Lecture 3: Behavioral Cloning Part 2`](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-3.pdf) 中 multimodal behavior、action chunking 和 diffusion policy 部分。

选择该课程片段，是因为它从机器人序列决策的视角连接了监督学习、闭环分布偏移、action chunk 和 Diffusion Policy，与 B0–B6 的出口直接对应。暂不学习其中的 RL Basics、Policy Gradient、Actor-Critic、完整课程作业、预训练和多任务学习内容。

### 论文材料

- B0–B4 没有必读论文，先通过最小实验建立 observation-action 数据、开环预测和闭环控制之间的联系。
- B5 在观察到 covariate shift 后，选读 DAgger 原论文 [`A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning`](https://arxiv.org/abs/1011.0686) 的摘要、Introduction 和算法思想；不要求阅读完整证明，也不实现 DAgger。
- B6 必读 [`Diffusion Policy: Visuomotor Policy Learning via Action Diffusion`](https://arxiv.org/abs/2303.04137v5) 的 Abstract、Introduction、方法总览和 action sequence / conditioning / receding-horizon 相关内容。实验细节和全部 benchmark 表格按问题回查。

### BC 与强化学习的分类边界

Behavior Cloning 通常属于 imitation learning，而不是 reinforcement learning。二者都学习策略，但监督信号不同：

| Behavior Cloning | Reinforcement Learning |
| --- | --- |
| 从专家示范 `(observation, expert action)` 学习 | 从智能体与环境交互及 reward 学习 |
| 直接拟合专家动作 | 优化期望累计回报 |
| 训练本质是监督学习 | 训练本质是序列决策优化 |
| 通常不需要 reward 或主动探索 | 通常需要 reward，并涉及探索问题 |

BC 被放入强化学习课程，是因为部署时二者都形成 `policy → action → environment → next observation` 的闭环。策略的当前动作会影响未来输入，所以 BC 也需要研究普通独立同分布监督学习中不突出的 covariate shift 和误差累积。

在本路线中，Diffusion Policy 使用专家示范和条件去噪目标生成动作序列，仍属于 behavior cloning / imitation learning；使用扩散模型或输出机器人策略，本身不会使它成为强化学习。只有引入 reward 并以累计回报优化策略时，才进入强化学习阶段。

## 最小主实验：一维点质量跟踪

使用一个不依赖机器人框架的离散点质量系统。状态为位置和速度，目标是到达指定位置并停稳。

观测：

$$
o_t=[x_t,\ v_t,\ x_{goal}]
$$

动作是有界加速度：

$$
a_t\in[-a_{max},a_{max}]
$$

动力学：

$$
v_{t+1}=\operatorname{clip}(v_t+a_t\Delta t,-v_{max},v_{max})
$$

$$
x_{t+1}=x_t+v_{t+1}\Delta t
$$

专家策略使用 PD 控制器：

$$
a_t^*
=
\operatorname{clip}
\left(
k_p(x_{goal}-x_t)-k_dv_t,
-a_{max},a_{max}
\right)
$$

选择该任务是因为：

- 专家动作的来源透明，可以解释其物理意义。
- 数据生成、策略训练和闭环执行可以完全分离。
- MLP 足够完成任务，不会把 CNN 学习混入 BC 主问题。
- 可以通过初始状态和外部扰动主动制造训练分布外状态。

所有动力学参数、成功判据和数据范围必须在首次正式实验前固定并记录。调试配置可以修改，但不能在看到测试结果后偷偷修改成功判据。

## 最终出口

完成本计划后，应当能够：

1. 画出 `expert rollout → dataset → MLP policy → closed-loop rollout` 的完整数据流。
2. 解释 BC 为什么是监督学习，以及监督标签从哪里来。
3. 独立完成 episode 级数据划分、归一化、训练、checkpoint 加载和评估。
4. 同时报告开环预测误差和闭环任务指标，不用其中一个替代另一个。
5. 用实验说明策略如何因自己的预测误差进入专家数据未覆盖的状态。
6. 解释单步 MSE BC、action chunk BC 与 Diffusion Policy 的联系和区别。

---

## B0：建立行为克隆全局图景

### 本轮问题

1. 专家、环境、示范数据集和学习策略分别是什么？
2. BC 的输入、监督标签和 loss 分别是什么？
3. 训练时与部署时，策略接收到的 observation 来自哪里？
4. 为什么 BC 不需要 reward，也仍然能学习行为？

### 必须画出的数据流

```text
训练：专家与环境交互 → 保存 episode → 拆成 (observation, action) → 监督训练 policy

部署：policy 输出 action → 环境更新状态 → 新 observation 返回 policy → 重复执行
```

### 通过标准

- 能区分 expert policy、learned policy 和 environment dynamics。
- 能说明 action 是监督标签，不是环境自动提供的真值。
- 能指出训练数据中的 observation 由专家轨迹产生，而部署 observation 由学习策略自己的历史动作产生。
- 能解释为什么上述差异可能导致 covariate shift。

## B1：实现环境、专家与轨迹数据

### 学习内容

- 连续状态、连续动作与离散控制时间步。
- PD 控制器中位置误差项与阻尼项的作用。
- transition 与 episode 的区别。
- 时间顺序、终止条件和随机初始状态。

### 实验

1. 实现点质量环境的 `reset` 和 `step`。
2. 实现 PD expert，并先执行若干完整 episode。
3. 记录每个时间步的：
   - `episode_id`
   - `t`
   - `observation_t`
   - `expert_action_t`
   - `next_observation`
   - `done`
4. 绘制至少一条专家轨迹的 `position-time`、`velocity-time` 和 `action-time` 曲线。

### 关键检查

- 保存的 action 必须是从同一行 `observation_t` 计算出的专家动作。
- `next_observation` 只能由执行该 action 后的环境产生。
- episode 终止后不能把下一次 reset 的状态拼接到上一条轨迹。
- 动作限幅后，数据集中保存的是实际执行的动作。

### 通过标准

- 专家能在预先定义的训练初始状态范围内稳定完成任务。
- 所有数组长度和张量形状一致，无错一帧对齐。
- 能任选一行数据，完整说明它在物理系统中的含义。
- 能解释为什么只有独立随机采样的状态—动作对，不能替代真实轨迹数据。

## B2：构造无泄漏的数据集

### 学习内容

- 为什么相邻时间步高度相关。
- 为什么应按 episode 而不是按单行 transition 随机划分。
- observation 与 action 归一化的目的。
- 训练统计量泄漏如何污染验证和测试结果。

### 实验

1. 固定数据生成 seed，并保存原始 episode。
2. 按完整 episode 划分 train、validation 和 test。
3. 只用 train split 计算 observation/action 的均值和标准差。
4. 保存 split 清单和 normalization statistics。
5. 打印每个 split 的 episode 数、transition 数、状态范围和动作范围。

### 通过标准

- 同一 episode 不会跨越多个 split。
- validation/test 信息没有参与归一化统计量计算。
- 数据读取后能够恢复 `observation → expert action` 的对应关系。
- 能解释逐行随机切分为何可能让验证 loss 过度乐观。

## B3：训练状态输入的 MLP BC

### 学习内容

- MLP policy 的输入输出形状。
- 连续动作 MSE 的含义。
- `train`/`eval` 模式、优化器、batch 与 epoch。
- training loss、validation loss 与 checkpoint selection。

### 实验

1. 先对一个固定小 batch 过拟合，检查数据流和梯度。
2. 使用全部 train split 正式训练。
3. 同时记录 train/validation loss。
4. 按预先规定的 validation 指标保存 checkpoint。
5. 在新进程中加载 checkpoint，对固定 observation 输出动作。

### 基线

至少比较：

- 恒零动作预测器。
- train split 的平均动作预测器。
- 未训练的同结构 MLP。
- 训练后的 BC policy。

### 通过标准

- 小 batch 可以被明显过拟合。
- 训练过程中无 `NaN/Inf`，参数得到有限梯度。
- 训练后的模型在 validation/test action MSE 上优于简单基线。
- checkpoint 重新加载后的固定输入输出与保存前一致。
- 能逐层说明 `[batch, observation_dim] → [batch, action_dim]` 的形状变化。

## B4：区分开环误差与闭环表现

### 本轮问题

开环 action MSE 回答的是：

> 在专家访问过的 observation 上，策略能否预测接近专家的 action？

闭环 rollout 回答的是：

> 策略连续执行自己的 action 后，能否把环境带到目标状态？

二者不是同一个问题。

### 实验

对 expert、简单基线和 BC policy 使用相同的一组测试初始状态，分别执行完整 rollout。至少记录：

- 成功次数与总次数。
- 最终位置误差。
- episode 长度。
- 最大速度或预先规定的安全越界次数。
- action 的变化量或其他简洁的平滑性指标。
- 典型成功轨迹与失败轨迹。

### 受控要求

- 测试 seed 和初始状态列表在比较前固定。
- expert 与 BC 使用相同动力学、控制频率和终止条件。
- 不因 BC 表现不佳而单独放宽成功判据。
- 报告成功次数和总次数，不只报告百分比。

### 通过标准

- 开环评估与闭环评估由不同函数完成。
- 能找到并分析至少一个闭环 rollout，而不只展示 loss 曲线。
- 能说明某次动作小误差如何改变下一时刻 observation，并继续影响后续预测。
- 不把低 validation loss 写成“策略已经学会控制”的充分证据。

## B5：观察 covariate shift 与误差累积

### 学习内容

专家数据来自状态分布：

$$
o_t\sim d_{\pi^*}
$$

部署时 observation 来自学习策略诱导的状态分布：

$$
o_t\sim d_{\pi_\theta}
$$

即使单步误差很小，一次偏差也可能把系统带到示范数据较少覆盖的位置，使后续预测进一步恶化。

### 受控实验

保持模型与 checkpoint 不变，只改变评测条件：

1. 在训练范围内的初始状态执行 rollout。
2. 在训练范围边缘执行 rollout。
3. 在某个固定时间步对位置或速度加入预先定义的小扰动。

将访问到的 `(position, velocity)` 与专家训练数据覆盖范围画在同一张图中，并对比闭环指标。

### 通过标准

- 扰动实验只改变一个明确变量。
- 能指出策略何时开始离开专家数据的主要覆盖区域。
- 能结合轨迹而非只凭最终成功率解释误差累积。
- 能说明“增加网络规模”为什么不一定解决缺少状态覆盖的问题。
- 知道 DAgger 的核心思路是让专家标注学习策略实际访问的状态，但本阶段不实现 DAgger。

## B6：映射到 action chunk 与 Diffusion Policy

### 单步 BC

普通确定性 BC 直接预测下一步动作：

$$
\hat a_t=\pi_\theta(o_t)
$$

### Action chunk BC

策略一次预测未来一段动作：

$$
\hat A_t
=
[\hat a_t,\hat a_{t+1},\ldots,\hat a_{t+H-1}]
$$

action chunk 可以减少高层策略重新决策的次数，并显式建模动作之间的时间相关性；但 chunk 太长会降低对新 observation 和扰动的响应速度。

### Diffusion Policy

Diffusion Policy 不再让确定性 MLP 用单个 MSE 输出唯一动作，而是以 observation 为条件，通过迭代去噪生成动作序列：

$$
\epsilon_\theta(A_t^k,k,o_t)\approx\epsilon
$$

其中被加噪和生成的是动作序列，图像、机器人状态等 observation 作为条件，通常不参与动作扩散过程。

### 对应关系

| 普通 BC | Diffusion Policy |
| --- | --- |
| 专家 observation-action 数据 | 同样来自专家示范的数据 |
| 状态或图像作为 policy 条件 | 状态或图像作为去噪条件 |
| 单步 action 或确定性 action chunk | 生成式 action chunk |
| 常用 action MSE | 常用 noise prediction loss |
| 一次前向得到动作 | 多步去噪得到动作序列 |
| 多峰标签可能被平均 | 能表达条件动作分布的多个模态 |
| 仍需闭环 rollout 评估 | 同样必须闭环 rollout 评估 |

### 必须回答

1. Diffusion Policy 为什么仍属于 behavior cloning / imitation learning？
2. 哪些变量被加噪，哪些变量只作为条件？
3. 为什么单步 MSE 在多种动作都合理时可能输出平均动作？
4. action horizon、prediction horizon 和 execution horizon 分别控制什么？
5. 为什么生成更复杂的动作分布仍不能自动消除 covariate shift？
6. 为什么 Diffusion Policy 的 validation loss 也不能替代闭环成功率？

### 最终通过标准

- B0–B5 的代码、曲线和评测记录齐全。
- 能从数据、目标函数、模型输出和部署方式四方面比较普通 BC 与 Diffusion Policy。
- 能明确说出普通 BC 实验暴露了什么问题，以及 Diffusion Policy 解决和没有解决哪些问题。
- 已具备进入 Diffusion Policy 实操的前置理解，不再继续扩展 BC 主线。

## 建议产物结构

具体文件名可以在开始实现时确定，但职责应保持分离：

```text
BehaviorCloning/
├── LEARNING_PLAN.md       # 本学习地图
├── environment.*          # 动力学、reset、step 与终止条件
├── expert.*               # PD expert
├── dataset.*              # 轨迹采集、split 与 normalization
├── policy.*               # MLP BC policy
├── train.*                # 监督训练与 checkpoint
├── evaluate_open_loop.*   # action prediction 指标
├── rollout.*              # 闭环评测与扰动实验
└── artifacts/             # 曲线、轨迹图和量化结果
```

不要在学习开始前一次性搭完所有文件。每通过一个 gate，再增加下一关真正需要的最小代码。

## 进入下一阶段的边界

通过 B6 后，直接回到 VLA 路线的 Diffusion Policy 实操。以下内容不是继续停留在 BC 的理由：

- 手写更多控制器。
- 更换多个 Gym 环境重复相同结论。
- 给最小 MLP 添加复杂训练框架。
- 提前实现 DAgger、ACT 或图像 BC。

如果主实验失败，先判断失败属于：数据时间对齐、split 泄漏、归一化、模型训练、闭环接口、状态覆盖还是动力学配置。一次只验证一个假设，并保留失败轨迹。
