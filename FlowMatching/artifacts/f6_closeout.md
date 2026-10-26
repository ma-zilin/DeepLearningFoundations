# Flow Matching 阶段收尾

2026-10-26：F0–F6 按核心机制学习、关键对话检查与二维有界验证范围收尾。阶段进度见 [学习计划](../LEARNING_PLAN.md)。代码主要由助手按明确请求实现；运行通过与学习者独立实现、完整理论掌握分别判断。

## 公式与代码

| 环节 | 核心关系或职责 | 实现 |
| --- | --- | --- |
| 目标数据 | 按 c 生成水平或竖直双峰 | [sample_target_data](../data.py) |
| 路径监督 | $x_t=(1-t)x_0+t x_1$，$u_{\mathrm{pair}}=x_1-x_0$ | [linear_path / sample_path_batch](../path.py) |
| 网络输入 | 位置、时间、任务独热编码拼为 [B,5] | [build_velocity_input](../conditioning.py) |
| 速度预测 | [B,5] → [B,2]，不读取端点 | [VelocityMLP](../model.py) |
| 训练 | 重采样噪声与时间，速度平方误差更新参数 | [train_step](../train.py) |
| 生成 | $h=1/N$；更新位置后重新查询速度，条件与参数固定 | [euler_integrate / sample_model](../sample.py) |
| 评估 | 峰比例、中心、逐坐标离散程度、远离比例 | [distribution_metrics](../evaluate.py) |

配对标签不是每个位置唯一的“真实速度”；理想 MSE 回归得到给定位置、时间与条件的平均速度场。它用于推进样本分布，不是把全部终点直接平均成一个动作。

## 产物与权重索引

| 证据 | 入口 |
| --- | --- |
| F1 数据、端点与有限差分检查 | [checks.json](f1_data/checks.json)、[路径图](f1_paths/pairs_and_velocities.png) |
| F2 固定监督诊断 | [500 步结果](f2_overfit/metrics.json)、[2000 步结果](f2_overfit_2000/metrics.json) |
| F2 正式训练与加载 | [结果](f2_training/metrics.json)、[曲线](f2_training/loss.png)、[复算](f2_training/reload_checks.json) |
| F3 积分与生成 | [解析检查](f3_euler_checks/checks.json)、[配置](f3_sampling/config.json)、[完整轨迹复现](f3_sampling/reproduction.json) |
| F4 步数与条件实验 | [汇总与全部失败配置](f4_evaluation/summary.md)、[观察与边界](f4_evaluation/observations.md)、[独立复算](f4_evaluation/independent_checks.json)、[复现](f4_evaluation/reproduction.json) |
| F5 同口径质量对照 | [汇总](f5_ddpm_reevaluation/summary.md)、[完整 CSV](f5_ddpm_reevaluation/quality_comparison.csv)、[图像](f5_ddpm_reevaluation/endpoints_comparison.png)、[来源指纹](f5_ddpm_reevaluation/config.json)、[独立复算](f5_ddpm_reevaluation/independent_checks.json)、[复现](f5_ddpm_reevaluation/reproduction.json) |

以下 checkpoint、数据和轨迹按忽略规则仅保留本地，不随本仓库推送：

| 权重 | 用途 | SHA-256 |
| --- | --- | --- |
| `FlowMatching/artifacts/f2_training/best.pt` | 第 1800 次更新；F3–F5 的 FM 生成权重 | `c539889269d9ef9b879c6e4cb9e61bb9c06fee6cf2dd40d8cc26e1e9ad1d8371` |
| `FlowMatching/artifacts/f2_training/last.pt` | 第 2000 次更新；最终训练状态 | `d7e9db99034d94b49577d0155fc2685e4109720a9411fb22aa9ef9139d021f69` |
| `diffusion/outputs/noise_predictor_g3b.pt` | 原 DDPM 权重；F5 质量重测 | `61e9d9d0cd97e5a8bf9656086d2011adc643f23864f055b107d4075bddd41c35` |

## 结果与未验证边界

- 固定监督 500 次更新未达预声明门槛；只增加预算至 2000 次后通过，原失败记录保留。后段 loss 波动的唯一原因未定位。
- 正式训练共 2000 次，最佳固定验证 loss=2.98239470，优于零速度 6.07949305 和未训练网络 6.13372231；验证误差未分解为条件方差与模型误差。
- F4 共 18 组，13 组通过。100 步下两种条件各通过三个采样 seed，水平条件最大远离比例 4.96%，接近 5% 阈值；仍有峰间残留。五组失败不删除、不改阈值。步数增加不保证全部指标单调改善，N=100 也不是精确 ODE 解。
- F4 每批 4096 样本的中位耗时约 5.3–5.4 / 26.5–26.6 / 52.5–53.6 ms，对应 10 / 50 / 100 步，包含检查和完整轨迹内存操作。早期部分计时偏高，原因未定位。
- F5 统一规则后 DDPM 1000 步与 FM 水平条件 100 步均为 3/3 通过，最大中心误差为 0.0582 / 0.0688，最大远离比例为 2.00% / 4.96%。这是两个当前模型的观察，不是公平算法排名。
- DDPM / FM 参数量为 21,250 / 4,674；输入条件、架构及训练设置不同。DDPM 源码默认 5000 次更新不等于旧权重实际训练次数的证据；FM 使用实际第 1800 次更新的权重。DDPM 共同口径墙钟耗时未测。
- 动作块与条件、生成时间与物理时间、队列与滚动执行已完成机制映射及引导式讨论；用户已指出闭环完成率是任务能力的关键证据。平均动作陷阱经过多轮纠正，尚未完成无提示独立推导或全部知识点书面验收。
- 未开展动作策略实现或训练、机器人闭环与真机实验、多训练 seed 泛化评估、精确 ODE 参考、最优传输耦合或论文性能复现。

## 复核与下一入口

仓库根目录、已有 `dl-foundations` 环境中可复核；采样复现需要上述本地权重和保存张量：

```bash
python -m FlowMatching.test_evaluate
python -m FlowMatching.sample --verify FlowMatching/artifacts/f3_sampling
python -m FlowMatching.evaluate --verify FlowMatching/artifacts/f4_evaluation
python -m FlowMatching.reevaluate_ddpm --verify FlowMatching/artifacts/f5_ddpm_reevaluation
```

这些命令不训练；采样复核会重新生成并核对已有结果，不重新计时。仅克隆仓库不能取得被忽略的权重与样本，不宣称无需本地数据即可完整复现。

下一方向为 ImitationPolicyLearning 的 ACT 阶段定位，学习范围与预算另定。该入口不自动授权 ACT 训练或恢复暂缓的 Diffusion Policy 长程实验。
