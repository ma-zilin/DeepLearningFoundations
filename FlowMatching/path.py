"""F1：独立配对的线性路径、中间位置和速度标签。"""

import torch


def linear_path(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """给定端点与时间，返回 (xt, velocity)，不采样随机数。

    x0、x1 必须为同形状的 [B, 2] 浮点张量，t 为 [B, 1]；三者的
    dtype 和 device 必须相同。每行的两个坐标共用一个 t，范围为 [0, 1]。
    两个输出均为 [B, 2]。velocity 是每对端点的监督标签，不是网络预测。
    """
    if x0.ndim != 2 or x0.shape[1] != 2 or x1.shape != x0.shape:
        raise ValueError("x0 和 x1 必须具有相同的 [B, 2] 形状")
    if t.shape != (x0.shape[0], 1):
        raise ValueError("t 必须为 [B, 1]，每个样本的两个坐标共用一个时间")
    if not (x0.is_floating_point() and x0.dtype == x1.dtype == t.dtype):
        raise ValueError("x0、x1 和 t 必须具有相同的浮点 dtype")
    if not (x0.device == x1.device == t.device):
        raise ValueError("x0、x1 和 t 必须位于同一设备")
    if not torch.isfinite(t).all() or ((t < 0) | (t > 1)).any():
        raise ValueError("t 必须为 [0, 1] 内的有限数值")

    # [B, 1] 的时间沿坐标维广播到 [B, 2]。
    xt = (1 - t) * x0 + t * x1
    velocity = x1 - x0
    return xt, velocity


def sample_path_batch(
    x1: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """为已有目标点构造训练样本，返回 (x0, t, xt, velocity)。

    x1 是 CPU 浮点张量，形状为 [B, 2]，可来自 data.sample_target_data。
    使用调用方提供的 CPU generator，为每行独立采样标准高斯起点
    x0 和均匀时间 t（torch.rand 的取值范围为 [0, 1)）。不重设种子。

    x0[i] 与 x1[i] 按索引配对，没有执行最优传输优化。目标点对应的
    任务条件 c 由调用方保留；本函数不打乱目标点，也不改变条件。
    x0、xt、velocity 为 [B, 2]，t 为 [B, 1]，dtype 与 x1 相同。
    """
    if x1.ndim != 2 or x1.shape[1] != 2 or x1.shape[0] == 0:
        raise ValueError("x1 必须是非空的 [B, 2] 张量")
    if x1.device.type != "cpu" or not x1.is_floating_point():
        raise ValueError("x1 必须是 CPU 浮点张量")
    if generator.device.type != "cpu":
        raise ValueError("generator 必须是 CPU 随机数生成器")

    x0 = torch.randn(
        x1.shape,
        generator=generator,
        dtype=x1.dtype,
        device="cpu",
    )
    t = torch.rand(
        (x1.shape[0], 1),
        generator=generator,
        dtype=x1.dtype,
        device="cpu",
    )
    xt, velocity = linear_path(x0, x1, t)
    return x0, t, xt, velocity
