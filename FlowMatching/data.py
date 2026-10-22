"""F1：按任务条件生成水平或竖直的二维双峰目标数据。"""

import torch


# 三个维度依次是：任务条件、条件内的峰编号、二维坐标。
CENTERS = torch.tensor(
    [
        [[-2.0, 0.0], [2.0, 0.0]],
        [[0.0, -2.0], [0.0, 2.0]],
    ],
    dtype=torch.float32,
    device="cpu",
)
PEAK_STD = 0.3


def sample_target_data(
    n: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """生成目标数据，返回 (x1, c, mode_ids)。

    c=0 对应水平双峰，c=1 对应竖直双峰；条件和峰编号分别等概率采样。
    x1 是 CPU float32 张量，形状为 [n, 2]；两个编号都是 CPU int64
    张量，形状为 [n]。mode_ids 供分峰统计使用，不是网络的任务条件。

    generator 必须是 CPU 生成器，由调用方创建并设置种子。本函数只推进
    它的随机序列，不重设种子。不同调用可使用不同生成器构造训练和验证数据。
    """
    if n <= 0:
        raise ValueError("n 必须是正整数")
    if generator.device.type != "cpu":
        raise ValueError("generator 必须是 CPU 随机数生成器")

    c = torch.randint(
        low=0,
        high=2,
        size=(n,),
        generator=generator,
        dtype=torch.int64,
        device="cpu",
    )
    mode_ids = torch.randint(
        low=0,
        high=2,
        size=(n,),
        generator=generator,
        dtype=torch.int64,
        device="cpu",
    )

    # 按每个样本的两个编号选中心，再加入独立的二维峰内随机偏移。
    centers = CENTERS[c, mode_ids]
    standard_noise = torch.randn(
        (n, 2),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    x1 = centers + PEAK_STD * standard_noise
    return x1, c, mode_ids
