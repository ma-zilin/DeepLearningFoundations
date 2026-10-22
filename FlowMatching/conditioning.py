"""F1：任务条件编码，以及速度网络的输入整理。"""

import torch
from torch.nn import functional as F


def encode_condition(
    c: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """将 [B] 的 int64 条件编号转为 [B, 2] 的浮点独热编码。

    c=0（水平双峰）编码为 [1, 0]，c=1（竖直双峰）编码为 [0, 1]。
    保留 c 所在设备，不修改原始编号。mode_ids 不参与任务条件编码。
    """
    if c.ndim != 1 or c.dtype != torch.int64:
        raise ValueError("c 必须是形状为 [B] 的 int64 张量")
    if ((c < 0) | (c > 1)).any():
        raise ValueError("任务条件 c 只能为 0 或 1")
    if not dtype.is_floating_point:
        raise ValueError("条件编码的 dtype 必须是浮点类型")

    # 独热编码使用 PyTorch 已有功能；转换为浮点类型以便与位置拼接。
    return F.one_hot(c, num_classes=2).to(dtype=dtype)


def build_velocity_input(
    xt: torch.Tensor,
    t: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """将位置、时间、任务条件沿特征维拼成 [B, 5] 输入。

    xt 为 [B, 2]，t 为 [B, 1]，二者具有相同的浮点 dtype 和 device；
    c 为同设备的 [B] int64 编号。输出各列依次为：
    [位置的 x 坐标, 位置的 y 坐标, 时间, 条件 0 分量, 条件 1 分量]。

    本函数只有输入整理，没有可学习参数。训练和生成可以共用；终点、
    速度标签和峰编号均不作为输入。
    """
    if xt.ndim != 2 or xt.shape[1] != 2 or not xt.is_floating_point():
        raise ValueError("xt 必须是 [B, 2] 浮点张量")
    if t.shape != (xt.shape[0], 1) or t.dtype != xt.dtype:
        raise ValueError("t 必须是 [B, 1]，且 dtype 与 xt 相同")
    if c.shape != (xt.shape[0],):
        raise ValueError("c 必须为 [B]，并与 xt 的样本逐行对应")
    if not (xt.device == t.device == c.device):
        raise ValueError("xt、t 和 c 必须位于同一设备")

    c_onehot = encode_condition(c, dtype=xt.dtype)
    return torch.cat((xt, t, c_onehot), dim=1)
