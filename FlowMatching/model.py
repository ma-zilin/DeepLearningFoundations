"""F2：根据位置、时间和任务条件预测二维速度。"""

import torch
from torch import nn


class VelocityMLP(nn.Module):
    """接收 [B, 5] 输入，输出 [B, 2] 速度预测。

    输入由 conditioning.build_velocity_input 整理，各列依次为
    [x, y, t, c0, c1]。本类不采样路径、不计算监督标签。
    网络采用两个宽度为 64 的 ReLU 隐藏层，最后一层保持线性，
    允许速度分量为正或负，且不限制在固定区间。
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, network_input: torch.Tensor) -> torch.Tensor:
        if network_input.ndim != 2 or network_input.shape[1] != 5:
            raise ValueError("network_input 必须是 [B, 5] 张量")
        if not network_input.is_floating_point():
            raise ValueError("network_input 必须是浮点张量")
        return self.net(network_input)
