"""一维点质量任务的 PD 专家策略。"""

from __future__ import annotations

import numpy as np


class PDExpert:
    """根据位置误差和当前速度输出有界加速度。"""

    def __init__(
        self,
        kp: float = 4.0,
        kd: float = 4.0,
        a_max: float = 4.0,
    ) -> None:
        if kp <= 0:
            raise ValueError("kp 必须大于 0")
        if kd < 0:
            raise ValueError("kd 不能为负数")
        if a_max <= 0:
            raise ValueError("a_max 必须大于 0")

        self.kp = float(kp)
        self.kd = float(kd)
        self.a_max = float(a_max)

    def __call__(self, observation: np.ndarray) -> float:
        """根据 ``[x, v, x_goal]`` 计算专家加速度。"""
        observation_array = np.asarray(observation, dtype=np.float64)
        if observation_array.shape != (3,):
            raise ValueError("observation 的形状必须是 (3,)")
        if not np.all(np.isfinite(observation_array)):
            raise ValueError("observation 必须只包含有限数值")

        x, v, x_goal = observation_array
        position_error = x_goal - x
        raw_action = self.kp * position_error - self.kd * v
        return float(np.clip(raw_action, -self.a_max, self.a_max))
