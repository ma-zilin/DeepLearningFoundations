"""一维点质量环境。

策略输出有界加速度，环境使用半隐式欧拉法依次更新速度和位置。
这个最小环境只服务于 Behavior Cloning 学习实验，不包含 reward 或 Gym 接口。
"""

from __future__ import annotations

from typing import Any

import numpy as np


class PointMassEnv:
    """带随机目标和初始状态的一维点质量环境。"""

    def __init__(
        self,
        seed: int = 0,
        dt: float = 0.05,
        a_max: float = 4.0,
        v_max: float = 3.0,
        position_tolerance: float = 0.02,
        velocity_tolerance: float = 0.02,
        max_steps: int = 200,
    ) -> None:
        if dt <= 0:
            raise ValueError("dt 必须大于 0")
        if a_max <= 0 or v_max <= 0:
            raise ValueError("a_max 和 v_max 必须大于 0")
        if position_tolerance < 0 or velocity_tolerance < 0:
            raise ValueError("成功判据的容差不能为负数")
        if max_steps <= 0:
            raise ValueError("max_steps 必须大于 0")

        self.dt = float(dt)
        self.a_max = float(a_max)
        self.v_max = float(v_max)
        self.position_tolerance = float(position_tolerance)
        self.velocity_tolerance = float(velocity_tolerance)
        self.max_steps = int(max_steps)

        self.rng = np.random.default_rng(seed)
        self.x = 0.0
        self.v = 0.0
        self.x_goal = 0.0
        self.step_count = 0
        self.done = True

    def _observation(self) -> np.ndarray:
        """返回策略可见的 observation。"""
        return np.array([self.x, self.v, self.x_goal], dtype=np.float32)

    def reset(self) -> np.ndarray:
        """随机开始一个新 episode，并返回初始 observation。"""
        self.x_goal = float(self.rng.uniform(-1.0, 1.0))
        distance = float(self.rng.uniform(0.5, 1.5))
        direction = float(self.rng.choice((-1.0, 1.0)))
        self.x = self.x_goal + direction * distance
        self.v = float(self.rng.uniform(-0.5, 0.5))

        self.step_count = 0
        self.done = False
        return self._observation()

    def step(self, action: float | np.ndarray) -> tuple[np.ndarray, bool, dict[str, Any]]:
        """执行一个加速度动作，返回下一观测、终止标记和终止详情。"""
        if self.done:
            raise RuntimeError("episode 已结束，请先调用 reset()")

        action_array = np.asarray(action, dtype=np.float64)
        if action_array.size != 1:
            raise ValueError("action 必须是标量或只含一个元素的数组")

        requested_action = float(action_array.item())
        if not np.isfinite(requested_action):
            raise ValueError("action 必须是有限数值")

        executed_action = float(
            np.clip(requested_action, -self.a_max, self.a_max)
        )

        # 半隐式欧拉：先更新速度，再用新速度更新位置。
        self.v = float(
            np.clip(
                self.v + executed_action * self.dt,
                -self.v_max,
                self.v_max,
            )
        )
        self.x = self.x + self.v * self.dt
        self.step_count += 1

        position_error = abs(self.x - self.x_goal)
        success = (
            position_error <= self.position_tolerance
            and abs(self.v) <= self.velocity_tolerance
        )
        timeout = not success and self.step_count >= self.max_steps
        self.done = success or timeout

        info = {
            "requested_action": requested_action,
            "executed_action": executed_action,
            "success": success,
            "timeout": timeout,
            "position_error": position_error,
            "step_count": self.step_count,
        }
        return self._observation(), self.done, info
