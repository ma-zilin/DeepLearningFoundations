"""Behavior Cloning 专家轨迹的采集、保存与基本校验。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from BehaviorCloning.environment import PointMassEnv
from BehaviorCloning.expert import PDExpert


def collect_expert_episodes(
    num_episodes: int,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """采集完整专家 episode，并按 transition 保存时间对齐的数据。"""
    if num_episodes <= 0:
        raise ValueError("num_episodes 必须大于 0")

    env = PointMassEnv(seed=seed)
    expert = PDExpert(a_max=env.a_max)

    episode_ids: list[int] = []
    timesteps: list[int] = []
    observations: list[np.ndarray] = []
    expert_actions: list[float] = []
    next_observations: list[np.ndarray] = []
    dones: list[bool] = []
    successes: list[bool] = []
    timeouts: list[bool] = []

    for episode_id in range(num_episodes):
        observation = env.reset()
        done = False
        timestep = 0

        while not done:
            current_observation = observation.copy()
            requested_action = expert(current_observation)
            observation, done, info = env.step(requested_action)

            episode_ids.append(episode_id)
            timesteps.append(timestep)
            observations.append(current_observation)
            expert_actions.append(info["executed_action"])
            next_observations.append(observation.copy())
            dones.append(done)
            successes.append(info["success"])
            timeouts.append(info["timeout"])

            timestep += 1

    dataset = {
        "episode_id": np.asarray(episode_ids, dtype=np.int64),
        "t": np.asarray(timesteps, dtype=np.int64),
        "observation": np.asarray(observations, dtype=np.float32),
        "expert_action": np.asarray(expert_actions, dtype=np.float32).reshape(-1, 1),
        "next_observation": np.asarray(next_observations, dtype=np.float32),
        "done": np.asarray(dones, dtype=np.bool_),
        "success": np.asarray(successes, dtype=np.bool_),
        "timeout": np.asarray(timeouts, dtype=np.bool_),
    }
    validate_expert_dataset(dataset, expected_num_episodes=num_episodes)
    return dataset


def validate_expert_dataset(
    dataset: dict[str, np.ndarray],
    expected_num_episodes: int | None = None,
) -> None:
    """检查数组形状、episode 边界和相邻 transition 的时间对齐。"""
    required_keys = {
        "episode_id",
        "t",
        "observation",
        "expert_action",
        "next_observation",
        "done",
        "success",
        "timeout",
    }
    missing_keys = required_keys - dataset.keys()
    if missing_keys:
        raise ValueError(f"数据集缺少字段: {sorted(missing_keys)}")

    num_transitions = len(dataset["episode_id"])
    if num_transitions == 0:
        raise ValueError("数据集不能为空")

    one_dimensional_keys = ("episode_id", "t", "done", "success", "timeout")
    for key in one_dimensional_keys:
        if dataset[key].shape != (num_transitions,):
            raise ValueError(f"{key} 的形状必须是 ({num_transitions},)")

    if dataset["observation"].shape != (num_transitions, 3):
        raise ValueError("observation 的形状必须是 [num_transitions, 3]")
    if dataset["next_observation"].shape != (num_transitions, 3):
        raise ValueError("next_observation 的形状必须是 [num_transitions, 3]")
    if dataset["expert_action"].shape != (num_transitions, 1):
        raise ValueError("expert_action 的形状必须是 [num_transitions, 1]")

    for key in ("observation", "expert_action", "next_observation"):
        if not np.all(np.isfinite(dataset[key])):
            raise ValueError(f"{key} 中包含 NaN 或 Inf")

    observations = dataset["observation"]
    actions = dataset["expert_action"][:, 0]
    next_observations = dataset["next_observation"]

    expected_actions = np.clip(
        4.0 * (observations[:, 2] - observations[:, 0]) - 4.0 * observations[:, 1],
        -4.0,
        4.0,
    )
    if not np.allclose(actions, expected_actions, rtol=0.0, atol=1e-6):
        raise ValueError("expert_action 不是由同一行 observation 计算得到")

    expected_next_velocity = np.clip(
        observations[:, 1] + actions * 0.05,
        -3.0,
        3.0,
    )
    expected_next_position = observations[:, 0] + expected_next_velocity * 0.05
    expected_next_observations = np.column_stack(
        (expected_next_position, expected_next_velocity, observations[:, 2])
    )
    if not np.allclose(
        next_observations,
        expected_next_observations,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("next_observation 与 action 对应的环境动力学不一致")

    episode_ids = dataset["episode_id"]
    unique_episode_ids = np.unique(episode_ids)
    if expected_num_episodes is not None and len(unique_episode_ids) != expected_num_episodes:
        raise ValueError("实际 episode 数与预期不一致")
    if not np.array_equal(unique_episode_ids, np.arange(len(unique_episode_ids))):
        raise ValueError("episode_id 必须从 0 开始连续编号")

    for episode_id in unique_episode_ids:
        indices = np.flatnonzero(episode_ids == episode_id)
        expected_timesteps = np.arange(len(indices))
        if not np.array_equal(dataset["t"][indices], expected_timesteps):
            raise ValueError(f"episode {episode_id} 的时间步不连续")

        episode_done = dataset["done"][indices]
        if np.any(episode_done[:-1]) or not episode_done[-1]:
            raise ValueError(f"episode {episode_id} 的 done 边界错误")

        terminal_reasons = dataset["success"][indices] | dataset["timeout"][indices]
        if not np.array_equal(episode_done, terminal_reasons):
            raise ValueError(f"episode {episode_id} 的终止原因与 done 不一致")
        if np.any(dataset["success"][indices] & dataset["timeout"][indices]):
            raise ValueError(f"episode {episode_id} 同时标记 success 和 timeout")

        if len(indices) > 1 and not np.allclose(
            dataset["next_observation"][indices[:-1]],
            dataset["observation"][indices[1:]],
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError(f"episode {episode_id} 的相邻 observation 错一帧")


def save_expert_dataset(dataset: dict[str, np.ndarray], output_path: str | Path) -> Path:
    """将原始 transition 数组压缩保存为 NPZ。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **dataset)
    return output_path
