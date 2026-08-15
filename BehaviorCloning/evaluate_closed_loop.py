"""B4：在相同测试初始状态上比较专家、基线与 BC 的闭环表现。"""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from BehaviorCloning.environment import PointMassEnv
from BehaviorCloning.expert import PDExpert
from BehaviorCloning.policy import BCPolicy


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
METADATA_PATH = ARTIFACT_DIR / "b2_dataset_metadata.json"
CHECKPOINT_PATH = ARTIFACT_DIR / "b3_bc_policy.pt"
RESULT_PATH = ARTIFACT_DIR / "b4_closed_loop_results.json"
TRAJECTORY_PLOT_PATH = ARTIFACT_DIR / "b4_success_failure_trajectories.png"

PolicyFunction = Callable[[np.ndarray], float]


def _load_test_initial_states(
    dataset: dict[str, np.ndarray],
    test_episode_ids: np.ndarray,
) -> list[tuple[int, np.ndarray]]:
    """按 metadata 中的顺序提取每个 test episode 的首个 observation。"""
    initial_states: list[tuple[int, np.ndarray]] = []
    for episode_id in test_episode_ids:
        indices = np.flatnonzero(
            (dataset["episode_id"] == episode_id) & (dataset["t"] == 0)
        )
        if len(indices) != 1:
            raise ValueError(f"episode {episode_id} 应恰好有一个 t=0 transition")
        initial_states.append(
            (int(episode_id), dataset["observation"][indices[0]].copy())
        )
    return initial_states


def _make_mlp_policy_function(
    policy: BCPolicy,
    normalization: dict[str, list[float]],
) -> PolicyFunction:
    """把使用标准化量的 MLP 包装成输入原始 observation、输出原始 action 的函数。"""
    observation_mean = np.asarray(
        normalization["observation_mean"], dtype=np.float32
    )
    observation_std = np.asarray(
        normalization["observation_std"], dtype=np.float32
    )
    action_mean = np.asarray(normalization["action_mean"], dtype=np.float32)
    action_std = np.asarray(normalization["action_std"], dtype=np.float32)
    policy.eval()

    def predict(observation: np.ndarray) -> float:
        normalized_observation = (
            np.asarray(observation, dtype=np.float32) - observation_mean
        ) / observation_std
        with torch.no_grad():
            normalized_action = policy(
                torch.from_numpy(normalized_observation).unsqueeze(0)
            ).numpy()[0]
        raw_action = normalized_action * action_std + action_mean
        return float(raw_action.item())

    return predict


def rollout_policy(
    policy_function: PolicyFunction,
    initial_observation: np.ndarray,
) -> dict[str, Any]:
    """让一个策略从指定状态闭环运行到成功或超时。"""
    env = PointMassEnv(seed=0)
    observation = env.reset_to(initial_observation)
    observations = [observation.copy()]
    requested_actions: list[float] = []
    executed_actions: list[float] = []
    done = False
    final_info: dict[str, Any] | None = None

    while not done:
        requested_action = policy_function(observation.copy())
        observation, done, info = env.step(requested_action)
        requested_actions.append(requested_action)
        executed_actions.append(float(info["executed_action"]))
        observations.append(observation.copy())
        final_info = info

    if final_info is None:
        raise RuntimeError("rollout 没有执行任何环境步")

    observation_array = np.asarray(observations, dtype=np.float32)
    action_array = np.asarray(executed_actions, dtype=np.float32)
    action_delta = np.abs(np.diff(action_array))
    return {
        "initial_observation": observation_array[0].tolist(),
        "success": bool(final_info["success"]),
        "timeout": bool(final_info["timeout"]),
        "episode_length": int(final_info["step_count"]),
        "final_position_error": float(final_info["position_error"]),
        "final_absolute_velocity": float(abs(observation_array[-1, 1])),
        "maximum_absolute_velocity": float(np.max(np.abs(observation_array[:, 1]))),
        "mean_absolute_action_delta": (
            float(action_delta.mean()) if len(action_delta) else 0.0
        ),
        "observations": observation_array,
        "requested_actions": np.asarray(requested_actions, dtype=np.float32),
        "executed_actions": action_array,
    }


def _aggregate_rollouts(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总一个策略在全部固定初始状态上的闭环指标。"""
    return {
        "success_count": int(sum(rollout["success"] for rollout in rollouts)),
        "total_episodes": len(rollouts),
        "timeout_count": int(sum(rollout["timeout"] for rollout in rollouts)),
        "mean_final_position_error": float(
            np.mean([rollout["final_position_error"] for rollout in rollouts])
        ),
        "mean_episode_length": float(
            np.mean([rollout["episode_length"] for rollout in rollouts])
        ),
        "maximum_absolute_velocity": float(
            max(rollout["maximum_absolute_velocity"] for rollout in rollouts)
        ),
        "mean_absolute_action_delta": float(
            np.mean([rollout["mean_absolute_action_delta"] for rollout in rollouts])
        ),
    }


def _serializable_rollout(
    episode_id: int,
    rollout: dict[str, Any],
) -> dict[str, Any]:
    """只保存逐 episode 指标；完整时间序列由轨迹图呈现。"""
    keys = (
        "initial_observation",
        "success",
        "timeout",
        "episode_length",
        "final_position_error",
        "final_absolute_velocity",
        "maximum_absolute_velocity",
        "mean_absolute_action_delta",
    )
    return {"episode_id": episode_id, **{key: rollout[key] for key in keys}}


def _choose_matched_comparison(
    all_rollouts: dict[str, list[dict[str, Any]]],
) -> tuple[tuple[str, dict[str, Any]], tuple[str, dict[str, Any]], int]:
    """选择同一初始状态上的一条成功轨迹和一条失败轨迹。"""
    for index, trained_rollout in enumerate(all_rollouts["trained_bc"]):
        if not trained_rollout["success"] and all_rollouts["expert"][index]["success"]:
            return (
                ("expert", all_rollouts["expert"][index]),
                ("trained BC", trained_rollout),
                index,
            )

    for baseline_name in ("zero_action", "train_mean_action", "untrained_mlp"):
        for index, baseline_rollout in enumerate(all_rollouts[baseline_name]):
            trained_rollout = all_rollouts["trained_bc"][index]
            if trained_rollout["success"] and not baseline_rollout["success"]:
                return (
                    ("trained BC", trained_rollout),
                    (baseline_name.replace("_", " "), baseline_rollout),
                    index,
                )

    raise RuntimeError("没有找到同一初始状态上的成功/失败对照轨迹")


def _plot_matched_trajectories(
    success_item: tuple[str, dict[str, Any]],
    failure_item: tuple[str, dict[str, Any]],
    episode_id: int,
) -> None:
    """绘制同一 test 初始状态上的典型成功与失败轨迹。"""
    figure, axes = plt.subplots(3, 2, figsize=(11, 8), sharex="col")
    for column, (name, rollout) in enumerate((success_item, failure_item)):
        observations = rollout["observations"]
        actions = rollout["executed_actions"]
        state_time = np.arange(len(observations)) * 0.05
        action_time = np.arange(len(actions)) * 0.05

        axes[0, column].plot(state_time, observations[:, 0], label="position")
        axes[0, column].axhline(
            observations[0, 2], color="black", linestyle="--", label="goal"
        )
        axes[0, column].set_ylabel("position")
        axes[0, column].legend()

        axes[1, column].plot(state_time, observations[:, 1], color="tab:orange")
        axes[1, column].axhline(0.0, color="black", linewidth=0.8)
        axes[1, column].set_ylabel("velocity")

        axes[2, column].step(action_time, actions, where="post", color="tab:green")
        axes[2, column].set_ylabel("executed action")
        axes[2, column].set_xlabel("time (s)")
        outcome = "success" if rollout["success"] else "failure"
        axes[0, column].set_title(f"{name}: {outcome}")

        for axis in axes[:, column]:
            axis.grid(alpha=0.3)

    figure.suptitle(f"Matched closed-loop comparison: test episode {episode_id}")
    figure.tight_layout()
    figure.savefig(TRAJECTORY_PLOT_PATH, dpi=160)
    plt.close(figure)


def main() -> None:
    torch.set_num_threads(1)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    with np.load(ARTIFACT_DIR / metadata["raw_dataset"]) as stored_dataset:
        dataset = {key: stored_dataset[key] for key in stored_dataset.files}

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    trained_policy = BCPolicy(**checkpoint["model_config"])
    trained_policy.load_state_dict(checkpoint["model_state_dict"])

    torch.manual_seed(3)
    untrained_policy = BCPolicy(**checkpoint["model_config"])
    normalization = checkpoint["normalization"]
    train_mean_action = float(normalization["action_mean"][0])
    expert = PDExpert()
    policies: dict[str, PolicyFunction] = {
        "expert": expert,
        "zero_action": lambda observation: 0.0,
        "train_mean_action": lambda observation: train_mean_action,
        "untrained_mlp": _make_mlp_policy_function(untrained_policy, normalization),
        "trained_bc": _make_mlp_policy_function(trained_policy, normalization),
    }

    test_episode_ids = np.asarray(metadata["episode_ids"]["test"], dtype=np.int64)
    initial_states = _load_test_initial_states(dataset, test_episode_ids)
    all_rollouts = {
        policy_name: [
            rollout_policy(policy_function, initial_observation)
            for _, initial_observation in initial_states
        ]
        for policy_name, policy_function in policies.items()
    }

    expected_initial_states = [state.tolist() for _, state in initial_states]
    for policy_rollouts in all_rollouts.values():
        actual_initial_states = [
            rollout["initial_observation"] for rollout in policy_rollouts
        ]
        if actual_initial_states != expected_initial_states:
            raise RuntimeError("不同策略没有使用完全相同的测试初始状态")

    results: dict[str, Any] = {
        "test_episode_ids": test_episode_ids.tolist(),
        "test_initial_observations": expected_initial_states,
        "environment_config": {
            "dt": 0.05,
            "a_max": 4.0,
            "v_max": 3.0,
            "position_tolerance": 0.02,
            "velocity_tolerance": 0.02,
            "max_steps": 200,
        },
        "policies": {},
    }
    for policy_name, policy_rollouts in all_rollouts.items():
        results["policies"][policy_name] = {
            "aggregate": _aggregate_rollouts(policy_rollouts),
            "episodes": [
                _serializable_rollout(episode_id, rollout)
                for (episode_id, _), rollout in zip(
                    initial_states, policy_rollouts, strict=True
                )
            ],
        }

    success_item, failure_item, comparison_index = _choose_matched_comparison(
        all_rollouts
    )
    comparison_episode_id = initial_states[comparison_index][0]
    results["plotted_comparison"] = {
        "episode_id": comparison_episode_id,
        "success_policy": success_item[0],
        "failure_policy": failure_item[0],
    }
    _plot_matched_trajectories(
        success_item, failure_item, comparison_episode_id
    )
    RESULT_PATH.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    for policy_name, policy_results in results["policies"].items():
        print(f"{policy_name}: {policy_results['aggregate']}")
    print(f"matched comparison: {results['plotted_comparison']}")
    print(f"results: {RESULT_PATH}")
    print(f"trajectory plot: {TRAJECTORY_PLOT_PATH}")


if __name__ == "__main__":
    main()
