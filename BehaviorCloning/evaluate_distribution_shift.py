"""B5：比较范围内部、范围边缘和单次状态扰动下的闭环表现。"""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from BehaviorCloning.environment import PointMassEnv
from BehaviorCloning.evaluate_closed_loop import _make_mlp_policy_function
from BehaviorCloning.expert import PDExpert
from BehaviorCloning.policy import BCPolicy


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
METADATA_PATH = ARTIFACT_DIR / "b2_dataset_metadata.json"
CHECKPOINT_PATH = ARTIFACT_DIR / "b3_bc_policy.pt"
RESULT_PATH = ARTIFACT_DIR / "b5_distribution_shift_results.json"
COVERAGE_PLOT_PATH = ARTIFACT_DIR / "b5_state_coverage.png"

PERTURB_AFTER_STEPS = 20
PERTURB_VELOCITY_MAGNITUDE = 0.5

PolicyFunction = Callable[[np.ndarray], float]


def _make_initial_states(
    goals: tuple[float, ...],
    distance: float,
    velocities: tuple[float, ...],
) -> list[np.ndarray]:
    """生成目标、左右方向和初速度的确定性组合。"""
    states = []
    for goal in goals:
        for direction in (-1.0, 1.0):
            for velocity in velocities:
                states.append(
                    np.array(
                        [goal + direction * distance, velocity, goal],
                        dtype=np.float32,
                    )
                )
    return states


def _rollout(
    policy_function: PolicyFunction,
    initial_observation: np.ndarray,
    expert: PDExpert,
    perturb: bool,
) -> dict[str, Any]:
    """执行闭环 rollout，并在策略访问的状态上测量专家动作差异。"""
    env = PointMassEnv(seed=0)
    observation = env.reset_to(initial_observation)
    observations = [observation.copy()]
    executed_actions: list[float] = []
    expert_action_errors: list[float] = []
    perturbation: dict[str, Any] | None = None
    done = False
    final_info: dict[str, Any] | None = None

    while not done:
        if perturb and env.step_count == PERTURB_AFTER_STEPS:
            before = observation.copy()
            away_direction = float(np.sign(observation[0] - observation[2]))
            if away_direction == 0.0:
                away_direction = 1.0
            velocity_delta = away_direction * PERTURB_VELOCITY_MAGNITUDE
            observation = env.perturb_state(velocity_delta=velocity_delta)
            observations[-1] = observation.copy()
            perturbation = {
                "after_steps": env.step_count,
                "velocity_delta": velocity_delta,
                "observation_before": before.tolist(),
                "observation_after": observation.tolist(),
            }

        action = policy_function(observation.copy())
        expert_action = expert(observation)
        expert_action_errors.append(abs(action - expert_action))
        observation, done, info = env.step(action)
        executed_actions.append(float(info["executed_action"]))
        observations.append(observation.copy())
        final_info = info

    if final_info is None:
        raise RuntimeError("rollout 没有执行任何环境步")
    if perturb and perturbation is None:
        raise RuntimeError("episode 在预定时间步之前结束，未能施加扰动")

    observation_array = np.asarray(observations, dtype=np.float32)
    action_array = np.asarray(executed_actions, dtype=np.float32)
    action_delta = np.abs(np.diff(action_array))
    return {
        "initial_observation": observation_array[0].tolist(),
        "success": bool(final_info["success"]),
        "timeout": bool(final_info["timeout"]),
        "episode_length": int(final_info["step_count"]),
        "final_position_error": float(final_info["position_error"]),
        "maximum_absolute_velocity": float(np.max(np.abs(observation_array[:, 1]))),
        "mean_absolute_action_delta": (
            float(action_delta.mean()) if len(action_delta) else 0.0
        ),
        "mean_expert_action_error": float(np.mean(expert_action_errors)),
        "maximum_expert_action_error": float(np.max(expert_action_errors)),
        "perturbation": perturbation,
        "observations": observation_array,
    }


def _aggregate(
    rollouts: list[dict[str, Any]],
    train_position_range: tuple[float, float],
    train_velocity_range: tuple[float, float],
) -> dict[str, Any]:
    """汇总成功率、控制质量和越出训练状态包围盒的比例。"""
    visited_states = np.concatenate(
        [rollout["observations"][:, :2] for rollout in rollouts], axis=0
    )
    outside = (
        (visited_states[:, 0] < train_position_range[0])
        | (visited_states[:, 0] > train_position_range[1])
        | (visited_states[:, 1] < train_velocity_range[0])
        | (visited_states[:, 1] > train_velocity_range[1])
    )
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
        "mean_expert_action_error": float(
            np.mean([rollout["mean_expert_action_error"] for rollout in rollouts])
        ),
        "maximum_expert_action_error": float(
            max(rollout["maximum_expert_action_error"] for rollout in rollouts)
        ),
        "visited_state_count": int(len(visited_states)),
        "outside_train_xy_box_count": int(outside.sum()),
        "outside_train_xy_box_fraction": float(outside.mean()),
    }


def _serializable_rollout(index: int, rollout: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "initial_observation",
        "success",
        "timeout",
        "episode_length",
        "final_position_error",
        "maximum_absolute_velocity",
        "mean_absolute_action_delta",
        "mean_expert_action_error",
        "maximum_expert_action_error",
        "perturbation",
    )
    return {"case_id": index, **{key: rollout[key] for key in keys}}


def _plot_coverage(
    train_xy: np.ndarray,
    bc_rollouts: dict[str, list[dict[str, Any]]],
) -> None:
    condition_titles = {
        "interior": "interior initial states",
        "edge": "edge initial states",
        "perturbed_interior": "interior + velocity perturbation",
    }
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True, sharey=True)
    for axis, (condition_name, title) in zip(
        axes, condition_titles.items(), strict=True
    ):
        axis.hexbin(
            train_xy[:, 0],
            train_xy[:, 1],
            gridsize=45,
            mincnt=1,
            cmap="Greys",
            alpha=0.65,
        )
        for rollout in bc_rollouts[condition_name]:
            observations = rollout["observations"]
            axis.plot(
                observations[:, 0],
                observations[:, 1],
                color="tab:blue",
                linewidth=1.0,
                alpha=0.75,
            )
            axis.scatter(
                observations[0, 0], observations[0, 1], color="tab:green", s=20
            )
            axis.scatter(
                observations[0, 2], 0.0, color="tab:orange", marker="*", s=35
            )
            if rollout["perturbation"] is not None:
                perturbed = rollout["perturbation"]["observation_after"]
                axis.scatter(
                    perturbed[0], perturbed[1], color="tab:red", marker="x", s=35
                )
        axis.set_title(title)
        axis.set_xlabel("position")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("velocity")
    figure.suptitle(
        "BC visited states over expert train coverage\n"
        "green=start, orange star=goal, red x=post-perturbation"
    )
    figure.tight_layout()
    figure.savefig(COVERAGE_PLOT_PATH, dpi=160)
    plt.close(figure)


def main() -> None:
    torch.set_num_threads(1)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    with np.load(ARTIFACT_DIR / metadata["raw_dataset"]) as stored_dataset:
        dataset = {key: stored_dataset[key] for key in stored_dataset.files}

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    trained_policy = BCPolicy(**checkpoint["model_config"])
    trained_policy.load_state_dict(checkpoint["model_state_dict"])
    bc_function = _make_mlp_policy_function(
        trained_policy, checkpoint["normalization"]
    )
    expert = PDExpert()
    policies: dict[str, PolicyFunction] = {
        "expert": expert,
        "trained_bc": bc_function,
    }

    interior_states = _make_initial_states(
        goals=(-0.5, 0.5), distance=1.0, velocities=(-0.2, 0.2)
    )
    edge_states = _make_initial_states(
        goals=(-1.0, 1.0), distance=1.5, velocities=(-0.5, 0.5)
    )
    conditions = {
        "interior": (interior_states, False),
        "edge": (edge_states, False),
        "perturbed_interior": (interior_states, True),
    }

    train_episode_ids = np.asarray(metadata["episode_ids"]["train"])
    train_mask = np.isin(dataset["episode_id"], train_episode_ids)
    train_xy = dataset["observation"][train_mask, :2]
    train_position_range = (float(train_xy[:, 0].min()), float(train_xy[:, 0].max()))
    train_velocity_range = (float(train_xy[:, 1].min()), float(train_xy[:, 1].max()))

    all_rollouts: dict[str, dict[str, list[dict[str, Any]]]] = {}
    results: dict[str, Any] = {
        "checkpoint_best_epoch": checkpoint["best_epoch"],
        "protocol": {
            "interior": {
                "goals": [-0.5, 0.5],
                "distance": 1.0,
                "velocities": [-0.2, 0.2],
            },
            "edge": {
                "goals": [-1.0, 1.0],
                "distance": 1.5,
                "velocities": [-0.5, 0.5],
            },
            "perturbation": {
                "base_condition": "interior",
                "after_steps": PERTURB_AFTER_STEPS,
                "velocity_magnitude": PERTURB_VELOCITY_MAGNITUDE,
                "direction": "away_from_goal",
            },
            "train_xy_box": {
                "position": list(train_position_range),
                "velocity": list(train_velocity_range),
            },
        },
        "conditions": {},
    }

    for condition_name, (initial_states, perturb) in conditions.items():
        condition_rollouts: dict[str, list[dict[str, Any]]] = {}
        for policy_name, policy_function in policies.items():
            condition_rollouts[policy_name] = [
                _rollout(policy_function, state, expert, perturb)
                for state in initial_states
            ]
        all_rollouts[condition_name] = condition_rollouts

        results["conditions"][condition_name] = {"policies": {}}
        for policy_name, rollouts in condition_rollouts.items():
            results["conditions"][condition_name]["policies"][policy_name] = {
                "aggregate": _aggregate(
                    rollouts, train_position_range, train_velocity_range
                ),
                "cases": [
                    _serializable_rollout(index, rollout)
                    for index, rollout in enumerate(rollouts)
                ],
            }

    bc_rollouts = {
        name: condition_rollouts["trained_bc"]
        for name, condition_rollouts in all_rollouts.items()
    }
    _plot_coverage(train_xy, bc_rollouts)
    RESULT_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    for condition_name, condition_results in results["conditions"].items():
        print(condition_name)
        for policy_name, policy_results in condition_results["policies"].items():
            print(f"  {policy_name}: {policy_results['aggregate']}")
    print(f"results: {RESULT_PATH}")
    print(f"coverage plot: {COVERAGE_PLOT_PATH}")


if __name__ == "__main__":
    main()
