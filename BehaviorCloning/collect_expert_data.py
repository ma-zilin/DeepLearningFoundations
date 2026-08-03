"""生成 B1 专家轨迹数据和一条代表性轨迹图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from BehaviorCloning.dataset import collect_expert_episodes, save_expert_dataset


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
DATASET_PATH = ARTIFACT_DIR / "b1_expert_episodes.npz"
TRAJECTORY_PLOT_PATH = ARTIFACT_DIR / "b1_expert_episode_000.png"


def plot_episode(
    dataset: dict[str, np.ndarray],
    episode_id: int,
    output_path: str | Path,
    dt: float = 0.05,
) -> Path:
    """绘制指定 episode 的位置、速度和实际动作。"""
    indices = np.flatnonzero(dataset["episode_id"] == episode_id)
    if len(indices) == 0:
        raise ValueError(f"数据集中不存在 episode {episode_id}")

    observations = dataset["observation"][indices]
    next_observations = dataset["next_observation"][indices]
    actions = dataset["expert_action"][indices, 0]

    state_times = np.arange(len(indices) + 1) * dt
    action_times = np.arange(len(indices)) * dt
    positions = np.concatenate((observations[:, 0], next_observations[-1:, 0]))
    velocities = np.concatenate((observations[:, 1], next_observations[-1:, 1]))
    goal = float(observations[0, 2])

    figure, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(state_times, positions, label="position")
    axes[0].axhline(goal, color="tab:red", linestyle="--", label="goal")
    axes[0].set_ylabel("position")
    axes[0].legend()

    axes[1].plot(state_times, velocities, color="tab:orange")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("velocity")

    axes[2].step(action_times, actions, where="post", color="tab:green")
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_ylabel("acceleration")
    axes[2].set_xlabel("time (s)")

    figure.suptitle(f"PD expert rollout: episode {episode_id}")
    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def main() -> None:
    dataset = collect_expert_episodes(num_episodes=100, seed=0)
    dataset_path = save_expert_dataset(dataset, DATASET_PATH)
    plot_path = plot_episode(dataset, episode_id=0, output_path=TRAJECTORY_PLOT_PATH)

    episode_lengths = np.bincount(dataset["episode_id"])
    success_count = int(dataset["success"].sum())
    timeout_count = int(dataset["timeout"].sum())
    print(f"dataset: {dataset_path}")
    print(f"trajectory plot: {plot_path}")
    print(f"episodes: {len(episode_lengths)}")
    print(f"transitions: {len(dataset['episode_id'])}")
    print(f"successes: {success_count}, timeouts: {timeout_count}")
    print(
        "episode length: "
        f"min={episode_lengths.min()}, "
        f"mean={episode_lengths.mean():.2f}, "
        f"max={episode_lengths.max()}"
    )


if __name__ == "__main__":
    main()
