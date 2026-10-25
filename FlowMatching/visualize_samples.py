"""F3 生成结果的绘图工具；参考中心只用于显示，不参与采样。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .data import CENTERS


NAMES = ("c=0: horizontal", "c=1: vertical")
COLORS = ("#2878b5", "#d95f02")


def format_axis(ax, limit: float) -> None:
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="x", ylabel="y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)


def plot_generated_samples(samples: dict[str, torch.Tensor], output_dir: Path) -> None:
    trajectories = [samples[f"trajectory_c{condition}"] for condition in (0, 1)]
    num_steps = trajectories[0].shape[0] - 1
    # 共用坐标范围，包含全部生成轨迹，避免截去离群点。
    limit = max(3.0, max(trajectory.abs().max().item() for trajectory in trajectories) + 0.3)
    snapshot_indices = [round(num_steps * fraction) for fraction in (0, 0.25, 0.5, 0.75, 1)]

    fig, axes = plt.subplots(2, 5, figsize=(16, 7), layout="constrained")
    for condition, trajectory in enumerate(trajectories):
        for column, index in enumerate(snapshot_indices):
            ax = axes[condition, column]
            points = trajectory[index]
            ax.scatter(points[:, 0], points[:, 1], s=5, alpha=0.4, color=COLORS[condition])
            ax.set_title(f"{NAMES[condition]} | t={samples['times'][index].item():g}")
            format_axis(ax, limit)
    fig.suptitle("Model-generated time slices: same initial noise, fixed condition")
    fig.savefig(output_dir / "time_slices.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for condition, trajectory in enumerate(trajectories):
        ax = axes[condition]
        points = trajectory[-1]
        ax.scatter(points[:, 0], points[:, 1], s=8, alpha=0.45,
                   color=COLORS[condition], label="Generated endpoints")
        # 没有为各生成点指定目标端点；此处仅标记目标分布的两个中心。
        centers = CENTERS[condition]
        ax.scatter(centers[:, 0], centers[:, 1], marker="x", s=80,
                   color="black", linewidths=2, label="Reference centers")
        ax.set_title(NAMES[condition])
        format_axis(ax, limit)
        ax.legend(fontsize=8)
    fig.suptitle(f"Generated endpoints after {num_steps} forward Euler steps")
    fig.savefig(output_dir / "endpoints.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for condition, trajectory in enumerate(trajectories):
        ax = axes[condition]
        # 固定取前 16 个样本，两种条件显示相同的起点索引。
        count = min(16, trajectory.shape[1])
        selected = trajectory[:, :count]
        for index in range(count):
            path = selected[:, index]
            ax.plot(path[:, 0], path[:, 1], color=plt.get_cmap("tab20")(index % 20),
                    linewidth=1.2, alpha=0.8)
        ax.scatter(selected[0, :, 0], selected[0, :, 1], marker="o", facecolors="none",
                   edgecolors="black", s=35, label="Initial noise")
        ax.scatter(selected[-1, :, 0], selected[-1, :, 1], marker="x",
                   color=COLORS[condition], s=35, label="Generated endpoint")
        ax.set_title(f"{NAMES[condition]} | first {count} samples")
        format_axis(ax, limit)
        ax.legend(fontsize=8)
    fig.suptitle("Model-generated trajectories (not endpoint-paired training paths)")
    fig.savefig(output_dir / "trajectories.png", dpi=150)
    plt.close(fig)
