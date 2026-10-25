"""F4 评估图：显示全部终点、固定索引轨迹和质量—耗时对照。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .data import CENTERS


COLORS = ("#2878b5", "#d95f02")


def format_axis(ax, limit: float) -> None:
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="x", ylabel="y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)


def plot_evaluation(all_samples: dict[int, dict], rows: list[dict], protocol: dict, output_dir: Path) -> None:
    steps_list = protocol["steps"]
    # 所有 seed / N / c 共用范围，不裁掉离群点，也不挑选最佳 seed。
    endpoint_limit = max(3.0, max(t[-1].abs().max().item() for s in all_samples.values()
                                   for k, t in s.items() if k != "x0") + 0.3)
    trajectory_limit = max(3.0, max(t.abs().max().item() for s in all_samples.values()
                                     for k, t in s.items() if k != "x0") + 0.3)
    for seed, samples in all_samples.items():
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
        for c in (0, 1):
            for column, steps in enumerate(steps_list):
                ax = axes[c, column]
                points = samples[f"trajectory_n{steps}_c{c}"][-1]
                ax.scatter(points[:, 0], points[:, 1], s=3, alpha=0.3, color=COLORS[c])
                centers = CENTERS[c]
                ax.scatter(centers[:, 0], centers[:, 1], marker="x", color="black", s=60)
                for center in centers:
                    ax.add_patch(plt.Circle(center.tolist(), protocol["thresholds"]["nearest_center_radius"],
                                            fill=False, linestyle="--", color="gray", alpha=0.5))
                row = next(r for r in rows if (r["seed"], r["condition"], r["steps"]) == (seed, c, steps))
                ax.set_title(f"c={c}, N={steps} | {'PASS' if row['passed'] else 'FAIL'} | outside={row['outside_fraction']:.1%}")
                format_axis(ax, endpoint_limit)
        fig.suptitle(f"All 4096 endpoints per condition | seed={seed} | identical initial noise across panels")
        fig.savefig(output_dir / f"endpoints_seed{seed}.png", dpi=140)
        plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
        for c in (0, 1):
            for column, steps in enumerate(steps_list):
                ax = axes[c, column]
                selected = samples[f"trajectory_n{steps}_c{c}"][:, :16]
                for index in range(selected.shape[1]):
                    path = selected[:, index]
                    ax.plot(path[:, 0], path[:, 1], color=plt.get_cmap("tab20")(index), alpha=0.8, linewidth=1)
                ax.scatter(selected[0, :, 0], selected[0, :, 1], facecolors="none", edgecolors="black", s=25)
                ax.scatter(selected[-1, :, 0], selected[-1, :, 1], color=COLORS[c], marker="x", s=25)
                ax.set_title(f"c={c}, N={steps}")
                format_axis(ax, trajectory_limit)
        fig.suptitle(f"First 16 sample indices | seed={seed} | circles=start, crosses=end")
        fig.savefig(output_dir / f"trajectories_seed{seed}.png", dpi=140)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for c in (0, 1):
        for seed in protocol["sampling_seeds"]:
            subset = sorted((r for r in rows if r["condition"] == c and r["seed"] == seed), key=lambda r: r["steps"])
            label = f"c={c}, seed={seed}"
            marker = ("o", "s", "^")[seed]
            times = [r["median_batch_ms"] for r in subset]
            axes[0].plot([r["steps"] for r in subset], times, marker=marker, color=COLORS[c], alpha=0.7, label=label)
            axes[1].plot(times, [max(r[f"group{m}_center_error"] for m in (0, 1))
                                if all(r[f"group{m}_center_error"] is not None for m in (0, 1)) else float("nan")
                                for r in subset], marker=marker, color=COLORS[c], alpha=0.7)
            axes[2].plot(times, [r["outside_fraction"] for r in subset], marker=marker, color=COLORS[c], alpha=0.7)
    axes[0].set(xlabel="Euler steps / batched network calls", ylabel="Median latency per 4096 samples (ms)")
    axes[1].set(xlabel="Batch latency (ms)", ylabel="Worst of two center errors")
    axes[2].set(xlabel="Batch latency (ms)", ylabel="Fraction outside radius 0.9")
    axes[1].axhline(protocol["thresholds"]["center_error_max"], linestyle="--", color="gray")
    axes[2].axhline(protocol["thresholds"]["outside_fraction_max"], linestyle="--", color="gray")
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle("Fixed checkpoint: sampling cost and selected quality metrics (see CSV for all criteria)")
    fig.savefig(output_dir / "quality_cost.png", dpi=150)
    plt.close(fig)
