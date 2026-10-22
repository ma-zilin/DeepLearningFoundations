"""F1：可视化人为构造的训练路径，不涉及网络训练或生成。

在仓库根目录运行：python -m FlowMatching.visualize_paths
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .data import sample_target_data
from .path import linear_path, sample_path_batch


TIMES = (0.0, 0.25, 0.5, 0.75, 1.0)
COLORS = ("#2878b5", "#d95f02")
CONDITION_NAMES = ("c=0: horizontal", "c=1: vertical")
ARROW_DT = 0.2


def format_axis(ax, limit: float) -> None:
    """所有子图共用坐标尺度，避免缩放造成错误的几何印象。"""
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="x", ylabel="y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)


def plot_time_slices(x0, x1, c, mode_ids, limit, output_dir) -> None:
    fig, axes = plt.subplots(2, len(TIMES), figsize=(16, 7), layout="constrained")
    for column, time in enumerate(TIMES):
        # 固定同一批端点；整个 batch 使用同一个时间，而不是各行随机时间。
        t = torch.full((len(x0), 1), time, dtype=x0.dtype)
        xt, _ = linear_path(x0, x1, t)
        if time == 0:
            torch.testing.assert_close(xt, x0, rtol=0, atol=0)
        elif time == 1:
            torch.testing.assert_close(xt, x1, rtol=0, atol=0)

        for condition in (0, 1):
            ax = axes[condition, column]
            for mode in (0, 1):
                points = xt[(c == condition) & (mode_ids == mode)]
                ax.scatter(
                    points[:, 0], points[:, 1], s=5, alpha=0.35,
                    color=COLORS[mode], label=f"Endpoint mode {mode}",
                )
            ax.set_title(f"{CONDITION_NAMES[condition]} | t={time:g}")
            format_axis(ax, limit)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Time slices of fixed training pairs (no trained model)")
    fig.savefig(output_dir / "time_slices.png", dpi=150)
    plt.close(fig)


def plot_pairs_and_velocities(x0, x1, c, mode_ids, limit, pair_count, output_dir) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), layout="constrained")
    for condition in (0, 1):
        # 从该条件下选少量样本，保持原始索引对应关系，不重配端点。
        indices = torch.where(c == condition)[0][:pair_count]
        starts, ends = x0[indices], x1[indices]
        midpoint, velocity = linear_path(
            starts, ends, torch.full((len(indices), 1), 0.5, dtype=x0.dtype),
        )
        path_ax, arrow_ax = axes[condition]
        for mode in (0, 1):
            mask = mode_ids[indices] == mode
            a, b = starts[mask], ends[mask]
            # 每一列是一对端点，连接同列的两个点。
            path_ax.plot(
                torch.stack((a[:, 0], b[:, 0])).numpy(),
                torch.stack((a[:, 1], b[:, 1])).numpy(),
                color=COLORS[mode], alpha=0.4, linewidth=1,
            )
            path_ax.scatter(
                a[:, 0], a[:, 1], marker="o", facecolors="none",
                edgecolors=COLORS[mode], s=40,
            )
            path_ax.scatter(b[:, 0], b[:, 1], marker="x", color=COLORS[mode], s=40)
            arrow_ax.scatter(
                midpoint[mask, 0], midpoint[mask, 1], color=COLORS[mode], s=22,
            )
            # 使用真实坐标单位，箭头为固定时间长度 ARROW_DT 对应的位移。
            arrow_ax.quiver(
                midpoint[mask, 0], midpoint[mask, 1],
                ARROW_DT * velocity[mask, 0], ARROW_DT * velocity[mask, 1],
                angles="xy", scale_units="xy", scale=1,
                color=COLORS[mode], width=0.006,
            )
        path_ax.set_title(f"{CONDITION_NAMES[condition]} | {len(indices)} fixed pairs")
        arrow_ax.set_title(f"t=0.5 | arrows = {ARROW_DT:g} * pair velocity")
        for ax in (path_ax, arrow_ax):
            format_axis(ax, limit)
    fig.suptitle("Constructed training paths: o = start, x = end; colors = endpoint modes")
    fig.savefig(output_dir / "pairs_and_velocities.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2048, help="两种条件合计的样本数")
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--path-seed", type=int, default=1)
    parser.add_argument("--pairs-per-condition", type=int, default=12)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "f1_paths",
    )
    args = parser.parse_args()
    if args.samples < 2 or args.pairs_per_condition < 1:
        parser.error("samples 至少为 2，pairs-per-condition 必须为正整数")

    data_rng = torch.Generator(device="cpu").manual_seed(args.data_seed)
    path_rng = torch.Generator(device="cpu").manual_seed(args.path_seed)
    x1, c, mode_ids = sample_target_data(args.samples, data_rng)
    # 此处仅取起点；随机时间的 xt 不能直接当成某个时间的切片。
    x0, _, _, _ = sample_path_batch(x1, path_rng)
    counts = [int((c == condition).sum()) for condition in (0, 1)]
    if min(counts) == 0:
        parser.error("本批次缺少一种条件，请增大 samples 或更换 data-seed")

    # 线性插值及图中箭头都位于端点之间，此范围可包含所有绘制内容。
    limit = max(3.0, float(torch.stack((x0.abs().max(), x1.abs().max())).max()) + 0.3)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_time_slices(x0, x1, c, mode_ids, limit, args.output_dir)
    plot_pairs_and_velocities(
        x0, x1, c, mode_ids, limit, args.pairs_per_condition, args.output_dir,
    )
    configuration = {
        "description": "Independent-pair training paths; no trained model or ODE rollout",
        "samples": args.samples,
        "data_seed": args.data_seed,
        "path_seed": args.path_seed,
        "condition_counts": counts,
        "times": TIMES,
        "pairs_per_condition": args.pairs_per_condition,
        "arrow_displacement_dt": ARROW_DT,
        "axis_limit": limit,
        "endpoint_checks_passed": True,
        "torch_version": torch.__version__,
        "matplotlib_version": matplotlib.__version__,
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(f"条件样本数：c=0 为 {counts[0]}，c=1 为 {counts[1]}")
    print(f"固定配对的 t=0/1 端点检查通过；图像与配置已保存至 {args.output_dir}")


if __name__ == "__main__":
    main()
