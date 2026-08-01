"""G1：构造并检查二维双峰高斯混合分布。"""

import os
from pathlib import Path

MATPLOTLIB_CACHE_DIR = Path("/tmp/deep_learning_foundations_matplotlib")
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from bimodal_data import (
    CENTERS,
    build_bimodal_samples,
    sample_bimodal_components,
)


NUM_SAMPLES = 4096
SEED = 0
STANDARD_DEVIATIONS = (0.3, 1.0)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def print_statistics(
    samples: torch.Tensor,
    mode_ids: torch.Tensor,
    std: float,
) -> None:
    """打印形状、峰比例、均值和峰内标准差。"""
    print(f"\n=== 目标标准差：{std:.1f} ===")
    print(f"数据形状：{list(samples.shape)}")
    print(f"全体均值：{samples.mean(dim=0).tolist()}")

    counts = torch.bincount(mode_ids, minlength=2)
    names = ("左峰", "右峰")

    for mode_id, name in enumerate(names):
        cluster = samples[mode_ids == mode_id]
        ratio = counts[mode_id].item() / samples.shape[0]
        print(
            f"{name}：数量={counts[mode_id].item()}，"
            f"比例={ratio:.4f}，"
            f"经验均值={cluster.mean(dim=0).tolist()}，"
            f"峰内标准差={cluster.std(dim=0).tolist()}"
        )


def save_scatter_plot(
    samples: torch.Tensor,
    mode_ids: torch.Tensor,
    std: float,
) -> Path:
    """保存使用固定坐标范围的散点图，便于公平比较两种标准差。"""
    figure, axis = plt.subplots(figsize=(7, 5))
    colors = ("tab:blue", "tab:orange")
    names = ("Left mode", "Right mode")

    for mode_id, (color, name) in enumerate(zip(colors, names)):
        cluster = samples[mode_ids == mode_id]
        axis.scatter(
            cluster[:, 0].numpy(),
            cluster[:, 1].numpy(),
            s=8,
            alpha=0.35,
            c=color,
            label=name,
            edgecolors="none",
        )

    axis.scatter(
        CENTERS[:, 0].numpy(),
        CENTERS[:, 1].numpy(),
        marker="x",
        s=100,
        linewidths=2,
        c="black",
        label="Target centers",
    )
    axis.set(
        title=f"2D Bimodal Gaussian (within-mode std={std:.1f})",
        xlabel="x",
        ylabel="y",
        xlim=(-6.0, 6.0),
        ylim=(-4.0, 4.0),
    )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()

    output_path = OUTPUT_DIR / f"bimodal_std_{std:.1f}.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator().manual_seed(SEED)
    mode_ids, standard_noise = sample_bimodal_components(
        num_samples=NUM_SAMPLES,
        generator=generator,
    )

    for std in STANDARD_DEVIATIONS:
        samples = build_bimodal_samples(mode_ids, standard_noise, std)

        assert samples.shape == (NUM_SAMPLES, 2)
        assert torch.isfinite(samples).all()
        assert torch.bincount(mode_ids, minlength=2).min() > 0

        print_statistics(samples, mode_ids, std)
        output_path = save_scatter_plot(samples, mode_ids, std)
        print(f"散点图：{output_path}")

    print(
        "\n实验解释：该数据是包含两个高概率峰的多模态分布。"
        "峰内标准差描述同一模式附近样本的离散程度；"
        "峰间距离描述两个模式的分离程度。"
        "标准差从0.3增至1.0时，峰中心不变，但峰变宽、重叠增多。"
    )


if __name__ == "__main__":
    main()
