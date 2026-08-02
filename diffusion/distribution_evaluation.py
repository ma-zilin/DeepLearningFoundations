"""G5：使用多个采样随机种子量化评测二维 DDPM 的生成分布。"""

import csv
import os
from dataclasses import dataclass
from pathlib import Path

MATPLOTLIB_CACHE_DIR = Path("/tmp/deep_learning_foundations_matplotlib")
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from bimodal_data import CENTERS
from reverse_sampling import (
    CHECKPOINT_PATH,
    load_model_and_schedule,
    sample_reverse_process,
)


SAMPLING_SEEDS = (0, 1, 2)
NUM_SAMPLES = 4096
MIN_MODE_RATIO = 0.40
MAX_MODE_RATIO = 0.60
MAX_CENTER_ERROR = 0.30
MAX_ABS_Y_MEAN = 0.15
MIN_MEAN_WITHIN_STD = 0.15
MAX_MEAN_WITHIN_STD = 0.55
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


@dataclass(frozen=True)
class DistributionMetrics:
    """一批二维生成样本的双峰分布指标。"""

    sampling_seed: int
    left_ratio: float
    right_ratio: float
    left_center: tuple[float, float]
    right_center: tuple[float, float]
    left_center_error: float
    right_center_error: float
    y_mean: float
    left_std: tuple[float, float]
    right_std: tuple[float, float]
    mean_within_std: float
    passed: bool


def tensor_pair(values: torch.Tensor) -> tuple[float, float]:
    """将包含两个标量的张量转换为便于记录的 Python tuple。"""
    return float(values[0].item()), float(values[1].item())


def evaluate_generated_distribution(
    samples: torch.Tensor,
    sampling_seed: int,
) -> DistributionMetrics:
    """按当前二维双峰目标的已知结构计算量化指标。"""
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError(
            f"samples 必须具有形状 [N, 2]，实际为 {list(samples.shape)}"
        )
    if samples.shape[0] < 2:
        raise ValueError("至少需要两个样本才能计算分布指标")
    if not torch.isfinite(samples).all():
        raise ValueError("生成样本中存在 NaN 或 Inf")

    left = samples[samples[:, 0] < 0]
    right = samples[samples[:, 0] >= 0]
    if left.shape[0] < 2 or right.shape[0] < 2:
        raise ValueError("至少一个模式的样本少于两个，无法计算峰内标准差")

    left_ratio = left.shape[0] / samples.shape[0]
    right_ratio = right.shape[0] / samples.shape[0]
    left_center_tensor = left.mean(dim=0)
    right_center_tensor = right.mean(dim=0)
    left_std_tensor = left.std(dim=0)
    right_std_tensor = right.std(dim=0)
    left_center_error = torch.linalg.vector_norm(
        left_center_tensor - CENTERS[0]
    ).item()
    right_center_error = torch.linalg.vector_norm(
        right_center_tensor - CENTERS[1]
    ).item()
    y_mean = samples[:, 1].mean().item()
    mean_within_std = torch.cat(
        (left_std_tensor, right_std_tensor)
    ).mean().item()

    passed = (
        MIN_MODE_RATIO <= left_ratio <= MAX_MODE_RATIO
        and MIN_MODE_RATIO <= right_ratio <= MAX_MODE_RATIO
        and left_center_error <= MAX_CENTER_ERROR
        and right_center_error <= MAX_CENTER_ERROR
        and abs(y_mean) <= MAX_ABS_Y_MEAN
        and MIN_MEAN_WITHIN_STD
        <= mean_within_std
        <= MAX_MEAN_WITHIN_STD
    )
    return DistributionMetrics(
        sampling_seed=sampling_seed,
        left_ratio=left_ratio,
        right_ratio=right_ratio,
        left_center=tensor_pair(left_center_tensor),
        right_center=tensor_pair(right_center_tensor),
        left_center_error=left_center_error,
        right_center_error=right_center_error,
        y_mean=y_mean,
        left_std=tensor_pair(left_std_tensor),
        right_std=tensor_pair(right_std_tensor),
        mean_within_std=mean_within_std,
        passed=passed,
    )


def print_metrics(metrics: DistributionMetrics) -> None:
    """打印单个采样随机种子的评测结果。"""
    status = "PASS" if metrics.passed else "FAIL"
    print(f"\n=== sampling seed {metrics.sampling_seed}: {status} ===")
    print(
        f"mode ratio: left={metrics.left_ratio:.4f}, "
        f"right={metrics.right_ratio:.4f}"
    )
    print(
        f"left center={metrics.left_center}, "
        f"error={metrics.left_center_error:.4f}"
    )
    print(
        f"right center={metrics.right_center}, "
        f"error={metrics.right_center_error:.4f}"
    )
    print(f"y mean={metrics.y_mean:.4f}")
    print(
        f"within-mode std: left={metrics.left_std}, "
        f"right={metrics.right_std}, "
        f"mean={metrics.mean_within_std:.4f}"
    )


def save_metrics_csv(metrics_by_seed: list[DistributionMetrics]) -> Path:
    """将各采样随机种子的指标保存为 CSV。"""
    output_path = OUTPUT_DIR / "distribution_metrics.csv"
    fieldnames = [
        "sampling_seed",
        "left_ratio",
        "right_ratio",
        "left_center_x",
        "left_center_y",
        "right_center_x",
        "right_center_y",
        "left_center_error",
        "right_center_error",
        "y_mean",
        "left_std_x",
        "left_std_y",
        "right_std_x",
        "right_std_y",
        "mean_within_std",
        "passed",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for metrics in metrics_by_seed:
            writer.writerow(
                {
                    "sampling_seed": metrics.sampling_seed,
                    "left_ratio": metrics.left_ratio,
                    "right_ratio": metrics.right_ratio,
                    "left_center_x": metrics.left_center[0],
                    "left_center_y": metrics.left_center[1],
                    "right_center_x": metrics.right_center[0],
                    "right_center_y": metrics.right_center[1],
                    "left_center_error": metrics.left_center_error,
                    "right_center_error": metrics.right_center_error,
                    "y_mean": metrics.y_mean,
                    "left_std_x": metrics.left_std[0],
                    "left_std_y": metrics.left_std[1],
                    "right_std_x": metrics.right_std[0],
                    "right_std_y": metrics.right_std[1],
                    "mean_within_std": metrics.mean_within_std,
                    "passed": metrics.passed,
                }
            )
    return output_path


def save_seed_comparison(
    generated_by_seed: list[tuple[int, torch.Tensor]],
) -> Path:
    """并排保存不同采样随机种子的最终生成分布。"""
    figure, axes = plt.subplots(
        1,
        len(generated_by_seed),
        figsize=(15, 4.5),
        sharex=True,
        sharey=True,
    )
    for axis, (sampling_seed, samples) in zip(axes, generated_by_seed):
        axis.scatter(
            samples[:, 0].numpy(),
            samples[:, 1].numpy(),
            s=5,
            alpha=0.3,
            edgecolors="none",
        )
        axis.scatter(
            CENTERS[:, 0].numpy(),
            CENTERS[:, 1].numpy(),
            marker="x",
            s=90,
            linewidths=2,
            color="black",
        )
        axis.set(
            title=f"sampling seed={sampling_seed}",
            xlabel="x",
            xlim=(-4.0, 4.0),
            ylim=(-2.0, 2.0),
        )
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)

    axes[0].set_ylabel("y")
    figure.suptitle("DDPM Generated Distribution Across Sampling Seeds")
    figure.tight_layout()
    output_path = OUTPUT_DIR / "distribution_across_seeds.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    model, betas, alphas, alpha_bars = load_model_and_schedule(
        CHECKPOINT_PATH
    )

    metrics_by_seed: list[DistributionMetrics] = []
    generated_by_seed: list[tuple[int, torch.Tensor]] = []
    for sampling_seed in SAMPLING_SEEDS:
        generator = torch.Generator().manual_seed(sampling_seed)
        snapshots = sample_reverse_process(
            model=model,
            num_samples=NUM_SAMPLES,
            betas=betas,
            alphas=alphas,
            alpha_bars=alpha_bars,
            generator=generator,
        )
        final_samples = snapshots[-1][1]
        metrics = evaluate_generated_distribution(
            samples=final_samples,
            sampling_seed=sampling_seed,
        )
        metrics_by_seed.append(metrics)
        generated_by_seed.append((sampling_seed, final_samples))
        print_metrics(metrics)

    metrics_path = save_metrics_csv(metrics_by_seed)
    comparison_path = save_seed_comparison(generated_by_seed)
    passed_count = sum(metrics.passed for metrics in metrics_by_seed)
    print(f"\n通过种子数={passed_count}/{len(metrics_by_seed)}")
    print(f"量化指标：{metrics_path}")
    print(f"多随机种子对比图：{comparison_path}")
    if passed_count != len(metrics_by_seed):
        raise RuntimeError("至少一个采样随机种子未通过 G5 量化指标")


if __name__ == "__main__":
    main()
