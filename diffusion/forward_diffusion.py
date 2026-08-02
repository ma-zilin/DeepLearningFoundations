"""G2：DDPM 线性噪声 schedule、前向采样与二维验证实验。"""

import os
from pathlib import Path

MATPLOTLIB_CACHE_DIR = Path("/tmp/deep_learning_foundations_matplotlib")
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from bimodal_data import generate_bimodal_data


NUM_SAMPLES = 4096
DATA_STD = 0.3
SEED = 0
T = 1000
BETA_START = 1e-4
BETA_END = 2e-2
SNAPSHOT_INDICES = (49, 199, 499, 999)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def build_noise_schedule(
    total_steps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造 beta、alpha 和 alpha_bar。"""
    betas = torch.linspace(beta_start, beta_end, total_steps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    assert torch.all(betas[1:] > betas[:-1])
    assert torch.all(alpha_bars[1:] < alpha_bars[:-1])
    return betas, alphas, alpha_bars


def q_sample(
    x_0: torch.Tensor,
    timestep_indices: int | torch.Tensor,
    alpha_bars: torch.Tensor,
    *,
    epsilon: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """使用闭式公式一步采样 q(x_t | x_0)。"""
    timesteps = torch.as_tensor(timestep_indices, device=x_0.device)
    if timesteps.ndim == 0:
        coefficient_shape: tuple[int, ...] = ()
    elif timesteps.ndim == 1 and timesteps.shape[0] == x_0.shape[0]:
        coefficient_shape = (x_0.shape[0],) + (1,) * (x_0.ndim - 1)
    else:
        raise ValueError(
            "timestep_indices 必须是标量或形状为 [batch_size] 的一维张量"
        )

    if torch.any(timesteps < 0) or torch.any(timesteps >= alpha_bars.shape[0]):
        raise ValueError(
            f"时间步下标必须位于 [0, {alpha_bars.shape[0] - 1}]"
        )

    if epsilon is None:
        epsilon = torch.randn(
            x_0.shape,
            dtype=x_0.dtype,
            device=x_0.device,
            generator=generator,
        )
    elif epsilon.shape != x_0.shape:
        raise ValueError(
            f"epsilon.shape 必须等于 x_0.shape，实际为 "
            f"{list(epsilon.shape)} 和 {list(x_0.shape)}"
        )

    alpha_bar_t = alpha_bars.to(x_0.device)[timesteps].reshape(
        coefficient_shape
    )
    x_t = (
        torch.sqrt(alpha_bar_t) * x_0
        + torch.sqrt(1.0 - alpha_bar_t) * epsilon
    )
    return x_t, epsilon


def empirical_covariance(samples: torch.Tensor) -> torch.Tensor:
    """计算形状为 [N, D] 的样本协方差矩阵。"""
    centered = samples - samples.mean(dim=0, keepdim=True)
    return centered.T @ centered / (samples.shape[0] - 1)


def print_schedule(
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
) -> None:
    """打印 schedule 的形状和关键时间点。"""
    print("=== Noise schedule ===")
    print(f"betas.shape={list(betas.shape)}")
    print(f"alphas.shape={list(alphas.shape)}")
    print(f"alpha_bars.shape={list(alpha_bars.shape)}")
    print(f"beta[0]={betas[0].item():.8f}")
    print(f"beta[999]={betas[-1].item():.8f}")
    for timestep_index in (0, *SNAPSHOT_INDICES):
        print(
            f"alpha_bar[{timestep_index}]="
            f"{alpha_bars[timestep_index].item():.8f}"
        )


def print_distribution_statistics(name: str, samples: torch.Tensor) -> None:
    """打印经验均值和协方差。"""
    print(f"\n{name}")
    print(f"shape={list(samples.shape)}")
    print(f"mean={samples.mean(dim=0).tolist()}")
    print(f"covariance={empirical_covariance(samples).tolist()}")


def save_snapshots(
    snapshots: list[tuple[str, torch.Tensor]],
    mode_ids: torch.Tensor,
) -> Path:
    """使用统一坐标范围保存前向扩散快照。"""
    figure, axes = plt.subplots(
        1,
        len(snapshots),
        figsize=(20, 4),
        sharex=True,
        sharey=True,
    )
    colors = ("tab:blue", "tab:orange")
    for axis, (title, samples) in zip(axes, snapshots):
        for mode_id, color in enumerate(colors):
            cluster = samples[mode_ids == mode_id]
            axis.scatter(
                cluster[:, 0].numpy(),
                cluster[:, 1].numpy(),
                s=4,
                alpha=0.28,
                c=color,
                edgecolors="none",
            )
        axis.set(
            title=title,
            xlabel="x",
            xlim=(-6.0, 6.0),
            ylim=(-4.0, 4.0),
        )
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)

    axes[0].set_ylabel("y")
    figure.suptitle("DDPM Forward Diffusion on a 2D Bimodal Distribution")
    figure.tight_layout()
    output_path = OUTPUT_DIR / "forward_diffusion.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(SEED)
    x_0, mode_ids = generate_bimodal_data(
        num_samples=NUM_SAMPLES,
        std=DATA_STD,
        generator=generator,
    )
    betas, alphas, alpha_bars = build_noise_schedule(
        total_steps=T,
        beta_start=BETA_START,
        beta_end=BETA_END,
    )

    print_schedule(betas, alphas, alpha_bars)
    print_distribution_statistics("clean x_0", x_0)
    snapshots = [("clean", x_0)]
    for timestep_index in SNAPSHOT_INDICES:
        x_t, _ = q_sample(
            x_0=x_0,
            timestep_indices=timestep_index,
            alpha_bars=alpha_bars,
            generator=generator,
        )
        snapshots.append((f"t={timestep_index}", x_t))
        print_distribution_statistics(f"x_t at t={timestep_index}", x_t)

    print(f"\n前向扩散图：{save_snapshots(snapshots, mode_ids)}")


if __name__ == "__main__":
    main()
