"""G4：加载训练后的噪声预测器，从标准高斯噪声反向生成二维双峰样本。"""

import os
from pathlib import Path

MATPLOTLIB_CACHE_DIR = Path("/tmp/deep_learning_foundations_matplotlib")
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch import nn

from bimodal_data import generate_bimodal_data
from forward_diffusion import build_noise_schedule, q_sample
from noise_prediction import NoisePredictor


SEED = 0
NUM_SAMPLES = 4096
SNAPSHOT_INDICES = (999, 750, 500, 250, 0)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CHECKPOINT_PATH = OUTPUT_DIR / "noise_predictor_g3b.pt"


def extract_coefficient(
    values: torch.Tensor,
    timesteps: torch.Tensor,
    sample_shape: torch.Size,
) -> torch.Tensor:
    """取出 batch 中各时间步的系数，并扩展为可与样本广播的形状。"""
    if timesteps.ndim != 1 or timesteps.shape[0] != sample_shape[0]:
        raise ValueError("timesteps 必须具有形状 [batch_size]")
    coefficient_shape = (sample_shape[0],) + (1,) * (len(sample_shape) - 1)
    return values.to(timesteps.device)[timesteps].reshape(coefficient_shape)


def predict_x_0_from_noise(
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    epsilon_prediction: torch.Tensor,
    alpha_bars: torch.Tensor,
) -> torch.Tensor:
    """由 x_t 和预测噪声估计干净样本 x_0。"""
    alpha_bar_t = extract_coefficient(alpha_bars, timesteps, x_t.shape)
    return (
        x_t - torch.sqrt(1.0 - alpha_bar_t) * epsilon_prediction
    ) / torch.sqrt(alpha_bar_t)


def reverse_mean_from_noise(
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    epsilon_prediction: torch.Tensor,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
) -> torch.Tensor:
    """使用噪声预测参数化计算 p_theta(x_{t-1} | x_t) 的均值。"""
    beta_t = extract_coefficient(betas, timesteps, x_t.shape)
    alpha_t = extract_coefficient(alphas, timesteps, x_t.shape)
    alpha_bar_t = extract_coefficient(alpha_bars, timesteps, x_t.shape)
    return (
        x_t
        - beta_t / torch.sqrt(1.0 - alpha_bar_t) * epsilon_prediction
    ) / torch.sqrt(alpha_t)


def posterior_variance(
    timesteps: torch.Tensor,
    sample_shape: torch.Size,
    betas: torch.Tensor,
    alpha_bars: torch.Tensor,
) -> torch.Tensor:
    """计算真实前向后验的方差 beta_tilde；第 0 步的方差为 0。"""
    alpha_bars_previous = torch.cat(
        (torch.ones(1, dtype=alpha_bars.dtype), alpha_bars[:-1])
    )
    beta_t = extract_coefficient(betas, timesteps, sample_shape)
    alpha_bar_t = extract_coefficient(alpha_bars, timesteps, sample_shape)
    alpha_bar_previous_t = extract_coefficient(
        alpha_bars_previous,
        timesteps,
        sample_shape,
    )
    return beta_t * (1.0 - alpha_bar_previous_t) / (1.0 - alpha_bar_t)


@torch.no_grad()
def reverse_step(
    model: nn.Module,
    x_t: torch.Tensor,
    timestep_index: int,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """执行一次 x_t -> x_{t-1}；最后一步只取均值，不再加入噪声。"""
    timesteps = torch.full(
        (x_t.shape[0],),
        timestep_index,
        dtype=torch.long,
        device=x_t.device,
    )
    epsilon_prediction = model(x_t, timesteps)
    mean = reverse_mean_from_noise(
        x_t=x_t,
        timesteps=timesteps,
        epsilon_prediction=epsilon_prediction,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
    )

    if timestep_index == 0:
        return mean

    variance = posterior_variance(
        timesteps=timesteps,
        sample_shape=x_t.shape,
        betas=betas,
        alpha_bars=alpha_bars,
    )
    noise = torch.randn(
        x_t.shape,
        dtype=x_t.dtype,
        device=x_t.device,
        generator=generator,
    )
    return mean + torch.sqrt(variance) * noise


@torch.no_grad()
def sample_reverse_process(
    model: nn.Module,
    num_samples: int,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
    generator: torch.Generator,
) -> list[tuple[int, torch.Tensor]]:
    """从标准高斯噪声开始，串行执行完整反向过程并保存关键快照。"""
    total_steps = betas.shape[0]
    requested_indices = set(SNAPSHOT_INDICES)
    if max(requested_indices) >= total_steps:
        raise ValueError("快照时间步不能超过噪声 schedule 的范围")

    x_t = torch.randn((num_samples, 2), generator=generator)
    snapshots: dict[int, torch.Tensor] = {
        total_steps - 1: x_t.detach().cpu().clone()
    }

    model.eval()
    for timestep_index in range(total_steps - 1, -1, -1):
        x_t = reverse_step(
            model=model,
            x_t=x_t,
            timestep_index=timestep_index,
            betas=betas,
            alphas=alphas,
            alpha_bars=alpha_bars,
            generator=generator,
        )
        if not torch.isfinite(x_t).all():
            raise RuntimeError(
                f"反向采样在时间步 {timestep_index} 出现 NaN 或 Inf"
            )
        if x_t.shape != (num_samples, 2):
            raise RuntimeError(
                f"反向采样形状发生变化：实际为 {list(x_t.shape)}"
            )

        produced_index = timestep_index - 1
        if produced_index in requested_indices:
            snapshots[produced_index] = x_t.detach().cpu().clone()
        if timestep_index == 0:
            # 此时得到的是最终干净样本，用 t=0 作为可视化标签。
            snapshots[0] = x_t.detach().cpu().clone()

    return [(index, snapshots[index]) for index in SNAPSHOT_INDICES]


def oracle_check(
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
) -> None:
    """用已知真实噪声检查 x_0 恢复公式和反向均值公式。"""
    generator = torch.Generator().manual_seed(SEED + 1)
    x_0, _ = generate_bimodal_data(
        num_samples=128,
        std=0.3,
        generator=generator,
    )
    timesteps = torch.randint(
        low=0,
        high=alpha_bars.shape[0],
        size=(x_0.shape[0],),
        generator=generator,
    )
    epsilon = torch.randn(x_0.shape, generator=generator)
    x_t, _ = q_sample(
        x_0=x_0,
        timestep_indices=timesteps,
        alpha_bars=alpha_bars,
        epsilon=epsilon,
    )

    recovered_x_0 = predict_x_0_from_noise(
        x_t=x_t,
        timesteps=timesteps,
        epsilon_prediction=epsilon,
        alpha_bars=alpha_bars,
    )
    recovery_error = torch.max(torch.abs(recovered_x_0 - x_0)).item()

    mean_from_noise = reverse_mean_from_noise(
        x_t=x_t,
        timesteps=timesteps,
        epsilon_prediction=epsilon,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
    )
    alpha_bars_previous = torch.cat(
        (torch.ones(1, dtype=alpha_bars.dtype), alpha_bars[:-1])
    )
    beta_t = extract_coefficient(betas, timesteps, x_t.shape)
    alpha_t = extract_coefficient(alphas, timesteps, x_t.shape)
    alpha_bar_t = extract_coefficient(alpha_bars, timesteps, x_t.shape)
    alpha_bar_previous_t = extract_coefficient(
        alpha_bars_previous,
        timesteps,
        x_t.shape,
    )
    mean_from_x_0 = (
        torch.sqrt(alpha_bar_previous_t) * beta_t / (1.0 - alpha_bar_t) * x_0
        + torch.sqrt(alpha_t)
        * (1.0 - alpha_bar_previous_t)
        / (1.0 - alpha_bar_t)
        * x_t
    )
    mean_error = torch.max(torch.abs(mean_from_noise - mean_from_x_0)).item()

    if recovery_error >= 1e-4 or mean_error >= 1e-4:
        raise RuntimeError(
            "oracle 检查失败："
            f"x_0 最大误差={recovery_error:.8f}，"
            f"均值最大误差={mean_error:.8f}"
        )
    print(f"oracle x_0 恢复最大误差={recovery_error:.8f}")
    print(f"oracle 反向均值最大误差={mean_error:.8f}")


def save_reverse_snapshots(snapshots: list[tuple[int, torch.Tensor]]) -> Path:
    """保存从标准高斯噪声逐渐形成双峰分布的过程。"""
    figure, axes = plt.subplots(
        1,
        len(snapshots),
        figsize=(20, 4),
        sharex=True,
        sharey=True,
    )
    for axis, (timestep_index, samples) in zip(axes, snapshots):
        axis.scatter(
            samples[:, 0].numpy(),
            samples[:, 1].numpy(),
            s=4,
            alpha=0.3,
            edgecolors="none",
        )
        title = "final x_0" if timestep_index == 0 else f"t={timestep_index}"
        axis.set(
            title=title,
            xlabel="x",
            xlim=(-6.0, 6.0),
            ylim=(-4.0, 4.0),
        )
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)

    axes[0].set_ylabel("y")
    figure.suptitle("DDPM Reverse Sampling on a 2D Bimodal Distribution")
    figure.tight_layout()
    output_path = OUTPUT_DIR / "reverse_sampling.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def load_model_and_schedule(
    checkpoint_path: Path,
) -> tuple[NoisePredictor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """从 G3-B checkpoint 恢复模型结构、权重和训练时的 schedule。"""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"找不到 checkpoint：{checkpoint_path}，请先运行 noise_prediction_train.py"
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model = NoisePredictor(
        time_embedding_dim=checkpoint["time_embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    betas, alphas, alpha_bars = build_noise_schedule(
        total_steps=checkpoint["total_steps"],
        beta_start=checkpoint["beta_start"],
        beta_end=checkpoint["beta_end"],
    )
    return model, betas, alphas, alpha_bars


def print_final_statistics(samples: torch.Tensor) -> None:
    """打印最终样本的整体统计量和按 x 正负粗分的两个峰。"""
    left = samples[samples[:, 0] < 0]
    right = samples[samples[:, 0] >= 0]
    print(f"最终样本 shape={list(samples.shape)}")
    print(f"最终样本均值={samples.mean(dim=0).tolist()}")
    print(f"左侧比例={left.shape[0] / samples.shape[0]:.4f}")
    print(f"右侧比例={right.shape[0] / samples.shape[0]:.4f}")
    if left.shape[0] > 0:
        print(f"左侧经验均值={left.mean(dim=0).tolist()}")
    if right.shape[0] > 0:
        print(f"右侧经验均值={right.mean(dim=0).tolist()}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)

    model, betas, alphas, alpha_bars = load_model_and_schedule(
        CHECKPOINT_PATH
    )
    oracle_check(betas, alphas, alpha_bars)
    snapshots = sample_reverse_process(
        model=model,
        num_samples=NUM_SAMPLES,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        generator=generator,
    )
    final_samples = snapshots[-1][1]
    print_final_statistics(final_samples)
    print(f"反向采样图：{save_reverse_snapshots(snapshots)}")


if __name__ == "__main__":
    main()
