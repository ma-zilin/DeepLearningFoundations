"""DDPM 线性噪声 schedule 与前向闭式采样。"""

import torch


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
    """使用闭式公式一步采样 q(x_t | x_0)。

    timestep_indices 可以是单个从 0 开始的下标，也可以是形状为 [B] 的
    batch 下标。后者允许 batch 中每个样本使用不同噪声级别。
    """
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
