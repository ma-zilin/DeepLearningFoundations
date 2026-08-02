"""G3-A：时间条件噪声预测网络与单批次过拟合测试。"""

import math
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


SEED = 0
BATCH_SIZE = 128
DATA_STD = 0.3
T = 1000
TIME_EMBEDDING_DIM = 32
HIDDEN_DIM = 128
LEARNING_RATE = 3e-4
TRAINING_STEPS = 3000
LOG_INTERVAL = 300
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


class SinusoidalTimeEmbedding(nn.Module):
    """将离散时间步映射为包含多种频率的连续向量。"""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim < 4 or embedding_dim % 2 != 0:
            raise ValueError("embedding_dim 必须是大于等于 4 的偶数")

        half_dim = embedding_dim // 2
        frequencies = torch.exp(
            -math.log(10_000)
            * torch.arange(half_dim, dtype=torch.float32)
            / (half_dim - 1)
        )
        self.register_buffer("frequencies", frequencies)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        angles = timesteps.float().unsqueeze(1) * self.frequencies.unsqueeze(0)
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)


class NoisePredictor(nn.Module):
    """输入二维带噪点和时间步，输出二维噪声预测。"""

    def __init__(
        self,
        time_embedding_dim: int = 32,
        hidden_dim: int = 128,
        use_time_condition: bool = True,
    ) -> None:
        super().__init__()
        self.use_time_condition = use_time_condition
        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(2 + time_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        time_features = self.time_embedding(timesteps)
        if not self.use_time_condition:
            time_features = torch.zeros_like(time_features)
        model_input = torch.cat((x_t, time_features), dim=1)
        return self.network(model_input)


def check_gradients(model: nn.Module) -> None:
    """确认所有参数得到有限梯度，且整体至少存在非零梯度。"""
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if any(gradient is None for gradient in gradients):
        raise RuntimeError("至少有一个可训练参数没有梯度")
    if not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("检测到 NaN 或 Inf 梯度")
    nonzero_count = sum(
        int(torch.count_nonzero(gradient).item())
        for gradient in gradients
    )
    if nonzero_count == 0:
        raise RuntimeError("所有参数的梯度均为 0")
    total_norm = torch.sqrt(
        sum(torch.sum(gradient**2) for gradient in gradients)
    )
    print(
        f"梯度检查通过：非零梯度元素={nonzero_count}，"
        f"总范数={total_norm.item():.6f}"
    )


def save_loss_curve(loss_history: list[float]) -> Path:
    """保存固定批次的过拟合曲线。"""
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(loss_history, linewidth=1.5)
    axis.set(
        title="Single-batch Overfitting",
        xlabel="Optimization step",
        ylabel="Noise prediction MSE",
        yscale="log",
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output_path = OUTPUT_DIR / "single_batch_overfit_loss.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    x_0, _ = generate_bimodal_data(
        num_samples=BATCH_SIZE,
        std=DATA_STD,
        generator=generator,
    )
    _, _, alpha_bars = build_noise_schedule(total_steps=T)

    timesteps = torch.randint(0, T, (BATCH_SIZE,), generator=generator)
    epsilon = torch.randn(x_0.shape, generator=generator)
    x_t, _ = q_sample(
        x_0=x_0,
        timestep_indices=timesteps,
        alpha_bars=alpha_bars,
        epsilon=epsilon,
    )

    model = NoisePredictor(TIME_EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.MSELoss()
    with torch.no_grad():
        initial_prediction = model(x_t, timesteps)
        initial_loss = loss_function(initial_prediction, epsilon).item()
        zero_baseline_loss = loss_function(
            torch.zeros_like(epsilon), epsilon
        ).item()

    assert initial_prediction.shape == epsilon.shape == (BATCH_SIZE, 2)
    print(f"x_0.shape={list(x_0.shape)}")
    print(f"timesteps.shape={list(timesteps.shape)}")
    print(f"x_t.shape={list(x_t.shape)}")
    print(f"epsilon.shape={list(epsilon.shape)}")
    print(f"epsilon_pred.shape={list(initial_prediction.shape)}")
    print(f"恒零预测器 loss={zero_baseline_loss:.6f}")
    print(f"未训练模型 loss={initial_loss:.6f}")

    loss_history: list[float] = []
    for step in range(1, TRAINING_STEPS + 1):
        optimizer.zero_grad()
        epsilon_prediction = model(x_t, timesteps)
        loss = loss_function(epsilon_prediction, epsilon)
        if not torch.isfinite(loss):
            raise RuntimeError(f"第 {step} 步出现 NaN 或 Inf loss")
        loss.backward()
        if step == 1:
            check_gradients(model)
        optimizer.step()

        loss_history.append(loss.item())
        if step == 1 or step % LOG_INTERVAL == 0:
            print(f"step={step:4d}, loss={loss.item():.8f}")

    final_loss = loss_history[-1]
    if final_loss >= 1e-3:
        raise RuntimeError(
            f"固定批次最终 loss={final_loss:.6f}，未达到 1e-3 验收线"
        )
    print(f"最终 loss={final_loss:.8f}")
    print(f"过拟合曲线：{save_loss_curve(loss_history)}")


if __name__ == "__main__":
    main()
