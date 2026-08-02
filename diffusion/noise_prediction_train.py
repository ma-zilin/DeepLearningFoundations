"""G3-B：在随机 batch 上正式训练二维时间条件噪声预测网络。"""

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
from noise_prediction import NoisePredictor, check_gradients


SEED = 0
NUM_SAMPLES = 4096
VALIDATION_FRACTION = 0.2
BATCH_SIZE = 128
DATA_STD = 0.3
T = 1000
BETA_START = 1e-4
BETA_END = 2e-2
TIME_EMBEDDING_DIM = 32
HIDDEN_DIM = 128
LEARNING_RATE = 3e-4
TRAINING_STEPS = 5000
VALIDATION_INTERVAL = 250
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


@torch.no_grad()
def evaluate(
    model: nn.Module,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    epsilon: torch.Tensor,
    loss_function: nn.Module,
) -> float:
    """在固定验证数据上计算噪声预测 MSE。"""
    was_training = model.training
    model.eval()
    epsilon_prediction = model(x_t, timesteps)
    loss = loss_function(epsilon_prediction, epsilon)
    if was_training:
        model.train()
    return loss.item()


def save_loss_curve(
    training_losses: list[float],
    validation_steps: list[int],
    validation_losses: list[float],
    initial_loss: float,
    zero_baseline_loss: float,
) -> Path:
    """保存正式训练和固定验证集的 loss 曲线。"""
    figure, axis = plt.subplots(figsize=(8, 4.8))
    training_steps = range(1, len(training_losses) + 1)
    axis.plot(
        training_steps,
        training_losses,
        linewidth=0.8,
        alpha=0.45,
        label="Training batch",
    )
    axis.plot(
        validation_steps,
        validation_losses,
        marker="o",
        markersize=3,
        linewidth=1.8,
        label="Fixed validation",
    )
    axis.axhline(
        initial_loss,
        color="tab:orange",
        linestyle="--",
        linewidth=1.2,
        label="Untrained model",
    )
    axis.axhline(
        zero_baseline_loss,
        color="tab:red",
        linestyle=":",
        linewidth=1.4,
        label="Zero predictor",
    )
    axis.set(
        title="Random-batch Noise Prediction",
        xlabel="Optimization step",
        ylabel="Noise prediction MSE",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()

    output_path = OUTPUT_DIR / "noise_prediction_train_val_loss.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 二维小 MLP 的矩阵很小，单线程可避免多线程调度开销。
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)

    data, _ = generate_bimodal_data(
        num_samples=NUM_SAMPLES,
        std=DATA_STD,
        generator=generator,
    )
    permutation = torch.randperm(NUM_SAMPLES, generator=generator)
    validation_size = int(NUM_SAMPLES * VALIDATION_FRACTION)
    validation_indices = permutation[:validation_size]
    training_indices = permutation[validation_size:]
    validation_x_0 = data[validation_indices]
    training_x_0 = data[training_indices]

    _, _, alpha_bars = build_noise_schedule(
        total_steps=T,
        beta_start=BETA_START,
        beta_end=BETA_END,
    )

    # 验证条件固定，保证不同训练阶段的 loss 可以直接比较。
    validation_timesteps = torch.randint(
        low=0,
        high=T,
        size=(validation_size,),
        generator=generator,
    )
    validation_epsilon = torch.randn(
        validation_x_0.shape,
        generator=generator,
    )
    validation_x_t, _ = q_sample(
        x_0=validation_x_0,
        timestep_indices=validation_timesteps,
        alpha_bars=alpha_bars,
        epsilon=validation_epsilon,
    )

    model = NoisePredictor(
        time_embedding_dim=TIME_EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.MSELoss()

    initial_loss = evaluate(
        model,
        validation_x_t,
        validation_timesteps,
        validation_epsilon,
        loss_function,
    )
    zero_baseline_loss = loss_function(
        torch.zeros_like(validation_epsilon),
        validation_epsilon,
    ).item()

    print(f"训练集大小={training_x_0.shape[0]}")
    print(f"验证集大小={validation_x_0.shape[0]}")
    print(f"固定验证 x_t.shape={list(validation_x_t.shape)}")
    print(f"恒零预测器验证 loss={zero_baseline_loss:.6f}")
    print(f"未训练模型验证 loss={initial_loss:.6f}")

    training_losses: list[float] = []
    validation_steps = [0]
    validation_losses = [initial_loss]

    model.train()
    for step in range(1, TRAINING_STEPS + 1):
        batch_indices = torch.randint(
            low=0,
            high=training_x_0.shape[0],
            size=(BATCH_SIZE,),
            generator=generator,
        )
        x_0 = training_x_0[batch_indices]
        timesteps = torch.randint(
            low=0,
            high=T,
            size=(BATCH_SIZE,),
            generator=generator,
        )
        epsilon = torch.randn(x_0.shape, generator=generator)
        x_t, _ = q_sample(
            x_0=x_0,
            timestep_indices=timesteps,
            alpha_bars=alpha_bars,
            epsilon=epsilon,
        )

        optimizer.zero_grad()
        epsilon_prediction = model(x_t, timesteps)
        loss = loss_function(epsilon_prediction, epsilon)
        if not torch.isfinite(loss):
            raise RuntimeError(f"第 {step} 步出现 NaN 或 Inf loss")

        loss.backward()
        if step == 1:
            check_gradients(model)
        optimizer.step()
        training_losses.append(loss.item())

        if step == 1 or step % VALIDATION_INTERVAL == 0:
            validation_loss = evaluate(
                model,
                validation_x_t,
                validation_timesteps,
                validation_epsilon,
                loss_function,
            )
            if not torch.isfinite(torch.tensor(validation_loss)):
                raise RuntimeError(
                    f"第 {step} 步出现 NaN 或 Inf 验证 loss"
                )
            validation_steps.append(step)
            validation_losses.append(validation_loss)
            print(
                f"step={step:4d}, train_loss={loss.item():.6f}, "
                f"val_loss={validation_loss:.6f}"
            )

    final_validation_loss = validation_losses[-1]
    baseline_loss = min(initial_loss, zero_baseline_loss)
    relative_improvement = 1.0 - final_validation_loss / baseline_loss
    if final_validation_loss >= baseline_loss:
        raise RuntimeError(
            "训练后验证 loss 没有优于未训练模型和恒零预测器"
        )

    curve_path = save_loss_curve(
        training_losses=training_losses,
        validation_steps=validation_steps,
        validation_losses=validation_losses,
        initial_loss=initial_loss,
        zero_baseline_loss=zero_baseline_loss,
    )
    checkpoint_path = OUTPUT_DIR / "noise_predictor_g3b.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "time_embedding_dim": TIME_EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "total_steps": T,
            "beta_start": BETA_START,
            "beta_end": BETA_END,
            "data_std": DATA_STD,
            "seed": SEED,
        },
        checkpoint_path,
    )

    print(f"最终验证 loss={final_validation_loss:.6f}")
    print(f"相对最佳基线改善={relative_improvement:.2%}")
    print(f"训练曲线：{curve_path}")
    print(f"模型权重：{checkpoint_path}")


if __name__ == "__main__":
    main()
