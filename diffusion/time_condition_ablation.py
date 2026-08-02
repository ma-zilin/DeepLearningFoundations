"""G5：保持其他条件不变，对比有无时间条件的噪声预测与生成结果。"""

import copy
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
from torch import nn

from bimodal_data import CENTERS, generate_bimodal_data
from distribution_evaluation import (
    DistributionMetrics,
    evaluate_generated_distribution,
)
from forward_diffusion import build_noise_schedule, q_sample
from noise_prediction import NoisePredictor, check_gradients
from noise_prediction_train import (
    BATCH_SIZE,
    BETA_END,
    BETA_START,
    DATA_STD,
    HIDDEN_DIM,
    LEARNING_RATE,
    NUM_SAMPLES,
    SEED,
    T,
    TIME_EMBEDDING_DIM,
    TRAINING_STEPS,
    VALIDATION_FRACTION,
    VALIDATION_INTERVAL,
    evaluate,
)
from reverse_sampling import sample_reverse_process


TRAINING_RANDOM_SEED = 100
SAMPLING_SEED = 0
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


@dataclass
class TrainingResult:
    """保存一个消融变体的模型和 loss 轨迹。"""

    name: str
    model: NoisePredictor
    training_losses: list[float]
    validation_steps: list[int]
    validation_losses: list[float]


def prepare_data(
    alpha_bars: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造一次固定的数据划分和验证条件，供两个变体共同使用。"""
    generator = torch.Generator().manual_seed(SEED)
    data, _ = generate_bimodal_data(
        num_samples=NUM_SAMPLES,
        std=DATA_STD,
        generator=generator,
    )
    permutation = torch.randperm(NUM_SAMPLES, generator=generator)
    validation_size = int(NUM_SAMPLES * VALIDATION_FRACTION)
    validation_x_0 = data[permutation[:validation_size]]
    training_x_0 = data[permutation[validation_size:]]
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
    return (
        training_x_0,
        validation_x_t,
        validation_timesteps,
        validation_epsilon,
    )


def train_variant(
    name: str,
    model: NoisePredictor,
    training_x_0: torch.Tensor,
    validation_x_t: torch.Tensor,
    validation_timesteps: torch.Tensor,
    validation_epsilon: torch.Tensor,
    alpha_bars: torch.Tensor,
) -> TrainingResult:
    """使用相同随机序列训练一个模型变体。"""
    generator = torch.Generator().manual_seed(TRAINING_RANDOM_SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.MSELoss()
    initial_validation_loss = evaluate(
        model,
        validation_x_t,
        validation_timesteps,
        validation_epsilon,
        loss_function,
    )
    training_losses: list[float] = []
    validation_steps = [0]
    validation_losses = [initial_validation_loss]

    print(f"\n=== training: {name} ===")
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
            raise RuntimeError(f"{name} 在第 {step} 步出现 NaN 或 Inf loss")
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
                    f"{name} 在第 {step} 步出现 NaN 或 Inf 验证 loss"
                )
            validation_steps.append(step)
            validation_losses.append(validation_loss)
            print(
                f"step={step:4d}, train_loss={loss.item():.6f}, "
                f"val_loss={validation_loss:.6f}"
            )

    return TrainingResult(
        name=name,
        model=model,
        training_losses=training_losses,
        validation_steps=validation_steps,
        validation_losses=validation_losses,
    )


def save_loss_comparison(results: list[TrainingResult]) -> Path:
    """保存基线与无时间条件模型的训练和验证 loss。"""
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for result in results:
        axes[0].plot(
            range(1, len(result.training_losses) + 1),
            result.training_losses,
            linewidth=0.7,
            alpha=0.55,
            label=result.name,
        )
        axes[1].plot(
            result.validation_steps,
            result.validation_losses,
            marker="o",
            markersize=3,
            linewidth=1.6,
            label=result.name,
        )
    axes[0].set(
        title="Training batches",
        xlabel="Optimization step",
        ylabel="Noise prediction MSE",
    )
    axes[1].set(
        title="Fixed validation set",
        xlabel="Optimization step",
        ylabel="Noise prediction MSE",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Time-condition Ablation")
    figure.tight_layout()
    output_path = OUTPUT_DIR / "time_condition_ablation_loss.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def generate_final_samples(
    result: TrainingResult,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
) -> torch.Tensor:
    """使用固定采样 seed 生成一个变体的最终样本。"""
    generator = torch.Generator().manual_seed(SAMPLING_SEED)
    snapshots = sample_reverse_process(
        model=result.model,
        num_samples=NUM_SAMPLES,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        generator=generator,
    )
    return snapshots[-1][1]


def save_generation_comparison(
    generated_results: list[tuple[TrainingResult, torch.Tensor]],
) -> Path:
    """并排保存有无时间条件的最终生成分布。"""
    figure, axes = plt.subplots(
        1,
        len(generated_results),
        figsize=(10, 4.5),
        sharex=True,
        sharey=True,
    )
    for axis, (result, samples) in zip(axes, generated_results):
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
            title=result.name,
            xlabel="x",
            xlim=(-4.0, 4.0),
            ylim=(-2.0, 2.0),
        )
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("y")
    figure.suptitle("Generated Distribution With and Without Time Condition")
    figure.tight_layout()
    output_path = OUTPUT_DIR / "time_condition_ablation_samples.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def save_metrics_csv(
    results_and_metrics: list[tuple[TrainingResult, DistributionMetrics]],
) -> Path:
    """保存两个变体的 loss 和核心生成指标。"""
    output_path = OUTPUT_DIR / "time_condition_ablation_metrics.csv"
    fieldnames = [
        "variant",
        "uses_time_condition",
        "final_training_loss",
        "final_validation_loss",
        "left_ratio",
        "right_ratio",
        "left_center_error",
        "right_center_error",
        "y_mean",
        "mean_within_std",
        "distribution_passed",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result, metrics in results_and_metrics:
            writer.writerow(
                {
                    "variant": result.name,
                    "uses_time_condition": result.model.use_time_condition,
                    "final_training_loss": result.training_losses[-1],
                    "final_validation_loss": result.validation_losses[-1],
                    "left_ratio": metrics.left_ratio,
                    "right_ratio": metrics.right_ratio,
                    "left_center_error": metrics.left_center_error,
                    "right_center_error": metrics.right_center_error,
                    "y_mean": metrics.y_mean,
                    "mean_within_std": metrics.mean_within_std,
                    "distribution_passed": metrics.passed,
                }
            )
    return output_path


def save_checkpoint(result: TrainingResult) -> Path:
    """保存消融变体，便于在新进程中复现采样结果。"""
    suffix = "with_time" if result.model.use_time_condition else "without_time"
    output_path = OUTPUT_DIR / f"noise_predictor_ablation_{suffix}.pt"
    torch.save(
        {
            "model_state_dict": result.model.state_dict(),
            "time_embedding_dim": TIME_EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "use_time_condition": result.model.use_time_condition,
            "total_steps": T,
            "beta_start": BETA_START,
            "beta_end": BETA_END,
            "data_std": DATA_STD,
            "seed": SEED,
        },
        output_path,
    )
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    betas, alphas, alpha_bars = build_noise_schedule(
        total_steps=T,
        beta_start=BETA_START,
        beta_end=BETA_END,
    )
    (
        training_x_0,
        validation_x_t,
        validation_timesteps,
        validation_epsilon,
    ) = prepare_data(alpha_bars)

    initial_model = NoisePredictor(
        time_embedding_dim=TIME_EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
    )
    baseline_model = copy.deepcopy(initial_model)
    no_time_model = copy.deepcopy(initial_model)
    no_time_model.use_time_condition = False
    baseline_parameter_count = sum(
        parameter.numel() for parameter in baseline_model.parameters()
    )
    no_time_parameter_count = sum(
        parameter.numel() for parameter in no_time_model.parameters()
    )
    if baseline_parameter_count != no_time_parameter_count:
        raise RuntimeError("两个变体的参数量不同，不满足单变量消融要求")
    print(f"两个变体参数量均为 {baseline_parameter_count}")

    results = [
        train_variant(
            name="With time condition",
            model=baseline_model,
            training_x_0=training_x_0,
            validation_x_t=validation_x_t,
            validation_timesteps=validation_timesteps,
            validation_epsilon=validation_epsilon,
            alpha_bars=alpha_bars,
        ),
        train_variant(
            name="Without time condition",
            model=no_time_model,
            training_x_0=training_x_0,
            validation_x_t=validation_x_t,
            validation_timesteps=validation_timesteps,
            validation_epsilon=validation_epsilon,
            alpha_bars=alpha_bars,
        ),
    ]

    results_and_metrics: list[tuple[TrainingResult, DistributionMetrics]] = []
    generated_results: list[tuple[TrainingResult, torch.Tensor]] = []
    for result in results:
        final_samples = generate_final_samples(
            result=result,
            betas=betas,
            alphas=alphas,
            alpha_bars=alpha_bars,
        )
        metrics = evaluate_generated_distribution(
            samples=final_samples,
            sampling_seed=SAMPLING_SEED,
        )
        results_and_metrics.append((result, metrics))
        generated_results.append((result, final_samples))
        print(f"\n=== result: {result.name} ===")
        print(f"final train loss={result.training_losses[-1]:.6f}")
        print(f"final val loss={result.validation_losses[-1]:.6f}")
        print(
            f"center errors=({metrics.left_center_error:.4f}, "
            f"{metrics.right_center_error:.4f})"
        )
        print(
            f"mode ratio=({metrics.left_ratio:.4f}, "
            f"{metrics.right_ratio:.4f})"
        )
        print(f"mean within-mode std={metrics.mean_within_std:.4f}")
        print(f"distribution passed={metrics.passed}")

    loss_path = save_loss_comparison(results)
    samples_path = save_generation_comparison(generated_results)
    metrics_path = save_metrics_csv(results_and_metrics)
    checkpoint_paths = [save_checkpoint(result) for result in results]
    print(f"\nloss 对比图：{loss_path}")
    print(f"生成分布对比图：{samples_path}")
    print(f"消融指标：{metrics_path}")
    for checkpoint_path in checkpoint_paths:
        print(f"模型权重：{checkpoint_path}")


if __name__ == "__main__":
    main()
