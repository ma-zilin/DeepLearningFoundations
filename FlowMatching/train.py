"""F2：训练条件速度网络，重采样训练路径并使用固定验证监督。

仓库根目录运行：python -m FlowMatching.train
默认在 CPU 上运行 2000 次更新，不包含积分采样或生成质量评测。
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .conditioning import build_velocity_input
from .model import VelocityMLP
from .path import linear_path, sample_path_batch


STEPS = 2000
BATCH_SIZE = 256
EVAL_EVERY = 100
LEARNING_RATE = 1e-3
MODEL_SEED = 0
INDEX_SEED = 4
PATH_SEED = 5


def velocity_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """每个样本的两个分量平方误差相加，再对样本取平均。"""
    if prediction.ndim != 2 or prediction.shape[1] != 2 or prediction.shape != target.shape:
        raise ValueError("预测和标签必须具有相同的 [B, 2] 形状")
    return (prediction - target).square().sum(dim=1).mean()


def train_step(model, optimizer, x1, c, path_generator) -> dict[str, float]:
    """对选出的目标点重新采样路径，完成一次参数更新。"""
    model.train()
    _, t, xt, target = sample_path_batch(x1, path_generator)
    network_input = build_velocity_input(xt, t, c)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(network_input)
    loss = velocity_mse(prediction, target)
    if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
        raise FloatingPointError("训练预测或 loss 出现 NaN/Inf")
    loss.backward()
    squared_norm = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"{name} 的梯度缺失或含 NaN/Inf")
        norm = parameter.grad.double().norm().item()
        squared_norm += norm * norm
    optimizer.step()
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise FloatingPointError("更新后的网络参数出现 NaN/Inf")
    return {"loss": loss.item(), "grad_l2": math.sqrt(squared_norm)}


def evaluate(model, network_input, target) -> float:
    """在已固定的验证输入、标签上计算 loss，不重新采样，也不更新参数。"""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            prediction = model(network_input)
            loss = velocity_mse(prediction, target)
            if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
                raise FloatingPointError("验证预测或 loss 出现 NaN/Inf")
            return loss.item()
    finally:
        model.train(was_training)


def validation_digest(network_input, target) -> str:
    digest = hashlib.sha256()
    for tensor in (network_input, target):
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path, model, optimizer, index_rng, path_rng, step, validation_loss, config):
    """保存参数及优化器、随机状态；后续恢复不必丢弃 Adam 历史。"""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "index_rng_state": index_rng.get_state(),
        "path_rng_state": path_rng.get_state(),
        "torch_rng_state": torch.random.get_rng_state(),
        "step": step,
        "validation_loss": validation_loss,
        "config": config,
    }, path)


def write_csv(path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=project_dir / "artifacts" / "f2_training",
    )
    args = parser.parse_args()
    data_dir = project_dir / "artifacts" / "f1_data"
    for name in ("train.pt", "validation.pt", "config.json"):
        if not (data_dir / name).is_file():
            parser.error("缺少 F1 数据，请先运行 python -m FlowMatching.prepare_data")

    # 训练只使用目标池；每步的 x0、t、xt 和速度标签由 train_step 重建。
    train = torch.load(data_dir / "train.pt", map_location="cpu", weights_only=True)
    validation = torch.load(data_dir / "validation.pt", map_location="cpu", weights_only=True)
    x1_pool, c_pool = train["x1"], train["c"]
    val_input, val_target = validation["network_input"], validation["velocity"]
    if x1_pool.shape != (len(c_pool), 2) or len(c_pool) == 0:
        raise ValueError("训练目标池形状错误或为空")
    if not torch.isfinite(x1_pool).all():
        raise ValueError("训练目标池含 NaN/Inf")
    # 确认固定验证输入确实来自其端点和时间，不引入新随机量。
    val_xt, expected_target = linear_path(validation["x0"], validation["x1"], validation["t"])
    torch.testing.assert_close(val_target, expected_target, rtol=0, atol=0)
    torch.testing.assert_close(
        val_input, build_velocity_input(val_xt, validation["t"], validation["c"]), rtol=0, atol=0,
    )
    fixed_val_digest = validation_digest(val_input, val_target)

    torch.set_num_threads(1)
    torch.manual_seed(MODEL_SEED)
    model = VelocityMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    index_rng = torch.Generator(device="cpu").manual_seed(INDEX_SEED)
    path_rng = torch.Generator(device="cpu").manual_seed(PATH_SEED)
    config = {
        "description": "Resampled-path velocity training with fixed validation supervision",
        "model": "VelocityMLP: 5 -> 64 -> 64 -> 2; ReLU hidden layers; linear output",
        "model_seed": MODEL_SEED,
        "index_seed": INDEX_SEED,
        "path_seed": PATH_SEED,
        "target_sampling": "Uniform indices with replacement from training x1 and corresponding c",
        "path_sampling": "New independent Gaussian x0 and uniform t on every update",
        "train_samples": len(x1_pool),
        "validation_samples": len(val_target),
        "data_seeds": json.loads((data_dir / "config.json").read_text())["splits"],
        "data_sha256": {
            name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
            for name in ("train.pt", "validation.pt")
        },
        "validation_sha256": fixed_val_digest,
        "batch_size": BATCH_SIZE,
        "updates": STEPS,
        "eval_every": EVAL_EVERY,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "optimizer_defaults": optimizer.defaults,
        "loss": "mean over samples of sum over two velocity components of squared error",
        "train_log": "Loss and gradient before each named update; validation after completed updates",
        "acceptance": "Final fixed-validation loss below both zero-velocity and untrained-network losses; finite numerics",
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "torch_version": str(torch.__version__),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    zero_loss = val_target.square().sum(dim=1).mean().item()
    if not math.isfinite(zero_loss):
        raise FloatingPointError("零速度基线 loss 非有限值")
    untrained_loss = evaluate(model, val_input, val_target)
    val_rows = [{"step": 0, "loss": untrained_loss}]
    train_rows = []
    best_loss, best_step = untrained_loss, 0
    save_checkpoint(args.output_dir / "best.pt", model, optimizer, index_rng, path_rng, 0, best_loss, config)
    print(f"固定验证基线：zero={zero_loss:.6f}, untrained={untrained_loss:.6f}")

    for step in range(1, STEPS + 1):
        indices = torch.randint(len(x1_pool), (BATCH_SIZE,), generator=index_rng, device="cpu")
        result = train_step(model, optimizer, x1_pool[indices], c_pool[indices], path_rng)
        train_rows.append({"step": step, **result})
        if step % EVAL_EVERY == 0:
            val_loss = evaluate(model, val_input, val_target)
            val_rows.append({"step": step, "loss": val_loss})
            if val_loss < best_loss:
                best_loss, best_step = val_loss, step
                save_checkpoint(
                    args.output_dir / "best.pt", model, optimizer, index_rng, path_rng, step, val_loss, config,
                )
            if step % 500 == 0:
                print(f"step={step}: train={result['loss']:.6f}, validation={val_loss:.6f}")

    if validation_digest(val_input, val_target) != fixed_val_digest:
        raise RuntimeError("固定验证数据在训练过程中被修改")
    final_loss = val_rows[-1]["loss"]
    save_checkpoint(
        args.output_dir / "last.pt", model, optimizer, index_rng, path_rng, STEPS, final_loss, config,
    )
    passed = final_loss < zero_loss and final_loss < untrained_loss
    metrics = {
        "passed": passed,
        "zero_velocity_validation_loss": zero_loss,
        "untrained_validation_loss": untrained_loss,
        "best_validation_loss": best_loss,
        "best_step": best_step,
        "final_validation_loss": final_loss,
        "completed_updates": STEPS,
        "all_predictions_losses_gradients_parameters_finite": True,
        "validation_unchanged": True,
        "gradient_l2": {
            "min": min(row["grad_l2"] for row in train_rows),
            "max": max(row["grad_l2"] for row in train_rows),
            "last": train_rows[-1]["grad_l2"],
        },
        "scope": "One training seed; velocity prediction on fixed validation data, not generation quality",
    }
    for filename, content in (("config.json", config), ("metrics.json", metrics)):
        (args.output_dir / filename).write_text(
            json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
    write_csv(args.output_dir / "train.csv", train_rows)
    write_csv(args.output_dir / "validation.csv", val_rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    axes[0].plot([row["step"] for row in train_rows], [row["loss"] for row in train_rows],
                 alpha=0.25, linewidth=0.7, label="Resampled batch loss")
    averaged = [sum(row["loss"] for row in train_rows[end-EVAL_EVERY:end]) / EVAL_EVERY
                for end in range(EVAL_EVERY, STEPS + 1, EVAL_EVERY)]
    axes[0].plot(range(EVAL_EVERY, STEPS + 1, EVAL_EVERY), averaged, label="Mean over preceding 100 batches")
    axes[0].set_title("Training: fresh paths on every update")
    axes[1].plot([row["step"] for row in val_rows], [row["loss"] for row in val_rows], label="Fixed validation")
    axes[1].axhline(zero_loss, linestyle="--", color="tab:red", label="Zero velocity baseline")
    axes[1].axhline(untrained_loss, linestyle=":", color="gray", label="Untrained baseline")
    axes[1].set_title("Validation: identical inputs and targets")
    for ax in axes:
        ax.set(xlabel="Parameter updates", ylabel="Velocity-vector squared error")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(args.output_dir / "loss.png", dpi=150)
    plt.close(fig)

    print(f"最佳验证 loss={best_loss:.6f}（step={best_step}），最终={final_loss:.6f}")
    print(f"正式训练与验证检查：{'通过' if passed else '未通过'}；结果：{args.output_dir}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
