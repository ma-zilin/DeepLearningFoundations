"""F2：固定 32 个训练样本，运行有界监督拟合诊断，默认 500 步。

仓库根目录运行：python -m FlowMatching.diagnose_overfit
已授权的预算对照：python -m FlowMatching.diagnose_overfit --steps 2000
本实验只检查固定监督能否被学习，不评估验证集或模型生成质量。
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
from .path import linear_path


BATCH_SIZE = 32
MODEL_SEED = 0
DEFAULT_STEPS = 500
MAX_STEPS = 2000
LEARNING_RATE = 1e-3
MAX_LOSS_RATIO = 0.01


def batch_digest(batch: dict[str, torch.Tensor]) -> str:
    """记录固定数据指纹，检查训练前后没有修改输入或标签。"""
    digest = hashlib.sha256()
    for name, tensor in sorted(batch.items()):
        digest.update(name.encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="参数更新次数，范围 1 到 2000")
    parser.add_argument(
        "--output-dir", type=Path,
    )
    args = parser.parse_args()
    if not 1 <= args.steps <= MAX_STEPS:
        parser.error("steps 必须在 1 到 2000 之间")
    if args.output_dir is None:
        folder_name = "f2_overfit" if args.steps == DEFAULT_STEPS else f"f2_overfit_{args.steps}"
        args.output_dir = project_dir / "artifacts" / folder_name
    data_file = project_dir / "artifacts" / "f1_data" / "train.pt"
    if not data_file.is_file():
        parser.error("缺少训练快照，请先运行 python -m FlowMatching.prepare_data")

    snapshot = torch.load(data_file, map_location="cpu", weights_only=True)
    if len(snapshot["x1"]) < BATCH_SIZE:
        parser.error("训练快照不足 32 个样本")

    # 端点、配对、时间和任务条件仅在循环外取一次，不重新采样。
    batch = {
        name: snapshot[name][:BATCH_SIZE].detach().clone()
        for name in ("x0", "x1", "t", "c", "mode_ids")
    }
    batch["xt"], batch["velocity"] = linear_path(batch["x0"], batch["x1"], batch["t"])
    batch["network_input"] = build_velocity_input(batch["xt"], batch["t"], batch["c"])
    for name in ("xt", "velocity", "network_input"):
        torch.testing.assert_close(batch[name], snapshot[name][:BATCH_SIZE], rtol=0, atol=0)
    fixed_digest = batch_digest(batch)

    torch.set_num_threads(1)
    torch.manual_seed(MODEL_SEED)
    model = VelocityMLP().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    inputs, targets = batch["network_input"], batch["velocity"]
    rows = []
    parameter_gradients = {name: [] for name, _ in model.named_parameters()}

    # step 表示已完成的更新次数：0 是未训练网络，args.steps 是最终网络。
    # 同行的 grad_l2 是该 loss 反传得到、用于下一次更新的梯度范数。
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs)
        loss = (prediction - targets).square().sum(dim=1).mean()
        if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
            raise FloatingPointError(f"step={step} 的预测或 loss 出现 NaN/Inf")
        row = {"step": step, "loss": loss.item(), "grad_l2": ""}
        if step == args.steps:
            rows.append(row)
            break

        loss.backward()
        squared_norm = 0.0
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"step={step}，{name} 的梯度缺失或含 NaN/Inf")
            norm = parameter.grad.double().norm().item()
            parameter_gradients[name].append(norm)
            squared_norm += norm * norm
        row["grad_l2"] = math.sqrt(squared_norm)
        rows.append(row)
        optimizer.step()
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise FloatingPointError(f"step={step + 1} 的网络参数出现 NaN/Inf")

    if batch_digest(batch) != fixed_digest:
        raise RuntimeError("固定 batch 在训练过程中被修改")
    initial_loss, final_loss = rows[0]["loss"], rows[-1]["loss"]
    loss_ratio = final_loss / initial_loss if initial_loss > 0 else None
    passed = loss_ratio is not None and loss_ratio <= MAX_LOSS_RATIO
    config = {
        "description": "Fixed-supervision diagnostic, not generalization or generation evaluation",
        "data_file": "FlowMatching/artifacts/f1_data/train.pt",
        "data_sha256": hashlib.sha256(data_file.read_bytes()).hexdigest(),
        "selection": "first 32 rows, indices 0 through 31",
        "fixed_batch_sha256": fixed_digest,
        "model": "VelocityMLP: 5 -> 64 -> 64 -> 2; ReLU hidden layers; linear output",
        "model_seed": MODEL_SEED,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "optimizer_defaults": optimizer.defaults,
        "updates": args.steps,
        "loss": "mean over samples of sum over two velocity components of squared error",
        "max_final_to_initial_loss_ratio": MAX_LOSS_RATIO,
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "torch_version": str(torch.__version__),
    }
    metrics = {
        "passed": passed,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "final_to_initial_loss_ratio": loss_ratio,
        "best_loss": min(row["loss"] for row in rows),
        "completed_updates": args.steps,
        "all_predictions_losses_gradients_parameters_finite": True,
        "fixed_batch_unchanged": True,
        "parameter_gradient_norms": {
            name: {"min": min(norms), "max": max(norms), "last": norms[-1]}
            for name, norms in parameter_gradients.items()
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in (("config.json", config), ("metrics.json", metrics)):
        (args.output_dir / filename).write_text(
            json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    with (args.output_dir / "loss.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("step", "loss", "grad_l2"))
        writer.writeheader()
        writer.writerows(rows)
    torch.save(batch, args.output_dir / "fixed_batch.pt")
    model.eval()
    torch.save({"model_state_dict": model.state_dict(), "config": config}, args.output_dir / "model.pt")

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.semilogy([row["step"] for row in rows], [row["loss"] for row in rows], label="Fixed batch loss")
    ax.axhline(initial_loss * MAX_LOSS_RATIO, color="tab:red", linestyle="--", label="1% of initial loss")
    if args.steps > DEFAULT_STEPS:
        ax.axvline(DEFAULT_STEPS, color="gray", linestyle=":", label="Original 500-update budget")
    ax.set(xlabel="Completed parameter updates", ylabel="Mean squared velocity-vector error",
           title="32 fixed training samples: bounded fitting diagnostic")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(args.output_dir / "loss.png", dpi=150)
    plt.close(fig)

    print(f"初始 loss={initial_loss:.8e}，最终 loss={final_loss:.8e}，比例={loss_ratio:.6e}")
    print(f"固定监督诊断：{'通过' if passed else '未通过'}；完成 {args.steps} 次更新")
    print(f"预测、loss、梯度和参数均有限；固定 batch 未改变。结果：{args.output_dir}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
