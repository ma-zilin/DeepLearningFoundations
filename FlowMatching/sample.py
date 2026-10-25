"""F3：欧拉积分、自检及训练网络的新噪声生成。

解析自检：python -m FlowMatching.sample
模型生成：python -m FlowMatching.sample --generate
新进程复现：python -m FlowMatching.sample --verify FlowMatching/artifacts/f3_sampling
"""

import argparse
from collections.abc import Callable
import hashlib
import json
from pathlib import Path

import torch

from .conditioning import build_velocity_input
from .model import VelocityMLP


@torch.no_grad()
def euler_integrate(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    c: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    """从 t=0 积分到 t=1，返回 [num_steps+1, B, 2] 的位置轨迹。

    x0 为非空的 [B, 2] 浮点起点；c 为同设备的 [B] int64 任务编号。
    第 j 次查询使用当前 x、填满 j/num_steps 的 [B, 1] 时间和固定 c。
    velocity_fn 返回与 x 同形状、dtype 和 device 的速度，不应原地修改
    接收到的位置、时间或条件。若封装神经网络，调用方需先设置 model.eval()。

    本函数关闭梯度并保留原始输入；不采样起点，不读取目标端点。
    返回值第 0 项是起点，第 j 项是完成 j 次更新后的位置。
    """
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError("num_steps 必须是正整数")
    if x0.ndim != 2 or x0.shape[1] != 2 or x0.shape[0] == 0 or not x0.is_floating_point():
        raise ValueError("x0 必须是非空的 [B, 2] 浮点张量")
    if c.shape != (x0.shape[0],) or c.dtype != torch.int64 or c.device != x0.device:
        raise ValueError("c 必须为与 x0 同设备的 [B] int64 张量")
    if ((c < 0) | (c > 1)).any():
        raise ValueError("任务条件 c 只能为 0 或 1")
    if not torch.isfinite(x0).all():
        raise ValueError("x0 必须全部为有限数值")

    x = x0.clone()
    trajectory = [x.clone()]
    h = 1.0 / num_steps
    for j in range(num_steps):
        t = torch.full((len(x), 1), j / num_steps, dtype=x.dtype, device=x.device)
        velocity = velocity_fn(x, t, c)
        if not isinstance(velocity, torch.Tensor) or velocity.shape != x.shape:
            raise ValueError("velocity_fn 必须返回 [B, 2] 速度张量")
        if velocity.dtype != x.dtype or velocity.device != x.device:
            raise ValueError("速度必须与位置具有相同的 dtype 和 device")
        if not torch.isfinite(velocity).all():
            raise FloatingPointError(f"第 {j} 步速度包含 NaN/Inf")

        x = x + h * velocity
        if not torch.isfinite(x).all():
            raise FloatingPointError(f"第 {j + 1} 步位置包含 NaN/Inf")
        trajectory.append(x.clone())
    return torch.stack(trajectory, dim=0)


def constant_velocity_checks() -> dict:
    """用已知解析轨迹检查 10/20 步更新，不调用训练网络。"""
    x0 = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    c = torch.tensor([0], dtype=torch.int64)
    velocity = torch.tensor([[2.0, -1.0]], dtype=x0.dtype)
    original_x0, original_c = x0.clone(), c.clone()
    tolerance = 1e-6
    cases = []

    for num_steps in (10, 20):
        queried_times = []

        def constant_field(x, t, condition):
            assert not torch.is_grad_enabled()
            assert torch.equal(condition, original_c)
            queried_times.append(t.clone())
            return velocity.expand_as(x)

        trajectory = euler_integrate(constant_field, x0, c, num_steps)
        times = torch.arange(num_steps + 1, dtype=x0.dtype) / num_steps
        expected = x0.unsqueeze(0) + times[:, None, None] * velocity.unsqueeze(0)
        torch.testing.assert_close(trajectory, expected, rtol=0, atol=tolerance)
        torch.testing.assert_close(trajectory[0], x0, rtol=0, atol=0)
        assert trajectory.shape == (num_steps + 1, 1, 2)
        assert len(queried_times) == num_steps
        torch.testing.assert_close(
            torch.stack(queried_times), times[:-1, None, None], rtol=0, atol=tolerance,
        )
        assert torch.equal(x0, original_x0) and torch.equal(c, original_c)
        assert not trajectory.requires_grad
        cases.append({
            "num_steps": num_steps,
            "step_size": 1.0 / num_steps,
            "trajectory_shape": list(trajectory.shape),
            "velocity_calls": len(queried_times),
            "final_position": trajectory[-1].tolist(),
            "max_trajectory_error": (trajectory - expected).abs().max().item(),
            "max_terminal_error": (trajectory[-1] - expected[-1]).abs().max().item(),
            "time_grid_checked": True,
            "condition_fixed": True,
            "gradient_disabled": True,
            "inputs_unchanged": True,
        })
    return {
        "passed": True,
        "scope": "Analytical constant-velocity integration only, not model generation quality",
        "initial_position": x0.tolist(),
        "constant_velocity": velocity.tolist(),
        "expected_endpoint": (x0 + velocity).tolist(),
        "time_interval": [0.0, 1.0],
        "absolute_tolerance": tolerance,
        "dtype": str(x0.dtype),
        "device": str(x0.device),
        "torch_version": str(torch.__version__),
        "cases": cases,
    }


def sample_model(model: VelocityMLP, x0: torch.Tensor, c: torch.Tensor, num_steps: int) -> torch.Tensor:
    """固定模型参数和任务条件，复用欧拉积分器生成完整轨迹。"""
    model.eval()
    fixed_c = c.clone()

    def model_velocity(x, t, condition):
        if torch.is_grad_enabled() or model.training:
            raise RuntimeError("采样必须在 eval 和无梯度模式下执行")
        if not torch.equal(condition, fixed_c):
            raise RuntimeError("采样过程中的任务条件发生变化")
        return model(build_velocity_input(x, t, condition))

    return euler_integrate(model_velocity, x0, c, num_steps)


def generate_trajectories(checkpoint_path: Path, num_samples: int, num_steps: int, seed: int) -> tuple[dict, dict]:
    """两种条件使用同一批新高斯起点；只加载模型权重，不加载训练数据。"""
    if num_samples <= 0:
        raise ValueError("num_samples 必须为正整数")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # 构造模型所需的临时初始化不影响调用方的全局随机状态。
    with torch.random.fork_rng(devices=[]):
        model = VelocityMLP()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    original_parameters = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.Generator(device="cpu").manual_seed(seed)
    x0 = torch.randn((num_samples, 2), generator=rng, dtype=torch.float32, device="cpu")
    original_x0 = x0.clone()
    samples = {
        "x0": x0,
        "times": torch.arange(num_steps + 1, dtype=x0.dtype) / num_steps,
    }
    for condition in (0, 1):
        c = torch.full((num_samples,), condition, dtype=torch.int64)
        trajectory = sample_model(model, x0, c, num_steps)
        assert trajectory.shape == (num_steps + 1, num_samples, 2)
        assert torch.equal(trajectory[0], original_x0)
        assert torch.equal(c, torch.full_like(c, condition))
        assert torch.isfinite(trajectory).all() and not trajectory.requires_grad
        samples[f"trajectory_c{condition}"] = trajectory
    assert torch.equal(x0, original_x0)
    assert all(torch.equal(value, original_parameters[name]) for name, value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())
    report = {
        "checkpoint_step": checkpoint["step"],
        "checkpoint_validation_loss": checkpoint["validation_loss"],
        "trajectory_shapes": {f"c{c}": list(samples[f"trajectory_c{c}"].shape) for c in (0, 1)},
        "all_positions_finite": True,
        "same_initial_noise_for_both_conditions": True,
        "initial_noise_unchanged": True,
        "conditions_fixed": True,
        "model_parameters_unchanged": True,
        "model_eval_and_no_grad": True,
        "scope": "Single-seed trajectory and point-cloud inspection, not F4 quantitative quality evaluation",
    }
    return samples, report


def run_generation(checkpoint_path: Path, num_samples: int, num_steps: int, seed: int, output_dir: Path) -> None:
    from .visualize_samples import plot_generated_samples

    torch.set_num_threads(1)
    samples, report = generate_trajectories(checkpoint_path, num_samples, num_steps, seed)
    project_root = Path(__file__).resolve().parent.parent
    try:
        checkpoint_label = str(checkpoint_path.resolve().relative_to(project_root))
    except ValueError:
        checkpoint_label = str(checkpoint_path.resolve())
    config = {
        "checkpoint": checkpoint_label,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "checkpoint_step": report["checkpoint_step"],
        "seed": seed,
        "samples_per_condition": num_samples,
        "num_steps": num_steps,
        "step_size": 1.0 / num_steps,
        "conditions": [0, 1],
        "source": "New standard 2D Gaussian, shared between conditions",
        "method": "Forward Euler, t=j/N, j=0,...,N-1",
        "network_input_columns": ["x", "y", "t", "c0", "c1"],
        "device": "cpu",
        "dtype": str(samples["x0"].dtype),
        "torch_num_threads": torch.get_num_threads(),
        "torch_version": str(torch.__version__),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(samples, output_dir / "samples.pt")
    for filename, content in (("config.json", config), ("checks.json", report)):
        (output_dir / filename).write_text(
            json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
    plot_generated_samples(samples, output_dir)
    print(f"checkpoint step={report['checkpoint_step']}；seed={seed}；Euler N={num_steps}")
    print(f"两种条件各 {num_samples} 个样本，轨迹形状={report['trajectory_shapes']}")
    print(f"有限值、固定条件、无梯度及参数保留检查通过。结果：{output_dir}")


def verify_generation(output_dir: Path) -> None:
    """按保存配置重载权重，逐元素核对完整轨迹，不重新训练。"""
    config = json.loads((output_dir / "config.json").read_text())
    checkpoint_path = Path(config["checkpoint"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(__file__).resolve().parent.parent / checkpoint_path
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != config["checkpoint_sha256"]:
        raise RuntimeError("checkpoint 内容已变化，无法按原配置复现")
    torch.set_num_threads(config["torch_num_threads"])
    saved = torch.load(output_dir / "samples.pt", map_location="cpu", weights_only=True)
    repeated, _ = generate_trajectories(
        checkpoint_path, config["samples_per_condition"], config["num_steps"], config["seed"],
    )
    if saved.keys() != repeated.keys():
        raise RuntimeError("保存结果与复现结果的张量字段不同")
    differences = {}
    for name, tensor in saved.items():
        if not torch.equal(tensor, repeated[name]):
            raise RuntimeError(f"{name} 无法逐元素复现")
        differences[name] = (tensor - repeated[name]).abs().max().item()
    report = {
        "passed": True,
        "checkpoint_sha256_matches": True,
        "all_saved_tensors_bitwise_equal": True,
        "maximum_absolute_differences": differences,
        "torch_version": str(torch.__version__),
    }
    (output_dir / "reproduction.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"同 seed 完整轨迹逐元素复现通过，最大差均为 0。记录：{output_dir / 'reproduction.json'}")


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--generate", action="store_true", help="加载训练权重生成两种条件的样本")
    mode.add_argument("--verify", type=Path, help="读取结果目录中的配置并复现完整轨迹")
    parser.add_argument("--checkpoint", type=Path, default=project_dir / "artifacts" / "f2_training" / "best.pt")
    parser.add_argument("--samples", type=int, default=1024, help="每种条件的样本数")
    parser.add_argument("--steps", type=int, default=100, help="欧拉积分步数")
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument(
        "--output-dir", type=Path,
    )
    args = parser.parse_args()
    if args.verify is not None:
        verify_generation(args.verify)
        return
    if args.generate:
        if args.samples <= 0 or args.steps <= 0:
            parser.error("samples 和 steps 必须为正整数")
        output_dir = args.output_dir or project_dir / "artifacts" / "f3_sampling"
        run_generation(args.checkpoint, args.samples, args.steps, args.seed, output_dir)
        return
    output_dir = args.output_dir or project_dir / "artifacts" / "f3_euler_checks"
    report = constant_velocity_checks()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checks.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    for case in report["cases"]:
        print(f"N={case['num_steps']}: trajectory={case['trajectory_shape']}, "
              f"endpoint={case['final_position']}, max_error={case['max_trajectory_error']:.3e}")
    print(f"常量速度、时间网格、调用次数和输入保留检查通过。结果：{output_dir}")


if __name__ == "__main__":
    main()
