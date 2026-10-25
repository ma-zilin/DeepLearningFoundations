"""F4：固定权重、共享起点的条件有效性与 Euler 步数评估。

运行：python -m FlowMatching.evaluate
复现：python -m FlowMatching.evaluate --verify FlowMatching/artifacts/f4_evaluation
协议先于正式采样写入；不训练模型，不覆盖已有结果目录。
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import torch

from .data import CENTERS, PEAK_STD
from .model import VelocityMLP
from .sample import sample_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = {
    "mode_ratio_min": 0.40,
    "mode_ratio_max": 0.60,
    "center_error_max": 0.30,
    "coordinate_std_min": 0.15,
    "coordinate_std_max": 0.55,
    "abs_off_axis_mean_max": 0.15,
    "nearest_center_radius": 0.90,
    "outside_fraction_max": 0.05,
}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    names = ("evaluate.py", "visualize_evaluation.py", "sample.py", "model.py", "conditioning.py", "data.py")
    return {name: sha256(Path(__file__).with_name(name)) for name in names}


def load_model(path: Path) -> tuple[VelocityMLP, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    with torch.random.fork_rng(devices=[]):
        model = VelocityMLP()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def distribution_metrics(points: torch.Tensor, condition: int, thresholds: dict) -> dict:
    """按符号划分区域，保留所有点；float64 统计，逐峰逐坐标检查样本标准差。

    区域编号不是生成点的真实混合成分标签；中心只用于评估，不传给采样器。
    区域少于两个点时标准差为 None，配置失败，仍保留其余统计结果。
    """
    if condition not in (0, 1):
        raise ValueError("condition 必须为 0 或 1")
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("points 必须为非空 [B,2]")
    if not points.is_floating_point() or not torch.isfinite(points).all():
        raise ValueError("points 必须为有限浮点数")
    points = points.detach().to(device="cpu", dtype=torch.float64)
    centers = CENTERS[condition].to(dtype=torch.float64)
    region = (points[:, condition] >= 0).long()
    result = {}
    failures = []
    for mode in (0, 1):
        group = points[region == mode]
        count = len(group)
        prefix = f"group{mode}"
        ratio = count / len(points)
        mean = group.mean(dim=0) if count else None
        std = group.std(dim=0, correction=1) if count >= 2 else None
        error = torch.linalg.vector_norm(mean - centers[mode]).item() if count else None
        result.update({
            f"{prefix}_count": count, f"{prefix}_ratio": ratio,
            f"{prefix}_mean_x": mean[0].item() if count else None,
            f"{prefix}_mean_y": mean[1].item() if count else None,
            f"{prefix}_std_x": std[0].item() if std is not None else None,
            f"{prefix}_std_y": std[1].item() if std is not None else None,
            f"{prefix}_center_error": error,
        })
        if not thresholds["mode_ratio_min"] <= ratio <= thresholds["mode_ratio_max"]:
            failures.append(f"{prefix}_ratio")
        if error is None or error > thresholds["center_error_max"]:
            failures.append(f"{prefix}_center_error")
        for axis, label in enumerate(("x", "y")):
            if std is None or not thresholds["coordinate_std_min"] <= std[axis].item() <= thresholds["coordinate_std_max"]:
                failures.append(f"{prefix}_std_{label}")
    abs_off_axis_mean = points[:, 1 - condition].mean().abs().item()
    nearest_distance = torch.linalg.vector_norm(points[:, None, :] - centers[None, :, :], dim=2).min(dim=1).values
    outside_fraction = (nearest_distance > thresholds["nearest_center_radius"]).double().mean().item()
    if abs_off_axis_mean > thresholds["abs_off_axis_mean_max"]:
        failures.append("abs_off_axis_mean")
    if outside_fraction > thresholds["outside_fraction_max"]:
        failures.append("outside_fraction")
    result.update({
        "abs_off_axis_mean": abs_off_axis_mean,
        "outside_fraction": outside_fraction,
        "passed": not failures, "failed_checks": failures,
    })
    return result


def make_protocol(checkpoint_path: Path, checkpoint: dict) -> dict:
    try:
        label = str(checkpoint_path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        label = str(checkpoint_path.resolve())
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": label, "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_step": checkpoint["step"],
        "checkpoint_validation_loss": checkpoint["validation_loss"],
        "sampling_seeds": [0, 1, 2], "steps": [10, 50, 100], "conditions": [0, 1],
        "samples_per_condition": 4096, "device": "cpu", "dtype": "torch.float32",
        "torch_num_threads": 1, "torch_version": str(torch.__version__),
        "python_version": platform.python_version(), "platform": platform.platform(),
        "cpu_model": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                           if line.startswith("model name")), "unknown"),
        "warmup_runs_per_configuration": 1, "timed_runs_per_configuration": 3,
        "timing_clock": "time.perf_counter; median of three batch latencies",
        "timing_includes": "sample_model: validation, condition encoding, network, Euler, trajectory clones/stack",
        "timing_excludes": "model loading, noise/condition construction, call-count hook, comparisons, metrics, plots, disk I/O",
        "network_calls": "Counted with a temporary forward hook during warmup; one batched call per Euler step",
        "shared_noise": "One local CPU generator per seed; same x0 for all N and both conditions",
        "method": "Forward Euler, t=j/N, j=0,...,N-1; eval and no_grad",
        "target_centers": CENTERS.tolist(), "target_coordinate_std": PEAK_STD,
        "region_rule": "c=0: x<0 / x>=0; c=1: y<0 / y>=0; all endpoints retained",
        "std_definition": "Sample standard deviation, correction=1; checked for each group and coordinate",
        "undefined_rule": "Empty group: mean/error null; group size<2: std null; configuration fails",
        "thresholds": dict(THRESHOLDS),
        "threshold_basis": "Predeclared teaching criteria using historical DDPM scale; not a paper benchmark",
        "historical_difference": "Historical DDPM checked mean of four within-mode stds; here all four must pass",
        "step_reference": "N=100 is a paired sensitivity reference, not exact ODE or target ground truth",
        "scope": "Single trained checkpoint; three sampling seeds; no retraining or algorithm ranking",
        "source_sha256": source_hashes(),
    }


def timed_sample(model: VelocityMLP, x0: torch.Tensor, c: torch.Tensor, steps: int, repeats: int) -> tuple[torch.Tensor, list[float], int]:
    """预热时核对实际调用次数；移除 hook 后计时，同起点重复结果逐元素检查。"""
    calls = 0

    def count_forward(_module, _args, _output):
        nonlocal calls
        calls += 1

    handle = model.register_forward_hook(count_forward)
    try:
        warmup = sample_model(model, x0, c, steps)
    finally:
        handle.remove()
    if calls != steps:
        raise RuntimeError(f"实际调用次数 {calls} 与 Euler 步数 {steps} 不符")
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        trajectory = sample_model(model, x0, c, steps)
        durations.append(time.perf_counter() - start)
        if not torch.equal(trajectory, warmup):
            raise RuntimeError("同一输入的重复采样未逐元素一致")
        # 释放上一轮结果，避免将其析构包含在下一次计时里。
        del trajectory
    return warmup, durations, calls


def paired_comparisons(all_samples: dict[int, dict], protocol: dict) -> dict:
    step_rows, condition_rows = [], []
    reference_steps = max(protocol["steps"])
    for seed, samples in all_samples.items():
        for steps in protocol["steps"]:
            p0, p1 = (samples[f"trajectory_n{steps}_c{c}"][-1].double() for c in (0, 1))
            condition_rows.append({
                "seed": seed, "steps": steps,
                "mean_endpoint_displacement_switching_c": torch.linalg.vector_norm(p1 - p0, dim=1).mean().item(),
                "note": "Displacement alone does not establish conditional quality; use both conditions' metrics",
            })
            for c in protocol["conditions"]:
                points = samples[f"trajectory_n{steps}_c{c}"][-1].double()
                reference = samples[f"trajectory_n{reference_steps}_c{c}"][-1].double()
                distances = torch.linalg.vector_norm(points - reference, dim=1)
                step_rows.append({
                    "seed": seed, "condition": c, "steps": steps, "reference_steps": reference_steps,
                    "mean_endpoint_distance": distances.mean().item(),
                    "max_endpoint_distance": distances.max().item(),
                })
    return {"step_sensitivity": step_rows, "condition_switch": condition_rows}


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for c in (0, 1):
        for steps in (10, 50, 100):
            subset = [r for r in rows if r["condition"] == c and r["steps"] == steps]
            errors = [r[f"group{m}_center_error"] for r in subset for m in (0, 1)]
            stds = [r[f"group{m}_std_{a}"] for r in subset for m in (0, 1) for a in ("x", "y")]
            summaries.append({
                "condition": c, "steps": steps, "seeds_passed": sum(r["passed"] for r in subset),
                "seeds_total": len(subset),
                "median_batch_ms_across_seeds": statistics.median(r["median_batch_ms"] for r in subset),
                "min_batch_ms": min(r["median_batch_ms"] for r in subset),
                "max_batch_ms": max(r["median_batch_ms"] for r in subset),
                "worst_center_error": max(errors) if all(e is not None for e in errors) else None,
                "coordinate_std_min": min(s for s in stds if s is not None) if any(s is not None for s in stds) else None,
                "coordinate_std_max": max(s for s in stds if s is not None) if any(s is not None for s in stds) else None,
                "max_outside_fraction": max(r["outside_fraction"] for r in subset),
            })
    return summaries


def write_tables(output_dir: Path, rows: list[dict], summary: list[dict], paired: dict) -> None:
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()})
    lines = [
        "# F4：固定 checkpoint 的条件与步数评估", "",
        "每组每条件 4096 个样本；CPU 单线程，float32。三个采样 seed，不是三个训练 seed。",
        "计时为预热一次后的三次完整采样中位数，包含轨迹内存操作与运行时检查。", "",
        "| 条件 | 步数/网络调用次数 | 通过 seeds | batch 耗时中位数 (ms) | seed 间耗时范围 (ms) | 最差中心误差 | 峰内各坐标 std 范围 | 最大远离比例 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        error = f"{row['worst_center_error']:.4f}" if row["worst_center_error"] is not None else "未定义"
        std_range = (f"{row['coordinate_std_min']:.4f}–{row['coordinate_std_max']:.4f}"
                     if row["coordinate_std_min"] is not None else "未定义")
        lines.append(f"| {row['condition']} | {row['steps']} | {row['seeds_passed']}/{row['seeds_total']} | "
                     f"{row['median_batch_ms_across_seeds']:.2f} | {row['min_batch_ms']:.2f}–{row['max_batch_ms']:.2f} | "
                     f"{error} | {std_range} | {row['max_outside_fraction']:.2%} |")
    lines += ["", "## 未通过配置", ""]
    failures = [r for r in rows if not r["passed"]]
    lines += [f"- seed={r['seed']}, c={r['condition']}, N={r['steps']}：{', '.join(r['failed_checks'])}" for r in failures] or ["无。"]
    lines += ["", "## 同起点的步数敏感性", "", "以下为与 N=100 终点的平均欧氏距离，按三个 seed 取平均。N=100 不是精确解。", "",
              "| 条件 | 步数 | 平均终点距离 |", "| --- | ---: | ---: |"]
    for c in (0, 1):
        for steps in (10, 50, 100):
            distances = [r["mean_endpoint_distance"] for r in paired["step_sensitivity"] if r["condition"] == c and r["steps"] == steps]
            lines.append(f"| {c} | {steps} | {statistics.mean(distances):.6f} |")
    lines += ["", "## 证据边界", "",
              "- 阈值先写入 config.json，再生成正式结果；所有点均参与统计，各峰各坐标标准差单独检查。",
              "- 区域标签按坐标符号划分，不是真实混合成分标签；矩与区域统计通过仍不证明完整分布相等，需结合散点图。",
              "- 条件切换的位移只说明输出发生变化；正确性还须结合对应条件的质量指标与图像。",
              "- 少步失败如实保留；步数敏感性不等于对精确 ODE 解的误差，也不能完全区分网络误差与积分误差。",
              "- 单 checkpoint、三采样 seed，仅支持此二维实验；耗时是当前含检查和轨迹记录的实现成本。",
              "- 不据此认定已经独立掌握 F4，不与 DDPM 排名，不自动更新学习计划。", ""]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_evaluation(checkpoint_path: Path, output_dir: Path) -> None:
    from .visualize_evaluation import plot_evaluation

    torch.set_num_threads(1)
    model, checkpoint = load_model(checkpoint_path)
    protocol = make_protocol(checkpoint_path, checkpoint)
    output_dir.mkdir(parents=True, exist_ok=False)
    # 必须先保存不可随结果调整的协议，再构造正式实验起点。
    write_json(output_dir / "config.json", protocol)
    original_params = {name: value.clone() for name, value in model.state_dict().items()}
    original_rng = torch.random.get_rng_state().clone()
    all_samples, rows = {}, []
    for seed in protocol["sampling_seeds"]:
        rng = torch.Generator(device="cpu").manual_seed(seed)
        x0 = torch.randn((protocol["samples_per_condition"], 2), generator=rng)
        original_x0 = x0.clone()
        samples = {"x0": x0}
        for steps in protocol["steps"]:
            for condition in protocol["conditions"]:
                c = torch.full((len(x0),), condition, dtype=torch.int64)
                trajectory, durations, calls = timed_sample(model, x0, c, steps, protocol["timed_runs_per_configuration"])
                assert trajectory.shape == (steps + 1, len(x0), 2)
                assert torch.equal(trajectory[0], original_x0) and torch.equal(x0, original_x0)
                assert torch.equal(c, torch.full_like(c, condition))
                assert torch.isfinite(trajectory).all() and not trajectory.requires_grad
                samples[f"trajectory_n{steps}_c{condition}"] = trajectory
                row = {"seed": seed, "condition": condition, "steps": steps,
                       **distribution_metrics(trajectory[-1], condition, protocol["thresholds"]),
                       "network_calls_per_batch": calls, "timed_batch_ms": [t * 1000 for t in durations],
                       "median_batch_ms": statistics.median(durations) * 1000}
                rows.append(row)
                print(f"seed={seed}, c={condition}, N={steps}: pass={row['passed']}, "
                      f"{row['median_batch_ms']:.2f} ms, failures={row['failed_checks']}", flush=True)
        torch.save(samples, output_dir / f"samples_seed{seed}.pt")
        all_samples[seed] = samples
    assert all(torch.equal(v, original_params[k]) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters()) and not model.training
    assert torch.equal(original_rng, torch.random.get_rng_state())
    assert sha256(checkpoint_path) == protocol["checkpoint_sha256"]
    assert source_hashes() == protocol["source_sha256"]
    summary = summarize(rows)
    paired = paired_comparisons(all_samples, protocol)
    write_json(output_dir / "metrics.json", {"rows": rows, "summary": summary})
    write_json(output_dir / "paired_comparisons.json", paired)
    write_tables(output_dir, rows, summary, paired)
    plot_evaluation(all_samples, rows, protocol, output_dir)
    write_json(output_dir / "checks.json", {
        "execution_passed": True, "configuration_count": len(rows),
        "configurations_meeting_quality_thresholds": sum(r["passed"] for r in rows),
        "protocol_saved_before_formal_noise": True,
        "all_trajectories_finite": True, "fixed_conditions": True,
        "shared_initial_noise_for_steps_and_conditions": True,
        "warmup_and_three_timed_trajectories_bitwise_equal": True,
        "network_calls_match_steps": True, "model_eval_and_no_grad": True,
        "model_parameters_unchanged": True, "global_rng_unchanged_during_sampling": True,
        "checkpoint_and_source_hashes_unchanged": True,
        "sample_files_sha256": {f"samples_seed{s}.pt": sha256(output_dir / f"samples_seed{s}.pt") for s in all_samples},
    })
    print(f"完成 {len(rows)} 组；质量阈值通过 {sum(r['passed'] for r in rows)}/{len(rows)}。结果：{output_dir}")


def verify_evaluation(output_dir: Path) -> None:
    """新进程重建全部起点和轨迹；不把计时数值当作可逐元素复现的量。"""
    protocol = json.loads((output_dir / "config.json").read_text())
    checks = json.loads((output_dir / "checks.json").read_text())
    checkpoint_path = Path(protocol["checkpoint"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    if sha256(checkpoint_path) != protocol["checkpoint_sha256"] or source_hashes() != protocol["source_sha256"]:
        raise RuntimeError("权重或源码指纹已变化")
    if str(torch.__version__) != protocol["torch_version"]:
        raise RuntimeError("PyTorch 版本与原实验不同")
    torch.set_num_threads(protocol["torch_num_threads"])
    model, _ = load_model(checkpoint_path)
    recorded = json.loads((output_dir / "metrics.json").read_text())["rows"]
    all_samples = {}
    verified = 0
    for seed in protocol["sampling_seeds"]:
        path = output_dir / f"samples_seed{seed}.pt"
        if sha256(path) != checks["sample_files_sha256"][path.name]:
            raise RuntimeError("保存张量的指纹不符")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        rng = torch.Generator(device="cpu").manual_seed(seed)
        x0 = torch.randn((protocol["samples_per_condition"], 2), generator=rng)
        assert torch.equal(x0, saved["x0"])
        for steps in protocol["steps"]:
            for c in protocol["conditions"]:
                condition = torch.full((len(x0),), c, dtype=torch.int64)
                repeated = sample_model(model, x0, condition, steps)
                assert torch.equal(saved[f"trajectory_n{steps}_c{c}"], repeated)
                row = next(r for r in recorded if (r["seed"], r["condition"], r["steps"]) == (seed, c, steps))
                for name, value in distribution_metrics(repeated[-1], c, protocol["thresholds"]).items():
                    assert value == row[name], (seed, c, steps, name)
                verified += 1
        all_samples[seed] = saved
    assert paired_comparisons(all_samples, protocol) == json.loads((output_dir / "paired_comparisons.json").read_text())
    write_json(output_dir / "reproduction.json", {
        "passed": True, "configurations_verified": verified,
        "all_initial_noise_and_full_trajectories_bitwise_equal": True,
        "maximum_absolute_difference": 0.0, "all_quality_metrics_equal": True,
        "paired_comparisons_equal": True, "checkpoint_and_source_hashes_match": True,
        "timings_reproduced": False, "note": "Wall-clock times vary; original raw timings retained",
    })
    print(f"新进程复现通过：{verified} 组完整轨迹逐元素一致，全部指标一致。")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parent / "artifacts/f2_training/best.pt")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts/f4_evaluation")
    parser.add_argument("--verify", type=Path, help="在新进程复现指定结果目录，不重复计时")
    args = parser.parse_args()
    if args.verify is not None:
        verify_evaluation(args.verify)
    else:
        run_evaluation(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
