"""F5：复用历史 DDPM 权重和采样器，按 F4 相同规则重新评估水平双峰。

python -m FlowMatching.reevaluate_ddpm
python -m FlowMatching.reevaluate_ddpm --verify FlowMatching/artifacts/f5_ddpm_reevaluation
不训练、不改变旧采样公式、不覆盖旧 CSV；本次不比较墙钟耗时。
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch

from .evaluate import PROJECT_ROOT, distribution_metrics, sha256, write_json


DDPM_DIR = PROJECT_ROOT / "diffusion"
# 历史 DDPM 使用同目录绝对导入；仅适配导入路径，不修改旧文件。
sys.path.insert(0, str(DDPM_DIR))
from reverse_sampling import load_model_and_schedule, sample_reverse_process


def source_hashes() -> dict[str, str]:
    names = ["FlowMatching/reevaluate_ddpm.py", "FlowMatching/evaluate.py", "FlowMatching/data.py",
             "diffusion/reverse_sampling.py", "diffusion/noise_prediction.py",
             "diffusion/forward_diffusion.py", "diffusion/bimodal_data.py"]
    return {name: sha256(PROJECT_ROOT / name) for name in names}


def load_ddpm(path: Path):
    with torch.random.fork_rng(devices=[]):
        return load_model_and_schedule(path)


def sample_endpoints(model, schedule: tuple, seed: int, count: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    snapshots = sample_reverse_process(model, count, *schedule, generator)
    assert all(points.shape == (count, 2) and torch.isfinite(points).all() for _, points in snapshots)
    endpoints = snapshots[-1][1]
    assert snapshots[-1][0] == 0 and not endpoints.requires_grad
    return endpoints


def check_historical(row: dict, old: dict) -> float:
    """同一批样本应复现旧统计；旧 float32 / 新 float64 的舍入允许绝对差 2e-6。"""
    names = {"group0_ratio": "left_ratio", "group1_ratio": "right_ratio",
             "group0_mean_x": "left_center_x", "group0_mean_y": "left_center_y",
             "group1_mean_x": "right_center_x", "group1_mean_y": "right_center_y",
             "group0_center_error": "left_center_error", "group1_center_error": "right_center_error",
             "group0_std_x": "left_std_x", "group0_std_y": "left_std_y",
             "group1_std_x": "right_std_x", "group1_std_y": "right_std_y"}
    differences = [abs(row[new] - float(old[previous])) for new, previous in names.items()]
    differences.append(abs(row["abs_off_axis_mean"] - abs(float(old["y_mean"]))))
    within_std = sum(row[f"group{m}_std_{a}"] for m in (0, 1) for a in ("x", "y")) / 4
    differences.append(abs(within_std - float(old["mean_within_std"])))
    if max(differences) > 2e-6:
        raise RuntimeError(f"seed={row['seed']} 未复现旧 CSV 统计，最大差 {max(differences)}；请检查来源")
    return max(differences)


def write_comparison(output_dir: Path, rows: list[dict], fm_rows: list[dict], ddpm_points: dict[int, torch.Tensor], fm_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    combined = [{"method": "DDPM", **row} for row in rows]
    combined += [{"method": "CFM", **{k: row[k] for k in rows[0]}} for row in fm_rows]
    with (output_dir / "quality_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(combined[0]))
        writer.writeheader()
        for row in combined:
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})
    lines = ["# F5：历史 DDPM 权重重评测", "",
             "同一水平双峰目标；每组 4096 个样本，三个采样 seed；统一使用 F4 的全部质量阈值。", "",
             "| 方法 | 步数/每批网络调用次数 | 通过 seeds | 最大中心误差 | 各峰各坐标 std 范围 | 最大远离比例 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for method, steps in [("DDPM", rows[0]["steps"]), ("CFM", 10), ("CFM", 50), ("CFM", 100)]:
        group = [r for r in combined if r["method"] == method and r["steps"] == steps]
        errors = [r[f"group{m}_center_error"] for r in group for m in (0, 1)]
        stds = [r[f"group{m}_std_{a}"] for r in group for m in (0, 1) for a in ("x", "y")]
        lines.append(f"| {method} | {steps} | {sum(r['passed'] for r in group)}/{len(group)} | "
                     f"{max(errors):.4f} | {min(stds):.4f}–{max(stds):.4f} | {max(r['outside_fraction'] for r in group):.2%} |")
    lines += ["", "- DDPM 使用原权重、原 1000 步采样器，补算每坐标标准差检查与远离比例；旧文件保持原样。",
              "- 旧 CSV 各 seed 的 14 个既有统计量均在 2e-6 绝对容差内复现；容差用于 float32 / float64 统计舍入差异。",
              "- 这是统一质量规则下两个现有模型的表现比较；架构、输入条件和训练预算不同，不能归因为算法优劣。",
              "- 本次未按共同计时协议重测墙钟耗时；1000 次与 100 次网络调用不等于耗时恰好相差十倍。",
              "- 同 seed 的两种方法可有相同初始高斯点，但旧 DDPM 在中间步骤还会抽取新噪声；不把它们当作同一条生成轨迹。",
              "- 每个点参与全部区域统计，未删除离群点；图中两个方法的所有样本共用坐标范围。", ""]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    fm_points = {s: torch.load(fm_dir / f"samples_seed{s}.pt", map_location="cpu", weights_only=True)["trajectory_n100_c0"][-1]
                 for s in ddpm_points}
    limit = max(3.0, max(p.abs().max().item() for p in [*ddpm_points.values(), *fm_points.values()]) + 0.3)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    for row_index, (method, points_by_seed, color) in enumerate((("DDPM N=1000", ddpm_points, "#5a7d36"),
                                                               ("CFM c=0 N=100", fm_points, "#2878b5"))):
        for column, (seed, points) in enumerate(points_by_seed.items()):
            ax = axes[row_index, column]
            ax.scatter(points[:, 0], points[:, 1], s=3, alpha=0.3, color=color)
            ax.scatter([-2, 2], [0, 0], color="black", marker="x", s=50)
            for x in (-2, 2):
                ax.add_patch(plt.Circle((x, 0), 0.9, fill=False, color="gray", linestyle="--", alpha=0.5))
            ax.set(title=f"{method} | seed={seed}", xlim=(-limit, limit), ylim=(-limit, limit), xlabel="x", ylabel="y")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.15)
    fig.suptitle("Same target and quality criteria; different models and training setups")
    fig.savefig(output_dir / "endpoints_comparison.png", dpi=140)
    plt.close(fig)


def run(output_dir: Path) -> None:
    torch.set_num_threads(1)
    checkpoint_path = DDPM_DIR / "outputs/noise_predictor_g3b.pt"
    historical_path = DDPM_DIR / "outputs/distribution_metrics.csv"
    fm_dir = Path(__file__).resolve().parent / "artifacts/f4_evaluation"
    fm_protocol = json.loads((fm_dir / "config.json").read_text())
    model, *schedule = load_ddpm(checkpoint_path)
    metadata = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert metadata["data_std"] == 0.3 and len(schedule[0]) == 1000
    with historical_path.open(newline="") as stream:
        historical = {int(r["sampling_seed"]): r for r in csv.DictReader(stream)}
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)), "checkpoint_sha256": sha256(checkpoint_path),
        "historical_csv_sha256": sha256(historical_path),
        "fm_config_sha256": sha256(fm_dir / "config.json"), "fm_metrics_sha256": sha256(fm_dir / "metrics.json"),
        "checkpoint_metadata": {k: v for k, v in metadata.items() if k != "model_state_dict"},
        "sampling_seeds": [0, 1, 2], "samples_per_seed": 4096, "steps": len(schedule[0]),
        "device": "cpu", "dtype": "torch.float32", "torch_num_threads": 1, "torch_version": str(torch.__version__),
        "thresholds": fm_protocol["thresholds"], "target_centers": [[-2., 0.], [2., 0.]], "target_std": 0.3,
        "std_definition": "Per group and coordinate sample standard deviation, correction=1; all endpoints retained",
        "source_sha256": source_hashes(), "wall_clock_comparison": "Not measured in this quality reevaluation",
        "scope": "Reuse original DDPM weights and stochastic reverse sampler; no training or schedule change",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "config.json", protocol)
    original_params = {k: v.clone() for k, v in model.state_dict().items()}
    rows, points_by_seed, differences = [], {}, {}
    for seed in protocol["sampling_seeds"]:
        points = sample_endpoints(model, tuple(schedule), seed, protocol["samples_per_seed"])
        row = {"seed": seed, "condition": 0, "steps": protocol["steps"], **distribution_metrics(points, 0, protocol["thresholds"])}
        differences[str(seed)] = check_historical(row, historical[seed])
        rows.append(row)
        points_by_seed[seed] = points
        print(f"seed={seed}: pass={row['passed']}, outside={row['outside_fraction']:.4%}, old CSV max diff={differences[str(seed)]:.3e}", flush=True)
    assert not model.training and all(p.grad is None for p in model.parameters())
    assert all(torch.equal(v, original_params[k]) for k, v in model.state_dict().items())
    assert sha256(checkpoint_path) == protocol["checkpoint_sha256"]
    assert sha256(historical_path) == protocol["historical_csv_sha256"]
    assert source_hashes() == protocol["source_sha256"]
    torch.save({f"seed{s}": points for s, points in points_by_seed.items()}, output_dir / "endpoints.pt")
    write_json(output_dir / "metrics.json", {"rows": rows})
    fm_rows = [r for r in json.loads((fm_dir / "metrics.json").read_text())["rows"] if r["condition"] == 0]
    write_comparison(output_dir, rows, fm_rows, points_by_seed, fm_dir)
    write_json(output_dir / "checks.json", {
        "passed": True, "model_fixed_eval_no_grad": True, "all_endpoints_finite": True,
        "historical_metrics_reproduced": True, "historical_absolute_tolerance": 2e-6,
        "max_historical_metric_difference_by_seed": differences,
        "checkpoint_historical_csv_and_sources_unchanged": True,
        "endpoints_sha256": sha256(output_dir / "endpoints.pt"),
    })
    print(f"质量重评测完成：{output_dir}")


def verify(output_dir: Path) -> None:
    protocol = json.loads((output_dir / "config.json").read_text())
    path = PROJECT_ROOT / protocol["checkpoint"]
    assert sha256(path) == protocol["checkpoint_sha256"] and source_hashes() == protocol["source_sha256"]
    assert str(torch.__version__) == protocol["torch_version"]
    torch.set_num_threads(protocol["torch_num_threads"])
    model, *schedule = load_ddpm(path)
    checks = json.loads((output_dir / "checks.json").read_text())
    assert sha256(output_dir / "endpoints.pt") == checks["endpoints_sha256"]
    saved = torch.load(output_dir / "endpoints.pt", map_location="cpu", weights_only=True)
    rows = json.loads((output_dir / "metrics.json").read_text())["rows"]
    for seed in protocol["sampling_seeds"]:
        repeated = sample_endpoints(model, tuple(schedule), seed, protocol["samples_per_seed"])
        assert torch.equal(saved[f"seed{seed}"], repeated)
        recorded = next(r for r in rows if r["seed"] == seed)
        assert all(recorded[k] == v for k, v in distribution_metrics(repeated, 0, protocol["thresholds"]).items())
        print(f"seed={seed}: endpoints bitwise equal", flush=True)
    write_json(output_dir / "reproduction.json", {"passed": True, "seeds_verified": protocol["sampling_seeds"],
               "all_endpoints_bitwise_equal": True, "maximum_absolute_difference": 0.0, "all_quality_metrics_equal": True})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts/f5_ddpm_reevaluation")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(args.verify)
    else:
        run(args.output_dir)


if __name__ == "__main__":
    main()
