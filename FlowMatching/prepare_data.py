"""F1：构造可复现的训练、验证数据，并保存检查结果。

仓库根目录运行：python -m FlowMatching.prepare_data
训练文件中的路径是 F1 快照；F2 正式训练需要重新采样起点和时间。
验证文件中的端点、起点和时间保持固定，便于比较预测误差。
"""

import argparse
import json
from pathlib import Path

import torch

from .conditioning import build_velocity_input, encode_condition
from .data import CENTERS, PEAK_STD, sample_target_data
from .path import linear_path, sample_path_batch


SPLITS = {
    "train": {"size": 8192, "target_seed": 0, "path_seed": 1},
    "validation": {"size": 2048, "target_seed": 2, "path_seed": 3},
}
# 这是当前有限样本的数据合理性检查阈值，不是模型质量标准。
TOLERANCES = {
    "condition_fraction_error": 0.04,
    "mode_fraction_error": 0.07,
    "center_coordinate_error": 0.06,
    "std_coordinate_error": 0.04,
}


def build_split(size: int, target_seed: int, path_seed: int) -> dict[str, torch.Tensor]:
    """分别采样目标和路径；每次调用均创建本地 CPU 随机数生成器。"""
    target_rng = torch.Generator(device="cpu").manual_seed(target_seed)
    path_rng = torch.Generator(device="cpu").manual_seed(path_seed)
    x1, c, mode_ids = sample_target_data(size, target_rng)
    x0, t, xt, velocity = sample_path_batch(x1, path_rng)
    return {
        "x0": x0,
        "x1": x1,
        "t": t,
        "xt": xt,
        "velocity": velocity,
        "c": c,
        "mode_ids": mode_ids,
        "c_onehot": encode_condition(c, dtype=xt.dtype),
        "network_input": build_velocity_input(xt, t, c),
    }


def check_split(batch: dict[str, torch.Tensor]) -> dict:
    """检查形状、路径端点、有限差分，并按条件和峰统计目标数据。"""
    size = len(batch["x1"])
    widths = {"x0": 2, "x1": 2, "t": 1, "xt": 2, "velocity": 2,
              "c_onehot": 2, "network_input": 5}
    for name, tensor in batch.items():
        expected = (size,) if name in ("c", "mode_ids") else (size, widths[name])
        assert tensor.shape == expected, f"{name} 形状错误"
        assert torch.isfinite(tensor).all(), f"{name} 含非有限值"
        expected_dtype = torch.int64 if name in ("c", "mode_ids") else torch.float32
        assert tensor.dtype == expected_dtype and tensor.device.type == "cpu"

    x0, x1 = batch["x0"], batch["x1"]
    for time, endpoint in ((0.0, x0), (1.0, x1)):
        xt, _ = linear_path(x0, x1, torch.full((size, 1), time))
        torch.testing.assert_close(xt, endpoint, rtol=0, atol=0)

    # 在这份数据的固定端点上做中心有限差分，用 float64 减少舍入误差。
    starts, ends = x0[:32].double(), x1[:32].double()
    times = 0.25 + 0.5 * batch["t"][:32].double()
    step = 1e-5
    plus, _ = linear_path(starts, ends, times + step)
    minus, _ = linear_path(starts, ends, times - step)
    _, velocity = linear_path(starts, ends, times)
    error = ((plus - minus) / (2 * step) - velocity).abs().max().item()
    assert error < 1e-8, "有限差分速度检查失败"

    conditions, modes = batch["c"], batch["mode_ids"]
    groups = []
    for condition in (0, 1):
        condition_count = int((conditions == condition).sum())
        assert abs(condition_count / size - 0.5) < TOLERANCES["condition_fraction_error"]
        for mode in (0, 1):
            points = x1[(conditions == condition) & (modes == mode)]
            mean, std = points.mean(dim=0), points.std(dim=0, unbiased=False)
            fraction = len(points) / condition_count
            assert abs(fraction - 0.5) < TOLERANCES["mode_fraction_error"]
            assert (mean - CENTERS[condition, mode]).abs().max() < TOLERANCES["center_coordinate_error"]
            assert (std - PEAK_STD).abs().max() < TOLERANCES["std_coordinate_error"]
            groups.append({
                "condition": condition, "mode": mode, "count": len(points),
                "fraction_within_condition": fraction,
                "mean": mean.tolist(), "std": std.tolist(),
            })
    return {
        "shapes": {name: list(tensor.shape) for name, tensor in batch.items()},
        "endpoint_checks_passed": True,
        "finite_difference_max_error": error,
        "target_distribution_checks_passed": True,
        "target_groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "f1_data",
    )
    args = parser.parse_args()
    batches = {name: build_split(**settings) for name, settings in SPLITS.items()}
    reports = {name: check_split(batch) for name, batch in batches.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, batch in batches.items():
        torch.save(batch, args.output_dir / f"{name}.pt")

    configuration = {
        "splits": SPLITS,
        "centers": CENTERS.tolist(),
        "peak_std": PEAK_STD,
        "condition_probabilities": [0.5, 0.5],
        "mode_probabilities_within_condition": [0.5, 0.5],
        "source": "Independent standard 2D Gaussian",
        "time": "Independent Uniform[0, 1), one scalar per sample",
        "train_usage": "F1 path snapshot; resample x0 and t during F2 training",
        "validation_usage": "Keep x1, c, x0 and t fixed across checkpoint evaluations",
        "network_input_columns": ["x", "y", "t", "c0", "c1"],
        "sanity_check_tolerances": TOLERANCES,
        "torch_version": torch.__version__,
    }
    for filename, content in (("config.json", configuration), ("checks.json", reports)):
        (args.output_dir / filename).write_text(
            json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
    for name, report in reports.items():
        print(f"{name}: {report['shapes']}")
        print(f"  端点和双峰统计检查通过，有限差分最大误差={report['finite_difference_max_error']:.3e}")
    print(f"数据和检查结果已保存至 {args.output_dir}")


if __name__ == "__main__":
    main()
