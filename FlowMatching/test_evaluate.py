"""F4 指标的反例检查，不调用训练或正式采样。"""

import unittest

import torch

from .evaluate import THRESHOLDS, distribution_metrics


def balanced_points() -> torch.Tensor:
    # 每峰四个点；各坐标的样本标准差均为 sqrt(4*0.26^2/3)≈0.3002。
    offsets = torch.tensor([[-0.26, -0.26], [-0.26, 0.26], [0.26, -0.26], [0.26, 0.26]], dtype=torch.float64)
    return torch.cat((offsets + torch.tensor([-2.0, 0.0]), offsets + torch.tensor([2.0, 0.0])))


class DistributionMetricsChecks(unittest.TestCase):
    def test_balanced_and_rotated_targets(self):
        points = balanced_points()
        horizontal = distribution_metrics(points, 0, THRESHOLDS)
        vertical = distribution_metrics(points[:, [1, 0]], 1, THRESHOLDS)
        self.assertTrue(horizontal["passed"])
        self.assertTrue(vertical["passed"])
        self.assertAlmostEqual(horizontal["group0_std_x"], (4 * 0.26 ** 2 / 3) ** 0.5)
        self.assertEqual(horizontal["group0_ratio"], 0.5)
        self.assertEqual(horizontal["outside_fraction"], 0.0)

    def test_collapsed_peaks_fail_standard_deviation(self):
        points = torch.tensor([[-2., 0.], [-2., 0.], [2., 0.], [2., 0.]])
        result = distribution_metrics(points, 0, THRESHOLDS)
        self.assertFalse(result["passed"])
        self.assertEqual(set(result["failed_checks"]), {f"group{m}_std_{a}" for m in (0, 1) for a in ("x", "y")})

    def test_single_peak_and_singleton_group_are_recorded(self):
        for points, expected_count in ((balanced_points()[4:], 0),
                                      (torch.tensor([[-2., 0.], [2., 0.], [2.1, 0.1]]), 1)):
            with self.subTest(count=expected_count):
                result = distribution_metrics(points, 0, THRESHOLDS)
                self.assertFalse(result["passed"])
                self.assertEqual(result["group0_count"], expected_count)
                self.assertIsNone(result["group0_std_x"])
                if expected_count == 0:
                    self.assertIsNone(result["group0_center_error"])

    def test_each_coordinate_checked_instead_of_std_average(self):
        points = balanced_points()
        points[:, 1] *= 2.5  # 四个 std 的平均约0.525，仍小于0.55；两个 y std 已超标。
        result = distribution_metrics(points, 0, THRESHOLDS)
        stds = [result[f"group{m}_std_{a}"] for m in (0, 1) for a in ("x", "y")]
        self.assertLess(sum(stds) / 4, THRESHOLDS["coordinate_std_max"])
        self.assertIn("group0_std_y", result["failed_checks"])
        self.assertIn("group1_std_y", result["failed_checks"])

    def test_wrong_centers_and_outliers_are_not_filtered(self):
        points = balanced_points()
        points[:, 0] *= 0.05
        result = distribution_metrics(points, 0, THRESHOLDS)
        self.assertEqual(result["group0_ratio"], 0.5)
        self.assertGreater(result["group0_center_error"], 1.8)
        self.assertEqual(result["outside_fraction"], 1.0)
        self.assertEqual(result["group0_count"] + result["group1_count"], len(points))
        self.assertFalse(result["passed"])

    def test_off_axis_bias(self):
        points = balanced_points() + torch.tensor([0., 0.2])
        result = distribution_metrics(points, 0, THRESHOLDS)
        self.assertIn("abs_off_axis_mean", result["failed_checks"])
        self.assertNotIn("group0_center_error", result["failed_checks"])

    def test_region_and_radius_boundaries(self):
        points = torch.tensor([[0., 0.], [-2., 0.], [2., 0.9]], dtype=torch.float64)
        result = distribution_metrics(points, 0, THRESHOLDS)
        self.assertEqual(result["group1_count"], 2)  # 零归正侧。
        self.assertAlmostEqual(result["outside_fraction"], 1 / 3)  # 恰在半径上不算远离。

    def test_nonfinite_input_is_rejected(self):
        with self.assertRaises(ValueError):
            distribution_metrics(torch.tensor([[float("nan"), 0.]]), 0, THRESHOLDS)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
