"""Tests for ExperimentConfig validation."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ExperimentConfig


class DeviceConfigTests(unittest.TestCase):
    def test_accepts_cuda_index(self):
        config = ExperimentConfig(device="cuda:1")
        self.assertEqual(config.device, "cuda:1")

    def test_rejects_bad_device(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(device="tpu")

    def test_rejects_cuda_out_of_range_label(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(device="cuda:x")

    def test_rejects_bad_laya_mode(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(laya_mode="chaotic")

    def test_rejects_bad_rows(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(n_rows=0)

    def test_select_k_within_pool_k(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(pool_k=5, select_k=6)

    def test_output_directory_defaults_under_work_dir(self):
        config = ExperimentConfig()
        self.assertTrue(
            str(config.resolved_output_dir).endswith("outputs")
        )


class FusionAlphaConfigTests(unittest.TestCase):
    def test_default_alpha(self):
        self.assertEqual(ExperimentConfig().fusion_alpha, 0.5)

    def test_alpha_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(fusion_alpha=1.5)
        with self.assertRaises(ValueError):
            ExperimentConfig(fusion_alpha=-0.1)

    def test_alpha_none_disables_fusion(self):
        self.assertIsNone(ExperimentConfig(fusion_alpha=None).fusion_alpha)


if __name__ == "__main__":
    unittest.main()
