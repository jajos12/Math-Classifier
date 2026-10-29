"""Tests for dataset helpers (no network, no parquet writer needed)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import pool_row_key, state_fingerprint

try:
    import pandas as pd

    HAVE_PANDAS = True
except ImportError:
    HAVE_PANDAS = False


class PoolRowKeyTests(unittest.TestCase):
    def test_integer(self):
        self.assertEqual(pool_row_key(123), "123")

    def test_float_rendered(self):
        self.assertEqual(pool_row_key("123.0"), "123")
        self.assertEqual(pool_row_key(123.0), "123")

    def test_whitespace_stripped(self):
        self.assertEqual(pool_row_key(" 123 "), "123")

    def test_trailing_zero_of_wider_number_kept(self):
        self.assertEqual(pool_row_key("123.0.0"), "123.0.0")

    def test_missing_values_empty(self):
        self.assertEqual(pool_row_key(None), "")
        self.assertEqual(pool_row_key(float("nan")), "")
        self.assertEqual(pool_row_key(""), "")
        self.assertEqual(pool_row_key("nan"), "")

    def test_negative_float_rendered(self):
        self.assertEqual(pool_row_key("-5.0"), "-5")


class StateFingerprintTests(unittest.TestCase):
    def test_stable_content_hash(self):
        self.assertEqual(
            state_fingerprint("h : a = b\n⊢ a = b"),
            state_fingerprint("h : a = b\n⊢ a = b"),
        )

    def test_whitespace_sensitive(self):
        self.assertNotEqual(state_fingerprint("a b"), state_fingerprint("a  b"))

    def test_missing_values_empty(self):
        self.assertEqual(state_fingerprint(None), "")
        self.assertEqual(state_fingerprint(float("nan")), "")
        self.assertEqual(state_fingerprint("  "), "")


@unittest.skipUnless(HAVE_PANDAS, "pandas is required for sampling tests")
class SampleDatasetTests(unittest.TestCase):
    def _frame(self, rows):
        return pd.DataFrame({"row_index": range(rows), "value": [f"v{i}" for i in range(rows)]})

    def test_oversampled_frame_returns_all_rows(self):
        frame = self._frame(5)
        result = sample_dataset(frame, 100, seed=42)
        self.assertEqual(len(result), 5)

    def test_sample_size_and_determinism(self):
        frame = self._frame(50)
        first = sample_dataset(frame, 10, seed=42).sort_index()
        second = sample_dataset(frame, 10, seed=42).sort_index()
        self.assertEqual(len(first), 10)
        self.assertEqual(list(first["value"]), list(second["value"]))

    def test_invalid_rows_raise(self):
        with self.assertRaises(ValueError):
            sample_dataset(self._frame(5), 0, seed=42)


from data import sample_dataset  # noqa: E402  (kept after the pandas guard)


if __name__ == "__main__":
    unittest.main()
