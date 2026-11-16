"""Policy unit tests run before any final-test prediction."""
import sys
import tempfile
from unittest.mock import patch
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import accepted_mask, metric_from_cm, read_predictions, select_threshold, threshold_curve
from analysis import group_bootstrap


class ReferralTests(unittest.TestCase):
    def test_all_correct_accept_all(self):
        p, _ = select_threshold([.4, .7, .9], [True]*3)
        self.assertEqual(p["threshold"], 0.0)
        self.assertEqual(p["accepted"], 3)

    def test_all_wrong_refer_all(self):
        p, _ = select_threshold([.4, .9], [False, False])
        self.assertEqual(p["status"], "refer_all")
        self.assertIsNone(p["accepted_error"])
        self.assertFalse(accepted_mask([.4, .9], p).any())

    def test_complete_mixed_tie_not_partial(self):
        p, _ = select_threshold([.8, .8], [True, False])
        self.assertEqual(p["accepted"], 0)

    def test_exact_five_percent(self):
        p, _ = select_threshold([.8]*20, [False]+[True]*19)
        self.assertEqual(p["accepted"], 20)
        p19, _ = select_threshold([.8]*19, [False]+[True]*18)
        self.assertEqual(p19["accepted"], 0)

    def test_exhaustive_nonmonotonic_scan(self):
        # Descending scores: 1/20 errors, then 2/21, then 2/40.
        scores = np.r_[np.full(20, .9), .8, np.full(19, .7)]
        correct = np.r_[False, np.ones(19, dtype=bool), False, np.ones(19, dtype=bool)]
        p, c = select_threshold(scores, correct)
        self.assertEqual(p["accepted"], 40)
        self.assertFalse(bool(c.loc[c.threshold == .8, "feasible"].iloc[0]))

    def test_row_order_invariant(self):
        scores = np.r_[.5, np.full(19, .9)]
        correct = np.r_[False, np.ones(19, dtype=bool)]
        p, _ = select_threshold(scores, correct)
        q, _ = select_threshold(scores[::-1], correct[::-1])
        self.assertEqual(p, q)

    def test_full_precision_boundary(self):
        p = {"status": "threshold", "threshold": .9000000000001}
        self.assertEqual(accepted_mask([.9, .9000000000001], p).tolist(), [False, True])

    def test_invalid_scores_rejected(self):
        with self.assertRaises(ValueError):
            threshold_curve([np.nan], [True])

    def test_csv_round_trip_at_boundary(self):
        threshold = np.nextafter(.9000000000001, np.inf)
        scores = np.array([np.nextafter(threshold, 0), threshold, np.nextafter(threshold, 1)])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.csv"
            pd.DataFrame({"score": scores}).to_csv(path, index=False, float_format="%.17g")
            loaded = read_predictions(path).score.to_numpy()
        self.assertTrue(np.array_equal(scores, loaded))
        self.assertEqual(accepted_mask(loaded, {"status": "threshold", "threshold": threshold}).tolist(),
                         [False, True, True])

    def test_zero_class_not_dropped(self):
        cm = np.zeros((7, 7), dtype=int)
        cm[0, 0] = 20
        m = metric_from_cm(cm)
        self.assertAlmostEqual(m["macro_f1"], 1/7)

    def test_all_referred_bootstrap_is_undefined_not_zero(self):
        table = pd.DataFrame({"feature_group": [0, 0, 1], "actual": ["CALI"]*3,
                              "predicted": ["SIRA"]*3, "baseline_predicted": ["SIRA"]*3,
                              "accepted": [False]*3})
        with tempfile.TemporaryDirectory() as tmp, patch("analysis.RESULTS", Path(tmp)):
            result = group_bootstrap(table, replicates=20)
        interval = result["intervals"]["accepted_error"]
        self.assertIsNone(interval["lower"])
        self.assertIsNone(interval["upper"])
        self.assertEqual(interval["undefined_replicates"], 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
