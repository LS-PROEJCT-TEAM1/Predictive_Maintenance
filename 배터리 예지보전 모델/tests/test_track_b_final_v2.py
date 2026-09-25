from __future__ import annotations

import json
import unittest
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "track_b_final_v2"


class TrackBFinalV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metrics = pd.read_csv(OUTPUT / "metrics.csv")
        cls.predictions = pd.read_csv(OUTPUT / "predictions.csv")
        cls.splits = pd.read_csv(OUTPUT / "split_manifest.csv")
        cls.manifest = json.loads((OUTPUT / "run_manifest.json").read_text(encoding="utf-8"))

    def test_locked_files_are_independent(self) -> None:
        self.assertEqual(
            set(self.manifest["development_files"]),
            {"WeldingTest_01_OK", "WeldingTest_03_NG"},
        )
        self.assertEqual(
            set(self.manifest["locked_test_files"]),
            {"WeldingTest_02_OK", "WeldingTest_04_NG"},
        )
        self.assertFalse(
            set(self.manifest["development_files"])
            & set(self.manifest["locked_test_files"])
        )
        self.assertFalse(self.manifest["locked_test_used_for_selection"])

    def test_cycle_groups_do_not_overlap_development_splits(self) -> None:
        counts = self.splits.groupby("group_id")["split"].nunique()
        self.assertTrue((counts == 1).all())

    def test_all_models_use_identical_locked_rows(self) -> None:
        locked = self.predictions[self.predictions["split"].eq("locked_test")]
        expected = None
        for _, frame in locked.groupby("model"):
            keys = set(zip(frame["source_file"], frame["source_row"]))
            if expected is None:
                expected = keys
            self.assertEqual(keys, expected)

    def test_model_families_and_counts(self) -> None:
        unique = self.metrics.drop_duplicates("model")
        self.assertGreaterEqual((unique["family"] == "supervised").sum(), 2)
        self.assertGreaterEqual((unique["family"] == "unsupervised").sum(), 1)

    def test_confusion_counts_match_rows(self) -> None:
        for row in self.metrics.itertuples():
            self.assertEqual(row.tn + row.fp + row.fn + row.tp, row.rows)

    def test_locked_event_is_not_split_by_cycle(self) -> None:
        locked = self.metrics[self.metrics["split"].eq("locked_test")]
        self.assertTrue((locked["event_total"] == 1).all())

    def test_unsupervised_thresholds_use_only_normal_calibration(self) -> None:
        unique = self.metrics.drop_duplicates("model")
        unsupervised = unique[unique["family"].eq("unsupervised")]
        self.assertTrue(
            unsupervised["threshold_source"].str.startswith("historical_normal_calibration").all()
        )

    def test_saved_models_are_loadable(self) -> None:
        for model_name in self.metrics["model"].unique():
            payload = joblib.load(OUTPUT / "models" / f"{model_name}.joblib")
            self.assertEqual(payload["name"], model_name)
            self.assertEqual(payload["pipeline_version"], "track-b-final-v2")

    def test_required_outputs_exist(self) -> None:
        for name in [
            "training_plan.md",
            "final_validation_report.md",
            "per_file_metrics.csv",
            "reverse_file_stress_metrics.csv",
            "classification_reports.json",
            "feature_importance.csv",
            "data_quality.csv",
            "data_quality_report.md",
        ]:
            self.assertTrue((OUTPUT / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
