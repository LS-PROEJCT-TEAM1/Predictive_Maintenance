from __future__ import annotations

import json
import unittest
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "final_track_b"


class FinalTrackBOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metrics = pd.read_csv(OUTPUT / "metrics.csv")
        cls.predictions = pd.read_csv(OUTPUT / "predictions.csv")
        cls.splits = pd.read_csv(OUTPUT / "split_manifest.csv")
        cls.manifest = json.loads((OUTPUT / "run_manifest.json").read_text(encoding="utf-8"))

    def test_group_splits_do_not_overlap(self) -> None:
        counts = self.splits.groupby("group_id")["split"].nunique()
        self.assertTrue((counts == 1).all())

    def test_validation_and_test_have_both_classes(self) -> None:
        for split in ["validation", "test"]:
            labels = self.predictions[self.predictions["split"].eq(split)]["label"].unique()
            self.assertEqual(set(labels.tolist()), {0, 1})

    def test_models_use_identical_test_rows(self) -> None:
        test = self.predictions[self.predictions["split"].eq("test")]
        expected = None
        for _, frame in test.groupby("model"):
            keys = set(zip(frame["source_file"], frame["source_row"]))
            expected = keys if expected is None else expected
            self.assertEqual(keys, expected)

    def test_confusion_counts_match_test_rows(self) -> None:
        test_metrics = self.metrics[self.metrics["split"].eq("test")]
        for row in test_metrics.itertuples():
            self.assertEqual(row.tn + row.fp + row.fn + row.tp, row.rows)

    def test_saved_models_are_loadable(self) -> None:
        for model_name in self.metrics["model"].unique():
            payload = joblib.load(OUTPUT / "models" / f"{model_name}.joblib")
            self.assertEqual(payload["name"], model_name)
            self.assertIn("threshold", payload)

    def test_winner_was_selected_from_validation_models(self) -> None:
        winner = self.manifest["winner_selected_on_validation"]
        validation_models = set(self.metrics.loc[self.metrics["split"].eq("validation"), "model"])
        self.assertIn(winner, validation_models)

    def test_required_reports_exist(self) -> None:
        for name in [
            "final_validation_report.md",
            "data_quality_report.md",
            "classification_reports.json",
            "feature_importance.csv",
        ]:
            self.assertTrue((OUTPUT / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
