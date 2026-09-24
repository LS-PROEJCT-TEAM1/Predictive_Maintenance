from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference import predict_records  # noqa: E402


class InferenceTests(unittest.TestCase):
    def test_template_predicts_known_part(self) -> None:
        records = pd.read_csv(ROOT / "outputs" / "dashboard_data" / "inference_input_template.csv")
        result = predict_records(records)
        self.assertTrue(result["known_part"])
        self.assertEqual(result["recommended_model"], "CatBoost")
        self.assertGreaterEqual(result["recommended_forecast"], 0)
        self.assertEqual(
            pd.Timestamp(result["target_date"]),
            pd.Timestamp(result["origin_date"]) + pd.Timedelta(days=3),
        )

    def test_unknown_part_uses_fallback(self) -> None:
        records = pd.read_csv(ROOT / "outputs" / "dashboard_data" / "inference_input_template.csv")
        records["part_number"] = "New Part"
        result = predict_records(records)
        self.assertFalse(result["known_part"])
        self.assertEqual(result["recommended_model"], "3-day Moving Average")
        self.assertIsNone(result["catboost_prediction"])

    def test_nonconsecutive_dates_are_rejected(self) -> None:
        records = pd.read_csv(ROOT / "outputs" / "dashboard_data" / "inference_input_template.csv")
        records.loc[2, "date"] = "2022-01-10"
        with self.assertRaisesRegex(ValueError, "consecutive"):
            predict_records(records)


class FirestoreOutputTests(unittest.TestCase):
    def test_manifest_counts_and_ids(self) -> None:
        directory = ROOT / "outputs" / "dashboard_data" / "firestore"
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for item in manifest["collections"]:
            lines = [
                line
                for line in (directory / item["file"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            documents = [json.loads(line) for line in lines]
            self.assertEqual(len(documents), item["document_count"])
            ids = [document[item["document_id_field"]] for document in documents]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertFalse(any("NaN" in line or "Infinity" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
