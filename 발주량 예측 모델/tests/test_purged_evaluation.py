from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "purged_evaluation"
MODELS = ROOT / "models" / "purged_v1"
SCENARIOS = ("source_total", "slot_sum")
PREDICTION_COLUMNS = (
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "LSTM",
    "3-day Moving Average",
    "D+3 Plan Reference",
)


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    denominator = float(np.abs(actual).sum())
    total_sum_of_squares = float(np.square(actual - actual.mean()).sum())
    return {
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.square(error).mean())),
        "WAPE_pct": float(np.abs(error).sum() / denominator * 100),
        "Forecast_Bias": float(error.mean()),
        "R2": float(1 - np.square(error).sum() / total_sum_of_squares),
    }


class PurgedBoundaryTests(unittest.TestCase):
    def test_every_split_has_strict_three_day_purge(self) -> None:
        audit = pd.read_csv(OUTPUT / "all_split_audit.csv")
        self.assertEqual(set(audit["Scenario"]), set(SCENARIOS))
        self.assertEqual(len(audit), 8)
        for row in audit.itertuples(index=False):
            self.assertLess(
                pd.Timestamp(row.Train_Target_End),
                pd.Timestamp(row.Validation_Origin_Start),
            )
            self.assertLess(
                pd.Timestamp(row.Validation_Target_End),
                pd.Timestamp(row.Test_Origin_Start),
            )

    def test_data_coverage_and_zero_values_are_preserved(self) -> None:
        for scenario in SCENARIOS:
            metadata = json.loads(
                (OUTPUT / scenario / "run_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["quality"]["source_parts"], 117)
            self.assertEqual(metadata["sequence"]["eligible_parts"], 116)
            self.assertEqual(metadata["sequence"]["excluded_parts"], ["Part 115"])
            self.assertGreater(metadata["sequence"]["plan_zero_rows"], 0)
            self.assertGreater(metadata["sequence"]["target_zero_rows"], 0)


class SavedMetricTests(unittest.TestCase):
    def test_saved_metrics_recompute_from_predictions(self) -> None:
        for scenario in SCENARIOS:
            for stem in ("cv", "final_holdout"):
                predictions = pd.read_csv(
                    OUTPUT / scenario / f"{stem}_predictions.csv"
                )
                metrics_file = (
                    "cv_metrics_pooled.csv"
                    if stem == "cv"
                    else "final_holdout_metrics.csv"
                )
                saved = pd.read_csv(OUTPUT / scenario / metrics_file).set_index("Model")
                actual = predictions["actual"].to_numpy(dtype=float)
                self.assertFalse(predictions[list(PREDICTION_COLUMNS)].isna().any().any())
                for model in PREDICTION_COLUMNS:
                    recalculated = calculate_metrics(
                        actual, predictions[model].to_numpy(dtype=float)
                    )
                    self.assertEqual(int(saved.loc[model, "N"]), len(predictions))
                    for metric, value in recalculated.items():
                        self.assertTrue(
                            math.isclose(
                                float(saved.loc[model, metric]),
                                value,
                                rel_tol=1e-7,
                                abs_tol=1e-6,
                            ),
                            msg=f"{scenario}/{stem}/{model}/{metric}",
                        )

    def test_selection_uses_cv_not_final_holdout(self) -> None:
        summary = pd.read_csv(OUTPUT / "selection_summary.csv")
        self.assertTrue((summary["CV_Overall_Winner"] == "3-day Moving Average").all())
        self.assertTrue((summary["CV_ML_Winner"] == "XGBoost").all())


class SavedModelTests(unittest.TestCase):
    def test_tree_models_reload_and_lightgbm_uses_lf(self) -> None:
        for scenario in SCENARIOS:
            source = MODELS / scenario
            lightgbm_bytes = (source / "lightgbm.txt").read_bytes()
            self.assertNotIn(b"\r\n", lightgbm_bytes)
            with tempfile.TemporaryDirectory() as directory:
                temporary = Path(directory)
                for filename in ("xgboost.json", "catboost.cbm", "lightgbm.txt"):
                    shutil.copy2(source / filename, temporary / filename)
                xgboost = XGBRegressor()
                xgboost.load_model(temporary / "xgboost.json")
                catboost = CatBoostRegressor()
                catboost.load_model(str(temporary / "catboost.cbm"))
                lightgbm = lgb.Booster(model_file=str(temporary / "lightgbm.txt"))
                self.assertGreater(xgboost.get_booster().num_boosted_rounds(), 0)
                self.assertGreater(catboost.tree_count_, 0)
                self.assertGreater(lightgbm.num_trees(), 0)


if __name__ == "__main__":
    unittest.main()
