from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "final_v3"
REQUIRED_COLUMNS = ["part_number", "date", "actual_d", "plan_d3", "plan_d4", "plan_d5"]
SEQUENCE_FEATURES = ["actual_d", "plan_d3", "plan_d4", "plan_d5"]


@lru_cache(maxsize=1)
def load_metadata() -> dict[str, object]:
    return json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_catboost() -> CatBoostRegressor:
    model = CatBoostRegressor()
    model_path = MODEL_DIR / "catboost.cbm"
    try:
        model.load_model(str(model_path))
    except Exception:
        with tempfile.TemporaryDirectory() as directory:
            temporary_path = Path(directory) / "catboost.cbm"
            shutil.copy2(model_path, temporary_path)
            model.load_model(str(temporary_path))
    return model


def validate_records(records: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in records.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    clean = records[REQUIRED_COLUMNS].copy()
    if len(clean) != 3:
        raise ValueError("Inference input must contain exactly three daily rows.")
    if clean["part_number"].astype(str).nunique() != 1:
        raise ValueError("All three rows must use the same part_number.")
    clean["part_number"] = clean["part_number"].astype(str)
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce").dt.normalize()
    for column in SEQUENCE_FEATURES:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    if clean[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Input contains a missing or invalid value.")
    if (clean[SEQUENCE_FEATURES] < 0).any().any():
        raise ValueError("Demand and plan quantities must be zero or positive.")
    clean = clean.sort_values("date").reset_index(drop=True)
    if not np.all(clean["date"].diff().dropna().dt.days.to_numpy() == 1):
        raise ValueError("The three input dates must be consecutive calendar days.")
    return clean


def build_feature_row(records: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    clean = validate_records(records)
    metadata = load_metadata()
    model_metadata = metadata["model"]
    feature_columns = list(model_metadata["feature_columns"])
    origin_date = pd.Timestamp(clean.iloc[-1]["date"])
    source_start = pd.Timestamp(metadata["quality"]["source_start"])
    day_of_week = origin_date.dayofweek
    values: dict[str, object] = {
        "part_number": str(clean.iloc[-1]["part_number"]),
        "dow_sin": float(np.sin(2 * np.pi * day_of_week / 7)),
        "dow_cos": float(np.cos(2 * np.pi * day_of_week / 7)),
        "is_weekend": int(day_of_week >= 5),
        "month": int(origin_date.month),
        "days_since_start": int((origin_date - source_start).days),
    }
    for row_index, suffix in enumerate(["lag2", "lag1", "origin"]):
        for feature in SEQUENCE_FEATURES:
            values[f"{feature}_{suffix}"] = float(clean.iloc[row_index][feature])
    return pd.DataFrame([values], columns=feature_columns), {
        "part_number": values["part_number"],
        "origin_date": origin_date,
        "target_date": origin_date + pd.Timedelta(days=3),
        "moving_average_3d": float(clean["actual_d"].mean()),
        "plan_d3_reference": float(clean.iloc[-1]["plan_d3"]),
    }


def predict_records(records: pd.DataFrame) -> dict[str, object]:
    feature_row, context = build_feature_row(records)
    metadata = load_metadata()
    known_parts = set(metadata["model"]["part_to_id"])
    is_known_part = context["part_number"] in known_parts
    moving_average = max(0.0, float(context["moving_average_3d"]))

    catboost_prediction: float | None = None
    if is_known_part:
        catboost_prediction = max(0.0, float(load_catboost().predict(feature_row)[0]))

    recommended_model = "CatBoost" if is_known_part else "3-day Moving Average"
    recommended_forecast = catboost_prediction if is_known_part else moving_average
    return {
        "part_number": context["part_number"],
        "origin_date": context["origin_date"].date().isoformat(),
        "target_date": context["target_date"].date().isoformat(),
        "known_part": is_known_part,
        "catboost_prediction": catboost_prediction,
        "moving_average_3d": moving_average,
        "plan_d3_reference": context["plan_d3_reference"],
        "recommended_model": recommended_model,
        "recommended_forecast": recommended_forecast,
        "model_version": "final_v3",
        "fallback_reason": None if is_known_part else "part_not_seen_in_training",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict D+3 demand from three daily rows.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = predict_records(pd.read_csv(args.input_csv))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
