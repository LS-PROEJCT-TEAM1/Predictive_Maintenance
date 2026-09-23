from __future__ import annotations

import json
import math
import os
import platform
import random
import shutil
import tempfile
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import xgboost
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


SEED = 42
HORIZON_DAYS = 3
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Dataset_공급망 최적화 AI 데이터셋" / "data" / "data.xls"
OUTPUT_DIR = ROOT / "outputs" / "tree_model_evaluation"
MODEL_DIR = ROOT / "models" / "tree_v2"

REQUESTED_MODELS = [
    "XGBoost",
    "3-day Moving Average",
    "LightGBM",
    "CatBoost",
]
REFERENCE_MODEL = "D+3 Plan Reference"
FOLD_FRACTIONS = [
    (0.20, 0.40),
    (0.40, 0.55),
    (0.55, 0.70),
    (0.70, 0.85),
    (0.85, 1.00),
]


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)


def exact_lag(frame: pd.DataFrame, value_col: str, days: int) -> np.ndarray:
    lookup = frame.set_index(["part_number", "date"])[value_col]
    keys = pd.MultiIndex.from_arrays(
        [frame["part_number"], frame["date"] - pd.Timedelta(days=days)],
        names=["part_number", "date"],
    )
    return lookup.reindex(keys).to_numpy()


def load_and_prepare() -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, object]]:
    raw = pd.read_excel(DATA_PATH)
    source_rows, source_columns = raw.shape

    # The legacy XLS labels are damaged by encoding in some readers. These positions
    # follow the supplied guidebook and the previously verified project mapping.
    selected = pd.DataFrame(
        {
            "part_number": raw.iloc[:, 0],
            "actual_d": pd.to_numeric(raw.iloc[:, 11], errors="coerce"),
            "plan_d3": pd.to_numeric(raw.iloc[:, 44], errors="coerce"),
            "plan_d4": pd.to_numeric(raw.iloc[:, 55], errors="coerce"),
            "plan_d5": pd.to_numeric(raw.iloc[:, 56], errors="coerce"),
            "timestamp": pd.to_datetime(
                raw.iloc[:, 83].astype(str), format="%Y%m%d%H%M", errors="coerce"
            ),
        }
    )
    base_required = ["part_number", "actual_d", "plan_d3", "plan_d4", "plan_d5", "timestamp"]
    selected = selected.dropna(subset=base_required).copy()
    selected["part_number"] = selected["part_number"].astype(str)
    selected["date"] = selected["timestamp"].dt.normalize()
    selected = selected.sort_values(["part_number", "timestamp"])

    # One row per part and calendar day: keep the final intraday log.
    daily = (
        selected.groupby(["part_number", "date"], as_index=False, sort=False)
        .tail(1)
        .sort_values(["part_number", "date"])
        .reset_index(drop=True)
    )

    daily["plan_mean"] = daily[["plan_d3", "plan_d4", "plan_d5"]].mean(axis=1)
    daily["plan_max"] = daily[["plan_d3", "plan_d4", "plan_d5"]].max(axis=1)
    daily["plan_min"] = daily[["plan_d3", "plan_d4", "plan_d5"]].min(axis=1)
    daily["plan_range"] = daily["plan_max"] - daily["plan_min"]
    daily["plan_trend_d3_to_d5"] = daily["plan_d3"] - daily["plan_d5"]

    # Same-delivery-date plan revisions. Missing prior logs remain missing; negative
    # revisions are valid and are never subjected to a nonnegative quantity filter.
    daily["plan_d4_lag1"] = exact_lag(daily, "plan_d4", 1)
    daily["plan_d5_lag2"] = exact_lag(daily, "plan_d5", 2)
    daily["plan_revision_1d"] = daily["plan_d3"] - daily["plan_d4_lag1"]
    daily["plan_revision_2d"] = daily["plan_d4_lag1"] - daily["plan_d5_lag2"]
    daily["plan_revision_rate"] = daily["plan_revision_1d"] / (
        daily["plan_d4_lag1"].abs() + 1.0
    )

    for lag in range(1, 8):
        daily[f"actual_lag{lag}"] = exact_lag(daily, "actual_d", lag)

    lag3 = [f"actual_lag{i}" for i in range(1, 4)]
    lag7 = [f"actual_lag{i}" for i in range(1, 8)]
    daily["actual_roll3_count"] = daily[lag3].notna().sum(axis=1)
    daily["actual_roll3_mean"] = daily[lag3].mean(axis=1, skipna=True)
    daily["actual_roll3_std"] = daily[lag3].std(axis=1, skipna=True, ddof=0)
    daily["actual_roll7_count"] = daily[lag7].notna().sum(axis=1)
    daily["actual_roll7_mean"] = daily[lag7].mean(axis=1, skipna=True)
    daily["actual_roll7_std"] = daily[lag7].std(axis=1, skipna=True, ddof=0)
    daily["recent_nonzero_rate_7"] = daily[lag7].gt(0).sum(axis=1) / daily[
        "actual_roll7_count"
    ].replace(0, np.nan)

    missing_source_features = [
        "plan_d4_lag1",
        "plan_d5_lag2",
        "plan_revision_1d",
        "plan_revision_2d",
        "plan_revision_rate",
        "actual_lag1",
        "actual_lag2",
        "actual_lag3",
        "actual_lag7",
        "actual_roll3_mean",
        "actual_roll3_std",
        "actual_roll7_mean",
        "actual_roll7_std",
        "recent_nonzero_rate_7",
    ]
    for col in missing_source_features:
        daily[f"{col}_missing"] = daily[col].isna().astype(int)

    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["dow_sin"] = np.sin(2 * np.pi * daily["day_of_week"] / 7)
    daily["dow_cos"] = np.cos(2 * np.pi * daily["day_of_week"] / 7)
    daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)
    daily["month"] = daily["date"].dt.month
    daily["week_of_year"] = daily["date"].dt.isocalendar().week.astype(int)
    daily["days_since_global_start"] = (daily["date"] - daily["date"].min()).dt.days
    daily["days_since_part_start"] = (
        daily["date"] - daily.groupby("part_number")["date"].transform("min")
    ).dt.days
    daily["prior_observation_count"] = daily.groupby("part_number").cumcount()

    daily["target_date"] = daily["date"] + pd.Timedelta(days=HORIZON_DAYS)
    target_lookup = daily.set_index(["part_number", "date"])["actual_d"]
    target_keys = pd.MultiIndex.from_arrays(
        [daily["part_number"], daily["target_date"]], names=["part_number", "date"]
    )
    daily["target"] = target_lookup.reindex(target_keys).to_numpy()

    numeric_features = [
        "plan_d3",
        "plan_d4",
        "plan_d5",
        "plan_mean",
        "plan_max",
        "plan_min",
        "plan_range",
        "plan_trend_d3_to_d5",
        "plan_revision_1d",
        "plan_revision_2d",
        "plan_revision_rate",
        "actual_lag1",
        "actual_lag2",
        "actual_lag3",
        "actual_lag7",
        "actual_roll3_mean",
        "actual_roll3_std",
        "actual_roll3_count",
        "actual_roll7_mean",
        "actual_roll7_std",
        "actual_roll7_count",
        "recent_nonzero_rate_7",
        "dow_sin",
        "dow_cos",
        "is_weekend",
        "month",
        "week_of_year",
        "days_since_global_start",
        "days_since_part_start",
        "prior_observation_count",
        *[f"{col}_missing" for col in missing_source_features],
    ]

    eligible = daily.dropna(subset=["target"]).copy()
    quantity_columns = ["actual_d", "plan_d3", "plan_d4", "plan_d5", "target"]
    invalid_quantity_mask = (eligible[quantity_columns] < 0).any(axis=1)
    invalid_quantity_rows = int(invalid_quantity_mask.sum())
    supervised = eligible.loc[~invalid_quantity_mask].copy()
    supervised = supervised.sort_values(["target_date", "part_number", "date"]).reset_index(
        drop=True
    )

    # The baseline averages available exact calendar-day lags. A part with no prior
    # observation falls back to its valid D+3 plan. A plan value of zero stays zero.
    supervised["moving_average_3d"] = supervised[lag3].mean(axis=1, skipna=True)
    supervised["moving_average_fallback_used"] = supervised["moving_average_3d"].isna()
    supervised["moving_average_3d"] = supervised["moving_average_3d"].fillna(
        supervised["plan_d3"]
    )

    qa = {
        "source_rows": int(source_rows),
        "source_columns": int(source_columns),
        "source_parts": int(raw.iloc[:, 0].nunique(dropna=True)),
        "source_start": str(selected["date"].min().date()),
        "source_end": str(selected["date"].max().date()),
        "source_missing_cells": int(raw.isna().sum().sum()),
        "source_exact_duplicate_rows": int(raw.duplicated().sum()),
        "daily_rows_after_last_log": int(len(daily)),
        "daily_parts_after_last_log": int(daily["part_number"].nunique()),
        "rows_with_exact_d3_target": int(len(eligible)),
        "invalid_negative_quantity_rows_removed": invalid_quantity_rows,
        "supervised_rows": int(len(supervised)),
        "supervised_parts": int(supervised["part_number"].nunique()),
        "plan_d3_zero_rows": int((supervised["plan_d3"] == 0).sum()),
        "plan_d3_zero_rate": float((supervised["plan_d3"] == 0).mean()),
        "target_zero_rows": int((supervised["target"] == 0).sum()),
        "target_zero_rate": float((supervised["target"] == 0).mean()),
        "moving_average_fallback_rows": int(
            supervised["moving_average_fallback_used"].sum()
        ),
    }
    return daily, supervised, numeric_features, qa


def make_folds(frame: pd.DataFrame) -> list[dict[str, object]]:
    dates = np.array(sorted(frame["target_date"].unique()))
    folds: list[dict[str, object]] = []
    for fold_number, (train_fraction, test_fraction) in enumerate(FOLD_FRACTIONS, start=1):
        train_end_index = max(1, int(len(dates) * train_fraction))
        test_end_index = min(
            len(dates), max(train_end_index + 1, int(len(dates) * test_fraction))
        )
        train_end = pd.Timestamp(dates[train_end_index - 1])
        test_end = pd.Timestamp(dates[test_end_index - 1])
        train_mask = frame["target_date"] <= train_end
        test_mask = (frame["target_date"] > train_end) & (frame["target_date"] <= test_end)
        folds.append(
            {
                "fold": fold_number,
                "train_end": train_end,
                "test_end": test_end,
                "train_mask": train_mask.to_numpy(),
                "test_mask": test_mask.to_numpy(),
            }
        )
    return folds


def make_internal_validation_masks(train_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    dates = np.array(sorted(train_frame["target_date"].unique()))
    validation_date_count = max(2, int(math.ceil(len(dates) * 0.20)))
    validation_date_count = min(validation_date_count, len(dates) - 1)
    fit_end = pd.Timestamp(dates[-validation_date_count - 1])
    fit_mask = (train_frame["target_date"] <= fit_end).to_numpy()
    validation_mask = ~fit_mask
    return fit_mask, validation_mask


def make_tree_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "part",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                ["part_number"],
            ),
            ("numeric", "passthrough", numeric_features),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )


def make_xgboost(n_estimators: int, early_stopping_rounds: int | None = None) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:absoluteerror",
        eval_metric="mae",
        n_estimators=max(1, int(n_estimators)),
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=5.0,
        random_state=SEED,
        n_jobs=4,
        tree_method="hist",
        early_stopping_rounds=early_stopping_rounds,
    )


def make_lightgbm(n_estimators: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l1",
        n_estimators=max(1, int(n_estimators)),
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=5.0,
        random_state=SEED,
        n_jobs=4,
        verbosity=-1,
    )


def make_catboost(iterations: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=max(1, int(iterations)),
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5.0,
        random_seed=SEED,
        thread_count=4,
        allow_writing_files=False,
        verbose=False,
    )


def fit_predict_fold(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    numeric_features: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    feature_columns = ["part_number"] + numeric_features
    fit_mask, validation_mask = make_internal_validation_masks(train_frame)
    fit_frame = train_frame.loc[fit_mask]
    validation_frame = train_frame.loc[validation_mask]

    # Tune tree count on the last part of the training window only.
    tuning_preprocessor = make_tree_preprocessor(numeric_features)
    x_fit = tuning_preprocessor.fit_transform(fit_frame[feature_columns])
    x_validation = tuning_preprocessor.transform(validation_frame[feature_columns])
    y_fit = fit_frame["target"].to_numpy(dtype=float)
    y_validation = validation_frame["target"].to_numpy(dtype=float)

    xgb_tuning = make_xgboost(1500, early_stopping_rounds=60)
    xgb_tuning.fit(
        x_fit,
        y_fit,
        eval_set=[(x_validation, y_validation)],
        verbose=False,
    )
    xgb_iterations = int(xgb_tuning.best_iteration + 1)

    lgb_tuning = make_lightgbm(1500)
    lgb_tuning.fit(
        x_fit,
        y_fit,
        eval_set=[(x_validation, y_validation)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(60, verbose=False)],
    )
    lgb_iterations = int(lgb_tuning.best_iteration_)

    cat_fit = fit_frame[feature_columns].copy()
    cat_validation = validation_frame[feature_columns].copy()
    cat_tuning = make_catboost(1500)
    cat_tuning.fit(
        cat_fit,
        y_fit,
        cat_features=["part_number"],
        eval_set=(cat_validation, y_validation),
        early_stopping_rounds=60,
        verbose=False,
    )
    cat_iterations = max(1, int(cat_tuning.get_best_iteration() + 1))

    # Refit with the selected tree count on the complete fold training window.
    final_preprocessor = make_tree_preprocessor(numeric_features)
    x_train = final_preprocessor.fit_transform(train_frame[feature_columns])
    x_test = final_preprocessor.transform(test_frame[feature_columns])
    y_train = train_frame["target"].to_numpy(dtype=float)

    xgb_model = make_xgboost(xgb_iterations)
    xgb_model.fit(x_train, y_train, verbose=False)

    lgb_model = make_lightgbm(lgb_iterations)
    lgb_model.fit(x_train, y_train)

    cat_model = make_catboost(cat_iterations)
    cat_model.fit(
        train_frame[feature_columns],
        y_train,
        cat_features=["part_number"],
        verbose=False,
    )

    predictions = {
        "XGBoost": np.clip(xgb_model.predict(x_test), 0, None),
        "3-day Moving Average": test_frame["moving_average_3d"].to_numpy(dtype=float),
        "LightGBM": np.clip(lgb_model.predict(x_test), 0, None),
        "CatBoost": np.clip(cat_model.predict(test_frame[feature_columns]), 0, None),
        REFERENCE_MODEL: test_frame["plan_d3"].to_numpy(dtype=float),
    }
    iterations = {
        "xgboost_trees": xgb_iterations,
        "lightgbm_trees": lgb_iterations,
        "catboost_trees": cat_iterations,
    }
    return predictions, iterations


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = float(np.abs(y_true).sum())
    variance = float(np.var(y_true))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 and variance > 0 else np.nan
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "WAPE_pct": (
            float(np.abs(y_true - y_pred).sum() / denominator * 100)
            if denominator > 0
            else np.nan
        ),
        "Forecast_Bias": float(np.mean(y_pred - y_true)),
        "R2": r2,
    }


def metrics_from_prediction_frame(
    predictions: pd.DataFrame,
    model_names: list[str],
    grouping: dict[str, object] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    y_true = predictions["actual"].to_numpy(dtype=float)
    for model_name in model_names:
        row: dict[str, object] = {
            "Model": model_name,
            "N": int(len(predictions)),
            "Parts": int(predictions["part_number"].nunique()),
            **calculate_metrics(y_true, predictions[model_name].to_numpy(dtype=float)),
        }
        if grouping:
            row = {**grouping, **row}
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_subsets(predictions: pd.DataFrame) -> pd.DataFrame:
    subset_masks = {
        "All pooled walk-forward rows": np.ones(len(predictions), dtype=bool),
        "D+3 plan = 0": predictions["plan_d3"].eq(0).to_numpy(),
        "D+3 plan > 0": predictions["plan_d3"].gt(0).to_numpy(),
        "Actual = 0": predictions["actual"].eq(0).to_numpy(),
        "Actual > 0": predictions["actual"].gt(0).to_numpy(),
        "Warm-start part": (~predictions["is_cold_start"]).to_numpy(),
        "Cold-start part": predictions["is_cold_start"].to_numpy(),
    }
    frames: list[pd.DataFrame] = []
    for subset_name, mask in subset_masks.items():
        subset = predictions.loc[mask]
        if subset.empty:
            continue
        frames.append(
            metrics_from_prediction_frame(
                subset,
                REQUESTED_MODELS,
                grouping={"Subset": subset_name},
            )
        )
    return pd.concat(frames, ignore_index=True)


def evaluate_per_part(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for part_number, group in predictions.groupby("part_number", sort=True):
        for model_name in REQUESTED_MODELS:
            rows.append(
                {
                    "part_number": part_number,
                    "Model": model_name,
                    "N": int(len(group)),
                    **calculate_metrics(
                        group["actual"].to_numpy(dtype=float),
                        group[model_name].to_numpy(dtype=float),
                    ),
                }
            )
    detail = pd.DataFrame(rows)
    macro = (
        detail.groupby("Model", as_index=False)
        .agg(
            Parts=("part_number", "nunique"),
            Per_part_MAE_mean=("MAE", "mean"),
            Per_part_MAE_median=("MAE", "median"),
            Per_part_RMSE_mean=("RMSE", "mean"),
            Per_part_WAPE_pct_mean=("WAPE_pct", "mean"),
            Per_part_Forecast_Bias_mean=("Forecast_Bias", "mean"),
            Per_part_R2_mean=("R2", "mean"),
        )
        .sort_values("Per_part_MAE_mean")
        .reset_index(drop=True)
    )
    return detail, macro


def run_walk_forward(
    frame: pd.DataFrame,
    numeric_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    prediction_frames: list[pd.DataFrame] = []
    fold_metric_frames: list[pd.DataFrame] = []
    fold_coverage_rows: list[dict[str, object]] = []
    iteration_records: list[dict[str, int]] = []

    for fold_info in make_folds(frame):
        fold = int(fold_info["fold"])
        train_frame = frame.loc[fold_info["train_mask"]].copy()
        test_frame = frame.loc[fold_info["test_mask"]].copy()
        train_parts = set(train_frame["part_number"])
        test_parts = set(test_frame["part_number"])
        cold_parts = sorted(test_parts - train_parts)

        predictions, iterations = fit_predict_fold(
            train_frame,
            test_frame,
            numeric_features,
        )
        iteration_records.append({"fold": fold, **iterations})

        prediction_frame = test_frame[
            [
                "part_number",
                "date",
                "target_date",
                "target",
                "plan_d3",
                "moving_average_fallback_used",
            ]
        ].copy()
        prediction_frame = prediction_frame.rename(
            columns={"date": "forecast_origin", "target": "actual"}
        )
        prediction_frame.insert(0, "Fold", fold)
        prediction_frame["is_cold_start"] = prediction_frame["part_number"].isin(cold_parts)
        for model_name, values in predictions.items():
            prediction_frame[model_name] = values
        prediction_frames.append(prediction_frame)

        fold_metrics = metrics_from_prediction_frame(
            prediction_frame,
            REQUESTED_MODELS + [REFERENCE_MODEL],
            grouping={"Fold": fold},
        )
        fold_metrics.insert(1, "Train_Target_End", str(pd.Timestamp(fold_info["train_end"]).date()))
        fold_metrics.insert(
            2, "Test_Target_Start", str(test_frame["target_date"].min().date())
        )
        fold_metrics.insert(3, "Test_Target_End", str(pd.Timestamp(fold_info["test_end"]).date()))
        fold_metric_frames.append(fold_metrics)

        fold_coverage_rows.append(
            {
                "Fold": fold,
                "Train_Target_Start": str(train_frame["target_date"].min().date()),
                "Train_Target_End": str(pd.Timestamp(fold_info["train_end"]).date()),
                "Test_Target_Start": str(test_frame["target_date"].min().date()),
                "Test_Target_End": str(pd.Timestamp(fold_info["test_end"]).date()),
                "Train_Rows": int(len(train_frame)),
                "Train_Parts": int(len(train_parts)),
                "Test_Rows": int(len(test_frame)),
                "Test_Parts": int(len(test_parts)),
                "Cold_Start_Rows": int(prediction_frame["is_cold_start"].sum()),
                "Cold_Start_Parts": int(len(cold_parts)),
                "Cold_Start_Part_List": ", ".join(cold_parts),
                **iterations,
            }
        )
        print(
            f"Fold {fold}/5 complete: train={len(train_frame):,} rows/{len(train_parts)} parts, "
            f"test={len(test_frame):,} rows/{len(test_parts)} parts, cold={len(cold_parts)} parts",
            flush=True,
        )

    pooled_predictions = pd.concat(prediction_frames, ignore_index=True)
    fold_metrics = pd.concat(fold_metric_frames, ignore_index=True)
    fold_coverage = pd.DataFrame(fold_coverage_rows)
    iterations_frame = pd.DataFrame(iteration_records)
    selected_iterations = {
        column: max(1, int(round(iterations_frame[column].median())))
        for column in ["xgboost_trees", "lightgbm_trees", "catboost_trees"]
    }
    return pooled_predictions, fold_metrics, fold_coverage, selected_iterations


def fit_and_save_final_models(
    frame: pd.DataFrame,
    numeric_features: list[str],
    iterations: dict[str, int],
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    feature_columns = ["part_number"] + numeric_features
    y = frame["target"].to_numpy(dtype=float)

    preprocessor = make_tree_preprocessor(numeric_features)
    x = preprocessor.fit_transform(frame[feature_columns])
    xgb_model = make_xgboost(iterations["xgboost_trees"])
    xgb_model.fit(x, y, verbose=False)
    lgb_model = make_lightgbm(iterations["lightgbm_trees"])
    lgb_model.fit(x, y)
    cat_model = make_catboost(iterations["catboost_trees"])
    cat_model.fit(frame[feature_columns], y, cat_features=["part_number"], verbose=False)

    joblib.dump(preprocessor, MODEL_DIR / "tree_preprocessor.joblib")
    (MODEL_DIR / "lightgbm.txt").write_text(
        lgb_model.booster_.model_to_string(), encoding="utf-8"
    )

    # Native libraries are more reliable with an ASCII temporary path on Windows.
    with tempfile.TemporaryDirectory() as temp_directory:
        temp_path = Path(temp_directory)
        xgb_temp = temp_path / "xgboost.json"
        cat_temp = temp_path / "catboost.cbm"
        xgb_model.save_model(xgb_temp)
        cat_model.save_model(cat_temp)
        shutil.copy2(xgb_temp, MODEL_DIR / "xgboost.json")
        shutil.copy2(cat_temp, MODEL_DIR / "catboost.cbm")

    # Reload each serialization and check a small prediction batch.
    check_frame = frame.head(min(20, len(frame)))
    check_x = preprocessor.transform(check_frame[feature_columns])
    with tempfile.TemporaryDirectory() as temp_directory:
        temp_path = Path(temp_directory)
        xgb_temp = temp_path / "xgboost.json"
        cat_temp = temp_path / "catboost.cbm"
        shutil.copy2(MODEL_DIR / "xgboost.json", xgb_temp)
        shutil.copy2(MODEL_DIR / "catboost.cbm", cat_temp)
        loaded_xgb = XGBRegressor()
        loaded_xgb.load_model(xgb_temp)
        loaded_cat = CatBoostRegressor()
        loaded_cat.load_model(cat_temp)
        loaded_lgb = lgb.Booster(model_str=(MODEL_DIR / "lightgbm.txt").read_text(encoding="utf-8"))
        checks = [
            loaded_xgb.predict(check_x),
            loaded_lgb.predict(check_x),
            loaded_cat.predict(check_frame[feature_columns]),
        ]
        if not all(np.isfinite(values).all() for values in checks):
            raise RuntimeError("Reloaded model produced a non-finite prediction.")


def build_part_coverage(
    daily: pd.DataFrame,
    supervised: pd.DataFrame,
    pooled_predictions: pd.DataFrame,
) -> pd.DataFrame:
    daily_summary = daily.groupby("part_number").agg(
        Daily_Rows=("date", "size"),
        First_Daily_Date=("date", "min"),
        Last_Daily_Date=("date", "max"),
    )
    supervised_summary = supervised.groupby("part_number").agg(
        Supervised_Rows=("target", "size"),
        First_Target_Date=("target_date", "min"),
        Last_Target_Date=("target_date", "max"),
        Plan_Zero_Rows=("plan_d3", lambda values: int((values == 0).sum())),
        Actual_Zero_Rows=("target", lambda values: int((values == 0).sum())),
    )
    evaluation_summary = pooled_predictions.groupby("part_number").agg(
        Pooled_Evaluation_Rows=("actual", "size"),
        Cold_Start_Evaluation_Rows=("is_cold_start", "sum"),
    )
    coverage = daily_summary.join(supervised_summary, how="outer").join(
        evaluation_summary, how="outer"
    )
    coverage["Included_In_Final_Training"] = coverage["Supervised_Rows"].fillna(0).gt(0)
    coverage["Included_In_Pooled_Evaluation"] = coverage["Pooled_Evaluation_Rows"].fillna(0).gt(0)
    coverage = coverage.reset_index().sort_values("part_number")
    for column in [
        "First_Daily_Date",
        "Last_Daily_Date",
        "First_Target_Date",
        "Last_Target_Date",
    ]:
        coverage[column] = pd.to_datetime(coverage[column]).dt.strftime("%Y-%m-%d")
    return coverage


def markdown_table(frame: pd.DataFrame, decimals: int = 3) -> str:
    formatted = frame.copy()
    for column in formatted.select_dtypes(include=[np.number]).columns:
        formatted[column] = formatted[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.{decimals}f}"
        )
    header = "| " + " | ".join(map(str, formatted.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
    body = [
        "| " + " | ".join(map(str, row)) + " |"
        for row in formatted.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *body])


def write_report(
    qa: dict[str, object],
    pooled_metrics: pd.DataFrame,
    holdout_metrics: pd.DataFrame,
    plan_reference_metrics: pd.DataFrame,
    macro_metrics: pd.DataFrame,
    subset_metrics: pd.DataFrame,
    fold_coverage: pd.DataFrame,
    part_coverage: pd.DataFrame,
    selected_iterations: dict[str, int],
) -> None:
    best = pooled_metrics.sort_values("MAE").iloc[0]
    plan = plan_reference_metrics.iloc[0]
    plan_zero = subset_metrics[subset_metrics["Subset"] == "D+3 plan = 0"].sort_values("MAE")
    cold = subset_metrics[subset_metrics["Subset"] == "Cold-start part"].sort_values("MAE")
    comparison_columns = ["Model", "N", "Parts", "MAE", "RMSE", "WAPE_pct", "Forecast_Bias", "R2"]
    macro_columns = [
        "Model",
        "Parts",
        "Per_part_MAE_mean",
        "Per_part_MAE_median",
        "Per_part_RMSE_mean",
        "Per_part_WAPE_pct_mean",
        "Per_part_Forecast_Bias_mean",
        "Per_part_R2_mean",
    ]
    fold_columns = [
        "Fold",
        "Train_Target_End",
        "Test_Target_Start",
        "Test_Target_End",
        "Train_Rows",
        "Train_Parts",
        "Test_Rows",
        "Test_Parts",
        "Cold_Start_Parts",
        "Cold_Start_Part_List",
    ]

    report = f"""# 117개 부품 D+3 발주량 예측 모델 재평가

## 최종 결론

- 5개 시간순 walk-forward 구간을 합친 주 평가에서 MAE가 가장 낮은 방법은 **{best['Model']}**이다.
- 주 평가 표본은 {int(best['N']):,}개이며, 117개 부품이 모두 최소 한 번 평가에 포함됐다.
- 최종 저장 모델은 목표값을 만들 수 있는 {qa['supervised_rows']:,}개 표본과 117개 부품 전체를 다시 학습했다.
- D+3 계획량이 0인 행 {qa['plan_d3_zero_rows']:,}개와 실제 D+3 수요가 0인 행 {qa['target_zero_rows']:,}개는 삭제하지 않았다.
- 음수가 가능한 계획 수정량과 요일 사인·코사인은 필터링하지 않았다. 수량 열만 음수 여부를 검사했고 제거된 행은 {qa['invalid_negative_quantity_rows_removed']:,}개다.

## 주 평가: pooled walk-forward

{markdown_table(pooled_metrics[comparison_columns])}

Forecast Bias는 `예측값 - 실제값`의 평균이다. 양수는 과대예측, 음수는 과소예측이다.

D+3 계획량 자체를 그대로 예측값으로 사용한 참고 기준은 MAE {plan['MAE']:.3f}, RMSE {plan['RMSE']:.3f}, WAPE {plan['WAPE_pct']:.2f}%, Bias {plan['Forecast_Bias']:.3f}, R² {plan['R2']:.3f}였다. 이 참고 기준은 요청한 네 방법의 순위에는 포함하지 않았다.

## 마지막 기간 holdout

마지막 fold만 보면 최근 기간 {fold_coverage.iloc[-1]['Test_Target_Start']}~{fold_coverage.iloc[-1]['Test_Target_End']}의 {int(holdout_metrics.iloc[0]['N']):,}개 표본과 {int(holdout_metrics.iloc[0]['Parts'])}개 활동 부품을 평가한다. 이 구간에 관측값이 없는 부품은 평가할 수 없지만 최종 학습에서는 제외하지 않았다.

{markdown_table(holdout_metrics[comparison_columns])}

## 부품별 동일 가중치 요약

전체 행을 한꺼번에 계산하면 데이터가 많은 부품의 영향이 커진다. 아래는 각 부품의 지표를 먼저 계산한 뒤 117개 부품을 동일 가중치로 평균한 결과다.

{markdown_table(macro_metrics[macro_columns])}

## 계획량 0 및 cold-start 성능

### D+3 계획량이 0인 행

{markdown_table(plan_zero[comparison_columns])}

### 해당 fold 학습 구간에 없던 부품

{markdown_table(cold[comparison_columns])}

## 시간순 분할과 부품 포함 범위

{markdown_table(fold_coverage[fold_columns], decimals=0)}

- Fold 1의 cold-start 부품은 학습 구간 뒤에 처음 등장한 부품이다.
- Part 115는 짧은 기간과 불연속 날짜 때문에 과거 LSTM 시퀀스 조건에서 빠졌지만, 이번에는 9개 지도학습 표본을 최종 학습에 사용했다.
- Part 116은 늦게 등장해 앞선 fold에서는 학습할 수 없었다. 등장 후의 fold와 최종 전체 재학습에서는 포함됐다.
- Part 109처럼 일찍 종료된 부품도 첫 평가 구간에 포함해 pooled 평가의 부품 수를 117개로 맞췄다.

## 모델 저장 기준

- XGBoost: {selected_iterations['xgboost_trees']}개 트리
- LightGBM: {selected_iterations['lightgbm_trees']}개 트리
- CatBoost: {selected_iterations['catboost_trees']}개 트리
- 트리 수는 각 walk-forward fold 내부 검증에서 선택한 최적 트리 수의 중앙값이다.
- 저장 전용 최종 모델은 성능 평가가 끝난 뒤 5,097개 표본 전체로 재학습했다. 따라서 저장 모델 자체의 훈련 데이터 성능을 최종 성능처럼 보고하지 않았다.

## 해석상 제한

- 원본 기간은 {qa['source_start']}~{qa['source_end']}로 짧다. 계절성, 장기 추세, 휴일 효과를 충분히 검증하기 어렵다.
- 일부 부품은 매우 짧게 존재한다. pooled 평가는 모든 부품을 포함하지만 부품마다 평가 행 수가 다르므로 부품별 동일 가중치 결과도 함께 봐야 한다.
- 실제량이 모두 0인 부분집합에서는 WAPE와 R²가 정의되지 않을 수 있다. 이 경우 MAE, RMSE, Bias를 본다.
"""
    (OUTPUT_DIR / "TREE_MODEL_EVALUATION_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading and preparing data...", flush=True)
    daily, supervised, numeric_features, qa = load_and_prepare()
    if qa["supervised_parts"] != 117:
        raise RuntimeError(f"Expected 117 supervised parts, found {qa['supervised_parts']}.")

    print(
        f"Prepared {len(supervised):,} supervised rows across "
        f"{supervised['part_number'].nunique()} parts. Starting walk-forward evaluation...",
        flush=True,
    )
    pooled_predictions, fold_metrics, fold_coverage, selected_iterations = run_walk_forward(
        supervised, numeric_features
    )
    if pooled_predictions["part_number"].nunique() != 117:
        raise RuntimeError(
            "Pooled walk-forward evaluation did not cover all 117 parts: "
            f"{pooled_predictions['part_number'].nunique()} parts."
        )

    pooled_metrics = metrics_from_prediction_frame(pooled_predictions, REQUESTED_MODELS)
    plan_reference_metrics = metrics_from_prediction_frame(
        pooled_predictions, [REFERENCE_MODEL]
    )
    plan_mae = float(plan_reference_metrics.iloc[0]["MAE"])
    pooled_metrics["MAE_improvement_vs_D3_plan_pct"] = (
        (plan_mae - pooled_metrics["MAE"]) / plan_mae * 100
    )
    pooled_metrics = pooled_metrics.sort_values("MAE").reset_index(drop=True)

    last_fold = int(pooled_predictions["Fold"].max())
    last_predictions = pooled_predictions[pooled_predictions["Fold"] == last_fold]
    holdout_metrics = metrics_from_prediction_frame(last_predictions, REQUESTED_MODELS)
    holdout_metrics = holdout_metrics.sort_values("MAE").reset_index(drop=True)

    subset_metrics = evaluate_subsets(pooled_predictions)
    per_part_metrics, macro_metrics = evaluate_per_part(pooled_predictions)
    part_coverage = build_part_coverage(daily, supervised, pooled_predictions)

    qa.update(
        {
            "pooled_evaluation_rows": int(len(pooled_predictions)),
            "pooled_evaluation_parts": int(pooled_predictions["part_number"].nunique()),
            "final_training_rows": int(len(supervised)),
            "final_training_parts": int(supervised["part_number"].nunique()),
        }
    )

    pooled_metrics.to_csv(
        OUTPUT_DIR / "metrics_pooled_walk_forward.csv", index=False, encoding="utf-8-sig"
    )
    holdout_metrics.to_csv(
        OUTPUT_DIR / "metrics_last_holdout.csv", index=False, encoding="utf-8-sig"
    )
    plan_reference_metrics.to_csv(
        OUTPUT_DIR / "metrics_d3_plan_reference.csv", index=False, encoding="utf-8-sig"
    )
    fold_metrics.to_csv(
        OUTPUT_DIR / "metrics_by_fold.csv", index=False, encoding="utf-8-sig"
    )
    subset_metrics.to_csv(
        OUTPUT_DIR / "metrics_by_subset.csv", index=False, encoding="utf-8-sig"
    )
    per_part_metrics.to_csv(
        OUTPUT_DIR / "metrics_by_part.csv", index=False, encoding="utf-8-sig"
    )
    macro_metrics.to_csv(
        OUTPUT_DIR / "metrics_per_part_macro.csv", index=False, encoding="utf-8-sig"
    )
    pooled_predictions.to_csv(
        OUTPUT_DIR / "predictions_pooled_walk_forward.csv", index=False, encoding="utf-8-sig"
    )
    fold_coverage.to_csv(
        OUTPUT_DIR / "fold_coverage.csv", index=False, encoding="utf-8-sig"
    )
    part_coverage.to_csv(
        OUTPUT_DIR / "part_coverage.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([{"Metric": key, "Value": value} for key, value in qa.items()]).to_csv(
        OUTPUT_DIR / "data_quality_summary.csv", index=False, encoding="utf-8-sig"
    )

    print("Fitting final deployable models on all supervised rows...", flush=True)
    fit_and_save_final_models(supervised, numeric_features, selected_iterations)

    metadata = {
        "horizon_days": HORIZON_DAYS,
        "evaluation_design": "five expanding walk-forward folds; pooled metrics cover all 117 parts",
        "source_path": str(DATA_PATH),
        "feature_columns": ["part_number"] + numeric_features,
        "selected_iterations": selected_iterations,
        "final_training_rows": int(len(supervised)),
        "final_training_parts": int(supervised["part_number"].nunique()),
        "pooled_evaluation_rows": int(len(pooled_predictions)),
        "pooled_evaluation_parts": int(pooled_predictions["part_number"].nunique()),
        "plan_zero_rows_retained": int((supervised["plan_d3"] == 0).sum()),
        "target_zero_rows_retained": int((supervised["target"] == 0).sum()),
        "library_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lgb.__version__,
        },
    }
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    write_report(
        qa,
        pooled_metrics,
        holdout_metrics,
        plan_reference_metrics,
        macro_metrics,
        subset_metrics,
        fold_coverage,
        part_coverage,
        selected_iterations,
    )

    print("\nPooled walk-forward metrics", flush=True)
    print(pooled_metrics.to_string(index=False), flush=True)
    print("\nLast holdout metrics", flush=True)
    print(holdout_metrics.to_string(index=False), flush=True)
    print(f"\nOutputs: {OUTPUT_DIR}", flush=True)
    print(f"Models:  {MODEL_DIR}", flush=True)


if __name__ == "__main__":
    main()
