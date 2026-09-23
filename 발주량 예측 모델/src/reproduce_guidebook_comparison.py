from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import tensorflow as tf
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


SEED = 42
LOOKBACK = 3
DELAY = 3

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Dataset_공급망 최적화 AI 데이터셋" / "data" / "data.xls"
OUTPUT_DIR = ROOT / "outputs" / "guidebook_reproduction"
MODEL_DIR = ROOT / "models" / "guidebook_reproduction"

PART_COLUMN = "Part Number"
TIME_COLUMN = "CRET_TIME"
FEATURE_COLUMNS = [
    "D일 투입예정 수량(D일계획)",
    "D+3일 투입예정 수량(Total)",
    "D+4일 투입예정 수량(Total)",
    "D+5일 투입예정 수량",
]
MODEL_NAMES = ["XGBoost", "LightGBM", "CatBoost", "LSTM", "3-day Moving Average"]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    error = y_pred - y_true
    denominator = float(np.abs(y_true).sum())
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "WAPE_pct": float(np.abs(error).sum() / denominator * 100) if denominator else np.nan,
        "Forecast_Bias": float(error.mean()),
        "R2": float(r2_score(y_true, y_pred)),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in values) + " |"
        for values in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def load_part_daily(raw: pd.DataFrame, part_number: str) -> pd.DataFrame:
    # This intentionally mirrors guidebook Code 9-15: stored total columns are used,
    # and the final log within each calendar day is retained.
    selected = raw.loc[raw[PART_COLUMN] == part_number, FEATURE_COLUMNS + [TIME_COLUMN]].copy()
    selected[TIME_COLUMN] = pd.to_datetime(
        selected[TIME_COLUMN].astype(str), format="%Y%m%d%H%M", errors="raise"
    )
    selected = selected.sort_values(TIME_COLUMN)
    selected["year"] = selected[TIME_COLUMN].dt.year
    selected["month"] = selected[TIME_COLUMN].dt.month
    selected["day"] = selected[TIME_COLUMN].dt.day
    daily = (
        selected.groupby(["year", "month", "day"], sort=True, as_index=False)
        .last()
        .sort_values(TIME_COLUMN)
        .reset_index(drop=True)
    )
    daily["date"] = daily[TIME_COLUMN].dt.normalize()
    return daily[["date"] + FEATURE_COLUMNS]


def to_timeseries_data(
    daily: pd.DataFrame, lookback: int = LOOKBACK, delay: int = DELAY
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = daily[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    dates = daily["date"].to_numpy()
    output_len = len(values) - (lookback + delay) + 1
    inputs = np.zeros((output_len, lookback, values.shape[-1]), dtype=np.float32)
    targets = np.zeros(output_len, dtype=np.float32)
    origin_dates = np.empty(output_len, dtype="datetime64[ns]")
    target_dates = np.empty(output_len, dtype="datetime64[ns]")
    for i in range(output_len):
        inputs[i] = values[i : i + lookback]
        target_index = i + lookback + delay - 1
        targets[i] = values[target_index, 0]
        origin_dates[i] = dates[i + lookback - 1]
        target_dates[i] = dates[target_index]
    return inputs, targets, origin_dates, target_dates


def split_array(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = int(0.7 * len(values))
    second = int(0.8 * len(values))
    return tuple(np.split(values, [first, second]))  # type: ignore[return-value]


def scale_inputs(
    train: np.ndarray, validation: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train.reshape(-1, train.shape[-1])).reshape(train.shape)
    validation_scaled = scaler.transform(
        validation.reshape(-1, validation.shape[-1])
    ).reshape(validation.shape)
    test_scaled = scaler.transform(test.reshape(-1, test.shape[-1])).reshape(test.shape)
    return train_scaled, validation_scaled, test_scaled, scaler


def build_lstm(input_shape: tuple[int, int], scenario: str) -> tf.keras.Model:
    if scenario == "Part 94 only":
        first_units, second_units = 8, 8
    else:
        first_units, second_units = 32, 16
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(
                first_units, dropout=0.2, activation="relu", return_sequences=True
            ),
            tf.keras.layers.LSTM(second_units, dropout=0.2, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(optimizer="adam", loss="mae")
    return model


def train_lstm(
    scenario: str,
    x_train: np.ndarray,
    x_validation: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int], tf.keras.Model, StandardScaler]:
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_validation_scaled = y_scaler.transform(y_validation.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))

    model = build_lstm((x_train.shape[1], x_train.shape[2]), scenario)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        )
    ]
    history = model.fit(
        x_train,
        y_train_scaled,
        epochs=100,
        batch_size=4,
        validation_data=(x_validation, y_validation_scaled),
        callbacks=callbacks,
        verbose=0,
        shuffle=True,
    )
    scaled_mae = float(model.evaluate(x_test, y_test_scaled, verbose=0))
    predictions_scaled = model.predict(x_test, verbose=0)
    predictions = y_scaler.inverse_transform(predictions_scaled).reshape(-1)
    details: dict[str, float | int] = {
        "epochs_ran": int(len(history.history["loss"])),
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "best_validation_scaled_mae": float(np.min(history.history["val_loss"])),
        "test_scaled_mae": scaled_mae,
    }
    return predictions, details, model, y_scaler


def train_tree_models(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_validation: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    train_flat = x_train.reshape(len(x_train), -1)
    validation_flat = x_validation.reshape(len(x_validation), -1)
    test_flat = x_test.reshape(len(x_test), -1)

    xgboost = XGBRegressor(
        objective="reg:absoluteerror",
        eval_metric="mae",
        n_estimators=500,
        learning_rate=0.03,
        max_depth=2,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=4,
        tree_method="hist",
        early_stopping_rounds=30,
    )
    xgboost.fit(
        train_flat,
        y_train,
        eval_set=[(validation_flat, y_validation)],
        verbose=False,
    )

    lightgbm = LGBMRegressor(
        objective="regression_l1",
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=7,
        max_depth=3,
        min_child_samples=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=4,
        verbosity=-1,
    )
    lightgbm.fit(
        train_flat,
        y_train,
        eval_set=[(validation_flat, y_validation)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    catboost = CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=500,
        learning_rate=0.03,
        depth=3,
        l2_leaf_reg=3.0,
        random_seed=SEED,
        thread_count=4,
        allow_writing_files=False,
        verbose=False,
    )
    catboost.fit(
        train_flat,
        y_train,
        eval_set=(validation_flat, y_validation),
        early_stopping_rounds=30,
        use_best_model=True,
        verbose=False,
    )

    predictions = {
        "XGBoost": xgboost.predict(test_flat),
        "LightGBM": lightgbm.predict(test_flat),
        "CatBoost": catboost.predict(test_flat),
    }
    models: dict[str, object] = {
        "XGBoost": xgboost,
        "LightGBM": lightgbm,
        "CatBoost": catboost,
    }
    return predictions, models


def save_tree_models(models: dict[str, object], scenario_slug: str) -> None:
    xgboost = models["XGBoost"]
    lightgbm = models["LightGBM"]
    catboost = models["CatBoost"]
    xgboost.save_model(MODEL_DIR / f"{scenario_slug}_xgboost.json")  # type: ignore[attr-defined]
    # LightGBM's native writer cannot reliably handle this workspace's Korean path.
    joblib.dump(lightgbm, MODEL_DIR / f"{scenario_slug}_lightgbm.joblib")
    catboost.save_model(MODEL_DIR / f"{scenario_slug}_catboost.cbm")  # type: ignore[attr-defined]


def run_scenario(
    scenario: str,
    scenario_slug: str,
    x_train: np.ndarray,
    x_validation: np.ndarray,
    x_test: np.ndarray,
    x94_test_raw: np.ndarray,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    y_test: np.ndarray,
    origin_test: np.ndarray,
    target_test: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    tree_predictions, tree_models = train_tree_models(
        x_train, x_validation, x_test, y_train, y_validation
    )
    lstm_predictions, lstm_details, lstm_model, _ = train_lstm(
        scenario,
        x_train,
        x_validation,
        x_test,
        y_train,
        y_validation,
        y_test,
    )
    predictions = {
        **tree_predictions,
        "LSTM": lstm_predictions,
        "3-day Moving Average": x94_test_raw[:, :, 0].mean(axis=1),
    }

    metric_rows = []
    for model_name in MODEL_NAMES:
        metrics = calculate_metrics(y_test, predictions[model_name])
        metric_rows.append(
            {"Scenario": scenario, "Model": model_name, "N": len(y_test), **metrics}
        )
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame["MAE_Rank"] = metrics_frame["MAE"].rank(method="min").astype(int)
    metrics_frame = metrics_frame.sort_values(["MAE_Rank", "Model"]).reset_index(drop=True)

    prediction_frame = pd.DataFrame(
        {
            "Scenario": scenario,
            "Origin_Date": pd.to_datetime(origin_test).date,
            "Target_Date": pd.to_datetime(target_test).date,
            "Actual_Part_94": y_test,
            **predictions,
        }
    )

    save_tree_models(tree_models, scenario_slug)
    lstm_model.save(MODEL_DIR / f"{scenario_slug}_lstm.keras")
    details: dict[str, object] = {
        "lstm": lstm_details,
        "winner": metrics_frame.iloc[0]["Model"],
        "winner_mae": float(metrics_frame.iloc[0]["MAE"]),
    }
    return metrics_frame, prediction_frame, details


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_excel(DATA_PATH)
    daily94 = load_part_daily(raw, "Part 94")
    daily95 = load_part_daily(raw, "Part 95")
    if not np.array_equal(daily94["date"].to_numpy(), daily95["date"].to_numpy()):
        raise ValueError("Part 94 and Part 95 daily dates are not aligned.")

    x94, y94, origins, targets = to_timeseries_data(daily94)
    x95, _, origins95, targets95 = to_timeseries_data(daily95)
    if not np.array_equal(origins, origins95) or not np.array_equal(targets, targets95):
        raise ValueError("Part 94 and Part 95 sequence dates are not aligned.")
    if x94.shape != (44, 3, 4) or x95.shape != (44, 3, 4):
        raise ValueError(f"Guidebook shapes not reproduced: {x94.shape=}, {x95.shape=}")

    x94_train_raw, x94_validation_raw, x94_test_raw = split_array(x94)
    x95_train_raw, x95_validation_raw, x95_test_raw = split_array(x95)
    y_train, y_validation, y_test = split_array(y94)
    _, _, origin_test = split_array(origins)
    _, _, target_test = split_array(targets)
    if (len(y_train), len(y_validation), len(y_test)) != (30, 5, 9):
        raise ValueError("Guidebook 30/5/9 split not reproduced.")

    x94_train, x94_validation, x94_test, _ = scale_inputs(
        x94_train_raw, x94_validation_raw, x94_test_raw
    )
    x95_train, x95_validation, x95_test, _ = scale_inputs(
        x95_train_raw, x95_validation_raw, x95_test_raw
    )
    both_train = np.concatenate([x94_train, x95_train], axis=2)
    both_validation = np.concatenate([x94_validation, x95_validation], axis=2)
    both_test = np.concatenate([x94_test, x95_test], axis=2)

    all_metrics = []
    all_predictions = []
    scenario_details: dict[str, object] = {}
    for args in [
        ("Part 94 only", "part94_only", x94_train, x94_validation, x94_test),
        ("Part 94 + Part 95", "part94_part95", both_train, both_validation, both_test),
    ]:
        scenario, slug, x_train, x_validation, x_test = args
        print(f"Training scenario: {scenario}", flush=True)
        metrics, predictions, details = run_scenario(
            scenario,
            slug,
            x_train,
            x_validation,
            x_test,
            x94_test_raw,
            y_train,
            y_validation,
            y_test,
            origin_test,
            target_test,
        )
        all_metrics.append(metrics)
        all_predictions.append(predictions)
        scenario_details[scenario] = details

    metrics_frame = pd.concat(all_metrics, ignore_index=True)
    predictions_frame = pd.concat(all_predictions, ignore_index=True)
    metrics_frame.to_csv(OUTPUT_DIR / "metrics.csv", index=False, encoding="utf-8-sig")
    predictions_frame.to_csv(OUTPUT_DIR / "predictions.csv", index=False, encoding="utf-8-sig")

    published = {
        "Part 94 only": {"scaled_mae": 1.2037, "inverse_mae": 5.4498},
        "Part 94 + Part 95": {"scaled_mae": 1.0878, "inverse_mae": 4.9255},
    }
    comparison_rows = []
    for scenario in published:
        our_lstm_mae = float(
            metrics_frame.loc[
                (metrics_frame["Scenario"] == scenario) & (metrics_frame["Model"] == "LSTM"),
                "MAE",
            ].iloc[0]
        )
        our_scaled_mae = float(scenario_details[scenario]["lstm"]["test_scaled_mae"])  # type: ignore[index]
        comparison_rows.append(
            {
                "Scenario": scenario,
                "Guidebook_Scaled_MAE": published[scenario]["scaled_mae"],
                "Our_Scaled_MAE": our_scaled_mae,
                "Guidebook_Inverse_MAE": published[scenario]["inverse_mae"],
                "Our_Inverse_MAE": our_lstm_mae,
                "Inverse_MAE_Difference": our_lstm_mae - published[scenario]["inverse_mae"],
            }
        )
    comparison_frame = pd.DataFrame(comparison_rows)
    comparison_frame.to_csv(
        OUTPUT_DIR / "guidebook_lstm_comparison.csv", index=False, encoding="utf-8-sig"
    )

    overall_winner = metrics_frame.sort_values("MAE").iloc[0]
    metadata = {
        "seed": SEED,
        "data_path": str(DATA_PATH),
        "daily_rows_part94": len(daily94),
        "daily_rows_part95": len(daily95),
        "sequence_shape_part94": list(x94.shape),
        "sequence_shape_part95": list(x95.shape),
        "split_counts": {"train": len(y_train), "validation": len(y_validation), "test": len(y_test)},
        "test_target_dates": [str(pd.Timestamp(target_test.min()).date()), str(pd.Timestamp(target_test.max()).date())],
        "test_targets_part94": [float(value) for value in y_test],
        "published_guidebook": published,
        "scenario_details": scenario_details,
        "overall_best_scenario": overall_winner["Scenario"],
        "overall_best_model": overall_winner["Model"],
        "overall_best_mae": float(overall_winner["MAE"]),
        "notes": [
            "Guidebook stored total columns are used without replacing them by slot sums.",
            "Daily last log, lookback=3, delay=3, and chronological 70/10/20 split reproduce the guidebook.",
            "The target in both scenarios is Part 94 actual D quantity.",
            "Tree-model hyperparameters are fixed before test evaluation and use the five validation samples for early stopping.",
        ],
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_tables = []
    for scenario in ["Part 94 only", "Part 94 + Part 95"]:
        view = metrics_frame.loc[metrics_frame["Scenario"] == scenario].copy()
        for column in ["MAE", "RMSE", "WAPE_pct", "Forecast_Bias", "R2"]:
            view[column] = view[column].map(lambda value: f"{value:.6f}")
        report_tables.append(
            f"## {scenario}\n\n"
            + markdown_table(
                view[["MAE_Rank", "Model", "N", "MAE", "RMSE", "WAPE_pct", "Forecast_Bias", "R2"]]
            )
        )
    compare_view = comparison_frame.copy()
    for column in compare_view.columns[1:]:
        compare_view[column] = compare_view[column].map(lambda value: f"{value:.6f}")
    report = (
        "# Guidebook Reproduction Model Comparison\n\n"
        "This experiment reproduces the guidebook's Part 94 and Part 94+95 setups. "
        "Both setups predict Part 94 only.\n\n"
        "## Guidebook LSTM comparison\n\n"
        + markdown_table(compare_view)
        + "\n\n"
        + "\n\n".join(report_tables)
        + "\n\n## Overall result\n\n"
        + f"- Best scenario: {overall_winner['Scenario']}\n"
        + f"- Best model: {overall_winner['Model']}\n"
        + f"- Best MAE: {float(overall_winner['MAE']):.6f}\n"
        + "- Test size: 9 Part 94 target days\n"
    )
    (OUTPUT_DIR / "GUIDEBOOK_REPRODUCTION_REPORT.md").write_text(report, encoding="utf-8")

    print(metrics_frame.to_string(index=False), flush=True)
    print("\nGuidebook comparison", flush=True)
    print(comparison_frame.to_string(index=False), flush=True)
    print(
        f"\nOverall winner: {overall_winner['Scenario']} / {overall_winner['Model']} "
        f"(MAE={float(overall_winner['MAE']):.6f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
