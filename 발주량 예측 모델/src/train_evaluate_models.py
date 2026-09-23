from __future__ import annotations

import json
import math
import os
import random
from copy import deepcopy
from pathlib import Path

import joblib
import lightgbm as lgb
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "tmp" / "matplotlib")
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor


SEED = 42
HORIZON_DAYS = 3
SEQUENCE_LENGTH = 5

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Dataset_공급망 최적화 AI 데이터셋" / "data" / "data.xls"
OUTPUT_DIR = ROOT / "outputs" / "model_evaluation"
MODEL_DIR = ROOT / "models"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def exact_lag(frame: pd.DataFrame, value_col: str, days: int) -> np.ndarray:
    lookup = frame.set_index(["part_number", "date"])[value_col]
    keys = pd.MultiIndex.from_arrays(
        [frame["part_number"], frame["date"] - pd.Timedelta(days=days)],
        names=["part_number", "date"],
    )
    return lookup.reindex(keys).to_numpy()


def load_and_prepare() -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    raw = pd.read_excel(DATA_PATH)
    source_rows, source_cols = raw.shape

    # The XLS column labels are damaged by its legacy encoding in some readers.
    # The guidebook fixes the semantic positions used below.
    selected = pd.DataFrame(
        {
            "part_number": raw.iloc[:, 0].astype(str),
            "actual_d": pd.to_numeric(raw.iloc[:, 11], errors="coerce"),
            "plan_d3": pd.to_numeric(raw.iloc[:, 44], errors="coerce"),
            "plan_d4": pd.to_numeric(raw.iloc[:, 55], errors="coerce"),
            "plan_d5": pd.to_numeric(raw.iloc[:, 56], errors="coerce"),
            "timestamp": pd.to_datetime(
                raw.iloc[:, 83].astype(str), format="%Y%m%d%H%M", errors="coerce"
            ),
        }
    )
    selected["date"] = selected["timestamp"].dt.normalize()
    selected = selected.sort_values(["part_number", "timestamp"])

    # Guidebook rule: retain the final log for each part and calendar day.
    daily = (
        selected.dropna()
        .groupby(["part_number", "date"], as_index=False, sort=False)
        .tail(1)
        .sort_values(["part_number", "date"])
        .reset_index(drop=True)
    )

    daily["plan_mean"] = daily[["plan_d3", "plan_d4", "plan_d5"]].mean(axis=1)
    daily["plan_max"] = daily[["plan_d3", "plan_d4", "plan_d5"]].max(axis=1)
    daily["plan_min"] = daily[["plan_d3", "plan_d4", "plan_d5"]].min(axis=1)
    daily["plan_range"] = daily["plan_max"] - daily["plan_min"]
    daily["plan_trend_d3_to_d5"] = daily["plan_d3"] - daily["plan_d5"]

    # Compare forecasts for the same delivery date: today's D+3 vs yesterday's D+4,
    # and yesterday's D+4 vs the D+5 value from two days ago.
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
    daily["actual_roll3_mean"] = daily[lag3].mean(axis=1)
    daily["actual_roll3_std"] = daily[lag3].std(axis=1, ddof=0)
    daily["actual_roll7_mean"] = daily[lag7].mean(axis=1)
    daily["actual_roll7_std"] = daily[lag7].std(axis=1, ddof=0)

    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["dow_sin"] = np.sin(2 * np.pi * daily["day_of_week"] / 7)
    daily["dow_cos"] = np.cos(2 * np.pi * daily["day_of_week"] / 7)
    daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)
    daily["month"] = daily["date"].dt.month
    daily["week_of_year"] = daily["date"].dt.isocalendar().week.astype(int)
    daily["days_since_start"] = (daily["date"] - daily["date"].min()).dt.days

    # Align the target by calendar date so that an origin-day D+3 plan is compared
    # with the actual quantity exactly three days later.
    daily["target_date"] = daily["date"] + pd.Timedelta(days=HORIZON_DAYS)
    target_lookup = daily.set_index(["part_number", "date"])["actual_d"]
    target_keys = pd.MultiIndex.from_arrays(
        [daily["part_number"], daily["target_date"]],
        names=["part_number", "date"],
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
        "actual_roll7_mean",
        "actual_roll7_std",
        "dow_sin",
        "dow_cos",
        "is_weekend",
        "month",
        "week_of_year",
        "days_since_start",
    ]

    required = numeric_features + ["target"]
    supervised = daily.dropna(subset=required).copy()
    nonnegative_quantities = [
        "plan_d3",
        "plan_d4",
        "plan_d5",
        "actual_lag1",
        "actual_lag2",
        "actual_lag3",
        "actual_lag7",
        "actual_roll3_mean",
        "actual_roll7_mean",
        "target",
    ]
    supervised = supervised[
        (supervised[nonnegative_quantities] >= 0).all(axis=1)
    ].copy()
    supervised = supervised.sort_values(["part_number", "date"]).reset_index(drop=True)

    qa = {
        "source_rows": source_rows,
        "source_columns": source_cols,
        "source_parts": int(raw.iloc[:, 0].nunique()),
        "source_start": str(selected["date"].min().date()),
        "source_end": str(selected["date"].max().date()),
        "source_days": int(selected["date"].nunique()),
        "source_missing_cells": int(raw.isna().sum().sum()),
        "source_exact_duplicate_rows": int(raw.duplicated().sum()),
        "daily_rows_after_last_log": int(len(daily)),
        "supervised_rows_before_sequence_filter": int(len(supervised)),
        "source_zero_actual_rate": float((selected["actual_d"] == 0).mean()),
    }
    return supervised, numeric_features, qa


def make_sequences(
    frame: pd.DataFrame, numeric_features: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    row_indices: list[int] = []
    sequences: list[np.ndarray] = []
    for _, group in frame.groupby("part_number", sort=False):
        group = group.sort_values("date")
        positions = group.index.to_numpy()
        dates = group["date"].to_numpy()
        values = group[numeric_features].to_numpy(dtype=np.float32)
        for end in range(SEQUENCE_LENGTH - 1, len(group)):
            start = end - SEQUENCE_LENGTH + 1
            window_dates = dates[start : end + 1]
            if not np.all(np.diff(window_dates).astype("timedelta64[D]").astype(int) == 1):
                continue
            row_indices.append(int(positions[end]))
            sequences.append(values[start : end + 1])

    common = frame.loc[row_indices].copy().reset_index(drop=True)
    seq_array = np.stack(sequences).astype(np.float32)
    return common, seq_array


def split_masks(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    dates = np.array(sorted(frame["target_date"].unique()))
    train_end_pos = max(1, int(len(dates) * 0.70))
    val_end_pos = max(train_end_pos + 1, int(len(dates) * 0.80))
    train_end = pd.Timestamp(dates[train_end_pos - 1])
    val_end = pd.Timestamp(dates[val_end_pos - 1])
    train = (frame["target_date"] <= train_end).to_numpy()
    val = ((frame["target_date"] > train_end) & (frame["target_date"] <= val_end)).to_numpy()
    test = (frame["target_date"] > val_end).to_numpy()
    boundaries = {
        "train_target_end": str(train_end.date()),
        "validation_target_start": str(frame.loc[val, "target_date"].min().date()),
        "validation_target_end": str(val_end.date()),
        "test_target_start": str(frame.loc[test, "target_date"].min().date()),
        "test_target_end": str(frame.loc[test, "target_date"].max().date()),
    }
    return train, val, test, boundaries


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = float(np.abs(y_true).sum())
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "WAPE_pct": float(np.abs(y_true - y_pred).sum() / denom * 100) if denom else np.nan,
        "Forecast_Bias": float(np.mean(y_pred - y_true)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def make_tree_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "part",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["part_number"],
            ),
            ("numeric", "passthrough", numeric_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


class DemandLSTM(nn.Module):
    def __init__(self, n_features: int, n_parts: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.15,
            batch_first=True,
        )
        self.part_embedding = nn.Embedding(n_parts, 8)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + 8, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )

    def forward(self, sequence: torch.Tensor, part_id: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(sequence)
        combined = torch.cat([hidden[-1], self.part_embedding(part_id)], dim=1)
        return self.head(combined).squeeze(1)


def prepare_lstm_arrays(
    sequences: np.ndarray,
    part_ids: np.ndarray,
    targets: np.ndarray,
    fit_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, float, float]:
    scaler = StandardScaler()
    n_features = sequences.shape[-1]
    scaler.fit(sequences[fit_mask].reshape(-1, n_features))
    scaled_sequences = scaler.transform(sequences.reshape(-1, n_features)).reshape(sequences.shape)
    log_targets = np.log1p(targets.astype(np.float32))
    y_mean = float(log_targets[fit_mask].mean())
    y_std = float(log_targets[fit_mask].std())
    if y_std < 1e-8:
        y_std = 1.0
    scaled_targets = (log_targets - y_mean) / y_std
    return (
        scaled_sequences.astype(np.float32),
        part_ids.astype(np.int64),
        scaled_targets.astype(np.float32),
        scaler,
        y_mean,
        y_std,
    )


def train_lstm(
    sequences: np.ndarray,
    part_ids: np.ndarray,
    scaled_targets: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray | None,
    n_parts: int,
    max_epochs: int = 100,
    patience: int = 12,
) -> tuple[DemandLSTM, int]:
    model = DemandLSTM(sequences.shape[-1], n_parts)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    dataset = TensorDataset(
        torch.from_numpy(sequences[train_mask]),
        torch.from_numpy(part_ids[train_mask]),
        torch.from_numpy(scaled_targets[train_mask]),
    )
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, generator=generator)

    best_state = deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 1
    wait = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_x, batch_part, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x, batch_part), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if val_mask is None:
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            continue

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(
                model(torch.from_numpy(sequences[val_mask]), torch.from_numpy(part_ids[val_mask])),
                torch.from_numpy(scaled_targets[val_mask]),
            ).item()
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def predict_lstm(
    model: DemandLSTM,
    sequences: np.ndarray,
    part_ids: np.ndarray,
    y_mean: float,
    y_std: float,
) -> np.ndarray:
    with torch.no_grad():
        scaled = model(torch.from_numpy(sequences), torch.from_numpy(part_ids)).numpy()
    prediction = np.expm1(scaled * y_std + y_mean)
    return np.clip(prediction, 0, None)


def fit_holdout_models(
    frame: pd.DataFrame,
    sequences: np.ndarray,
    numeric_features: list[str],
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    part_to_id: dict[str, int],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, int]]:
    feature_cols = ["part_number"] + numeric_features
    y = frame["target"].to_numpy(dtype=float)
    part_ids = frame["part_number"].map(part_to_id).to_numpy(dtype=np.int64)

    preprocessor = make_tree_preprocessor(numeric_features)
    x_train = preprocessor.fit_transform(frame.loc[train_mask, feature_cols])
    x_val = preprocessor.transform(frame.loc[val_mask, feature_cols])
    x_test = preprocessor.transform(frame.loc[test_mask, feature_cols])

    xgb_model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=5.0,
        random_state=SEED,
        n_jobs=4,
        tree_method="hist",
        early_stopping_rounds=60,
    )
    xgb_model.fit(x_train, y[train_mask], eval_set=[(x_val, y[val_mask])], verbose=False)
    xgb_prediction = np.clip(xgb_model.predict(x_test), 0, None)
    print(f"XGBoost complete: {xgb_model.best_iteration + 1} trees", flush=True)

    lgb_model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=2000,
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
    lgb_model.fit(
        x_train,
        y[train_mask],
        eval_set=[(x_val, y[val_mask])],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(60, verbose=False)],
    )
    lgb_prediction = np.clip(lgb_model.predict(x_test), 0, None)
    print(f"LightGBM complete: {lgb_model.best_iteration_} trees", flush=True)

    scaled_seq, part_ids, scaled_y, seq_scaler, y_mean, y_std = prepare_lstm_arrays(
        sequences, part_ids, y, train_mask
    )
    lstm_model, best_epoch = train_lstm(
        scaled_seq,
        part_ids,
        scaled_y,
        train_mask,
        val_mask,
        n_parts=len(part_to_id),
    )
    lstm_prediction = predict_lstm(
        lstm_model, scaled_seq[test_mask], part_ids[test_mask], y_mean, y_std
    )
    print(f"LSTM complete: best epoch {best_epoch}", flush=True)

    predictions = {
        "D+3 Plan Baseline": frame.loc[test_mask, "plan_d3"].to_numpy(dtype=float),
        "3-day Moving Average": frame.loc[test_mask, "actual_roll3_mean"].to_numpy(dtype=float),
        "XGBoost": xgb_prediction,
        "LightGBM": lgb_prediction,
        "LSTM": lstm_prediction,
    }
    rows = []
    for name, pred in predictions.items():
        rows.append({"Model": name, **metrics(y[test_mask], pred)})
    result = pd.DataFrame(rows)

    baseline = result.loc[result["Model"] == "D+3 Plan Baseline"].iloc[0]
    for metric_name in ["MAE", "RMSE", "WAPE_pct"]:
        result[f"{metric_name}_improvement_vs_plan_pct"] = (
            (float(baseline[metric_name]) - result[metric_name]) / float(baseline[metric_name]) * 100
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, MODEL_DIR / "tree_preprocessor.joblib")
    xgb_model.save_model(MODEL_DIR / "xgboost.json")
    # LightGBM's native Windows writer cannot open some non-ASCII paths.
    # Writing its serialized model string through Python preserves the model.
    (MODEL_DIR / "lightgbm.txt").write_text(
        lgb_model.booster_.model_to_string(), encoding="utf-8"
    )
    torch.save(
        {
            "state_dict": lstm_model.state_dict(),
            "numeric_features": numeric_features,
            "sequence_length": SEQUENCE_LENGTH,
            "part_to_id": part_to_id,
            "feature_scaler_mean": seq_scaler.mean_.tolist(),
            "feature_scaler_scale": seq_scaler.scale_.tolist(),
            "target_log_mean": y_mean,
            "target_log_std": y_std,
            "hidden_size": 32,
        },
        MODEL_DIR / "lstm.pt",
    )
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "horizon_days": HORIZON_DAYS,
                "sequence_length": SEQUENCE_LENGTH,
                "part_to_id": part_to_id,
                "numeric_features": numeric_features,
                "xgboost_best_trees": int(xgb_model.best_iteration + 1),
                "lightgbm_best_trees": int(lgb_model.best_iteration_),
                "lstm_best_epochs": int(best_epoch),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    complexity = {
        "xgboost_trees": int(xgb_model.best_iteration + 1),
        "lightgbm_trees": int(lgb_model.best_iteration_),
        "lstm_epochs": int(best_epoch),
    }
    return result, predictions, complexity


def walk_forward_evaluation(
    frame: pd.DataFrame,
    sequences: np.ndarray,
    numeric_features: list[str],
    part_to_id: dict[str, int],
    complexity: dict[str, int],
) -> pd.DataFrame:
    unique_dates = np.array(sorted(frame["target_date"].unique()))
    bounds = [(0.55, 0.70), (0.70, 0.85), (0.85, 1.00)]
    feature_cols = ["part_number"] + numeric_features
    y = frame["target"].to_numpy(dtype=float)
    part_ids_all = frame["part_number"].map(part_to_id).to_numpy(dtype=np.int64)
    rows: list[dict[str, object]] = []

    for fold, (train_frac, test_frac) in enumerate(bounds, start=1):
        train_end_idx = max(1, int(len(unique_dates) * train_frac))
        test_end_idx = max(train_end_idx + 1, int(len(unique_dates) * test_frac))
        test_end_idx = min(test_end_idx, len(unique_dates))
        train_end = pd.Timestamp(unique_dates[train_end_idx - 1])
        test_end = pd.Timestamp(unique_dates[test_end_idx - 1])
        train_mask = (frame["target_date"] <= train_end).to_numpy()
        test_mask = (
            (frame["target_date"] > train_end) & (frame["target_date"] <= test_end)
        ).to_numpy()

        preprocessor = make_tree_preprocessor(numeric_features)
        x_train = preprocessor.fit_transform(frame.loc[train_mask, feature_cols])
        x_test = preprocessor.transform(frame.loc[test_mask, feature_cols])

        xgb_model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=max(1, complexity["xgboost_trees"]),
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=5,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.2,
            reg_lambda=5.0,
            random_state=SEED + fold,
            n_jobs=4,
            tree_method="hist",
        )
        xgb_model.fit(x_train, y[train_mask], verbose=False)

        lgb_model = LGBMRegressor(
            objective="regression_l1",
            n_estimators=max(1, complexity["lightgbm_trees"]),
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.2,
            reg_lambda=5.0,
            random_state=SEED + fold,
            n_jobs=4,
            verbosity=-1,
        )
        lgb_model.fit(x_train, y[train_mask])

        scaled_seq, part_ids, scaled_y, _, y_mean, y_std = prepare_lstm_arrays(
            sequences, part_ids_all, y, train_mask
        )
        lstm_model, _ = train_lstm(
            scaled_seq,
            part_ids,
            scaled_y,
            train_mask,
            val_mask=None,
            n_parts=len(part_to_id),
            max_epochs=max(1, complexity["lstm_epochs"]),
            patience=complexity["lstm_epochs"] + 1,
        )

        fold_predictions = {
            "D+3 Plan Baseline": frame.loc[test_mask, "plan_d3"].to_numpy(dtype=float),
            "3-day Moving Average": frame.loc[test_mask, "actual_roll3_mean"].to_numpy(dtype=float),
            "XGBoost": np.clip(xgb_model.predict(x_test), 0, None),
            "LightGBM": np.clip(lgb_model.predict(x_test), 0, None),
            "LSTM": predict_lstm(
                lstm_model, scaled_seq[test_mask], part_ids[test_mask], y_mean, y_std
            ),
        }
        for name, pred in fold_predictions.items():
            rows.append(
                {
                    "Fold": fold,
                    "Train_Target_End": str(train_end.date()),
                    "Test_Target_Start": str(frame.loc[test_mask, "target_date"].min().date()),
                    "Test_Target_End": str(test_end.date()),
                    "N_Test": int(test_mask.sum()),
                    "Model": name,
                    **metrics(y[test_mask], pred),
                }
            )
        print(f"Walk-forward fold {fold}/3 complete", flush=True)
    return pd.DataFrame(rows)


def save_plots(metrics_frame: pd.DataFrame, predictions: pd.DataFrame) -> None:
    order = metrics_frame.sort_values("MAE")["Model"].tolist()
    metric_specs = [
        ("MAE", "MAE (lower is better)"),
        ("RMSE", "RMSE (lower is better)"),
        ("WAPE_pct", "WAPE % (lower is better)"),
        ("Forecast_Bias", "Forecast Bias (closer to 0)"),
        ("R2", "R-squared (higher is better)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    colors = ["#2F5597", "#5B9BD5", "#70AD47", "#ED7D31", "#A5A5A5"]
    for ax, (metric_name, title) in zip(axes.flat, metric_specs):
        values = metrics_frame.set_index("Model").loc[order, metric_name]
        ax.barh(order, values, color=colors[: len(order)])
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        ax.invert_yaxis()
    axes.flat[-1].axis("off")
    fig.suptitle("Three-day demand forecast: holdout comparison", fontsize=16)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    agg = (
        predictions.groupby("target_date", as_index=False)
        .agg(
            actual=("actual", "sum"),
            plan_d3=("D+3 Plan Baseline", "sum"),
            xgboost=("XGBoost", "sum"),
            lightgbm=("LightGBM", "sum"),
            lstm=("LSTM", "sum"),
        )
        .sort_values("target_date")
    )
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(agg["target_date"], agg["actual"], marker="o", linewidth=2.5, label="Actual")
    ax.plot(agg["target_date"], agg["plan_d3"], marker="o", label="D+3 plan")
    ax.plot(agg["target_date"], agg["xgboost"], marker="o", label="XGBoost")
    ax.plot(agg["target_date"], agg["lightgbm"], marker="o", label="LightGBM")
    ax.plot(agg["target_date"], agg["lstm"], marker="o", label="LSTM")
    ax.set_title("Holdout period: daily total actual vs forecast")
    ax.set_ylabel("Quantity")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "test_predictions_aggregate.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    holdout: pd.DataFrame,
    walk_summary: pd.DataFrame,
    qa: dict[str, object],
    boundaries: dict[str, str],
    train_n: int,
    val_n: int,
    test_n: int,
) -> None:
    best = holdout.sort_values("MAE").iloc[0]
    plan = holdout.loc[holdout["Model"] == "D+3 Plan Baseline"].iloc[0]
    model_only = holdout[holdout["Model"].isin(["XGBoost", "LightGBM", "LSTM"])]
    best_ai = model_only.sort_values("MAE").iloc[0]

    def frame_to_markdown(frame: pd.DataFrame) -> str:
        rounded = frame.round(4)
        header = "| " + " | ".join(map(str, rounded.columns)) + " |"
        separator = "| " + " | ".join(["---"] * len(rounded.columns)) + " |"
        body = []
        for row in rounded.itertuples(index=False, name=None):
            body.append("| " + " | ".join(map(str, row)) + " |")
        return "\n".join([header, separator, *body])

    holdout_text = frame_to_markdown(holdout)
    walk_text = frame_to_markdown(walk_summary)
    report = f"""# VMI 3일 선행 발주량 예측 모델 평가

## 결론

- Holdout MAE가 가장 낮은 모델은 **{best['Model']}**이며 MAE는 **{best['MAE']:.3f}**이다.
- AI 모델 중 MAE가 가장 낮은 모델은 **{best_ai['Model']}**이다.
- 이 모델의 고객사 D+3 계획량 대비 MAE 개선율은 **{best_ai['MAE_improvement_vs_plan_pct']:.2f}%**이다.
- 고객사 D+3 계획량 baseline의 MAE는 **{plan['MAE']:.3f}**, WAPE는 **{plan['WAPE_pct']:.2f}%**이다.

## Holdout 성능

{holdout_text}

Forecast Bias는 `예측값 - 실제값`의 평균이다. 양수는 과대예측, 음수는 과소예측을 뜻한다.

## Walk-forward 평균 및 변동성

{walk_text}

## 데이터와 검증 설계

- 원본: {qa['source_rows']:,}행, {qa['source_columns']}열, {qa['source_parts']}개 부품, {qa['source_start']}~{qa['source_end']}.
- 가이드북 기준에 따라 `Part Number + 날짜`별 마지막 로그만 사용했다.
- 예측 기준일의 D+3 계획량과 정확히 3일 뒤 날짜의 실제 발주량을 연결했다.
- 부품별 연속 5일 입력 시퀀스가 존재하는 표본만 세 모델과 baseline의 공통 평가 표본으로 사용했다.
- 시간 순서 분할: Train target 종료 {boundaries['train_target_end']}, Validation {boundaries['validation_target_start']}~{boundaries['validation_target_end']}, Test {boundaries['test_target_start']}~{boundaries['test_target_end']}.
- 표본 수: Train {train_n:,}, Validation {val_n:,}, Test {test_n:,}.
- 입력: D+3/D+4/D+5 계획량, 동일 납품일 기준 계획 수정량, 실제량 lag 1/2/3/7일, rolling 통계, 달력 변수, 부품 식별자.
- 수량 예측값은 운영상 가능한 범위에 맞춰 0 이상으로 제한했다.

## 해석 시 주의사항

- 원본 기간이 49일로 짧아, 특히 LSTM 성능은 기간 변화에 민감할 수 있다.
- R²는 부품-일 단위 전체 표본에서 계산했으며 음수일 수 있다. 음수는 해당 평가 구간에서 평균 예측보다 설명력이 낮음을 뜻한다.
- WAPE는 전체 실제 수량 합을 분모로 계산해 실제량 0인 개별 행에서도 정의된다.
- 이번 결과는 재현 가능한 1차 기준 모델이다. 운영 적용 전에는 더 긴 기간, 휴일/생산중단 정보, 재고·리드타임 변수를 추가해 재검증해야 한다.
"""
    (OUTPUT_DIR / "MODEL_EVALUATION_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading and preparing source data...", flush=True)
    supervised, numeric_features, qa = load_and_prepare()
    common, sequences = make_sequences(supervised, numeric_features)
    qa["common_sequence_rows"] = int(len(common))
    qa["common_sequence_parts"] = int(common["part_number"].nunique())
    qa["common_sequence_zero_target_rate"] = float((common["target"] == 0).mean())

    parts = sorted(common["part_number"].unique())
    part_to_id = {part: i for i, part in enumerate(parts)}
    train_mask, val_mask, test_mask, boundaries = split_masks(common)
    print(
        f"Prepared {len(common):,} common samples: "
        f"train={train_mask.sum():,}, val={val_mask.sum():,}, test={test_mask.sum():,}",
        flush=True,
    )

    holdout, test_predictions, complexity = fit_holdout_models(
        common,
        sequences,
        numeric_features,
        train_mask,
        val_mask,
        test_mask,
        part_to_id,
    )
    holdout = holdout.sort_values("MAE").reset_index(drop=True)
    holdout.to_csv(OUTPUT_DIR / "metrics_holdout.csv", index=False, encoding="utf-8-sig")

    prediction_frame = common.loc[
        test_mask, ["part_number", "date", "target_date", "target"]
    ].copy()
    prediction_frame = prediction_frame.rename(columns={"date": "forecast_origin", "target": "actual"})
    for name, prediction in test_predictions.items():
        prediction_frame[name] = prediction
    prediction_frame.to_csv(
        OUTPUT_DIR / "predictions_test.csv", index=False, encoding="utf-8-sig"
    )

    print("Starting walk-forward evaluation...", flush=True)
    walk = walk_forward_evaluation(
        common, sequences, numeric_features, part_to_id, complexity
    )
    walk.to_csv(OUTPUT_DIR / "walk_forward_metrics.csv", index=False, encoding="utf-8-sig")
    walk_summary = (
        walk.groupby("Model", as_index=False)
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            WAPE_pct_mean=("WAPE_pct", "mean"),
            WAPE_pct_std=("WAPE_pct", "std"),
            Forecast_Bias_mean=("Forecast_Bias", "mean"),
            Forecast_Bias_std=("Forecast_Bias", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
        )
        .sort_values("MAE_mean")
    )
    walk_summary.to_csv(
        OUTPUT_DIR / "walk_forward_summary.csv", index=False, encoding="utf-8-sig"
    )

    pd.DataFrame([{"Metric": key, "Value": value} for key, value in qa.items()]).to_csv(
        OUTPUT_DIR / "data_quality_summary.csv", index=False, encoding="utf-8-sig"
    )
    save_plots(holdout, prediction_frame)
    write_report(
        holdout,
        walk_summary,
        qa,
        boundaries,
        int(train_mask.sum()),
        int(val_mask.sum()),
        int(test_mask.sum()),
    )
    print("\nHoldout metrics", flush=True)
    print(holdout.to_string(index=False), flush=True)
    print("\nWalk-forward summary", flush=True)
    print(walk_summary.to_string(index=False), flush=True)
    print(f"\nOutputs written to: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
