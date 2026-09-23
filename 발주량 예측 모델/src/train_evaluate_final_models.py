from __future__ import annotations

import json
import math
import platform
import random
from copy import deepcopy
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor


SEED = 42
HORIZON_DAYS = 3
SEQUENCE_LENGTH = 3
MIN_SEQUENCE_COUNT = 10
EXCLUDED_PARTS = {"Part 115"}

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Dataset_공급망 최적화 AI 데이터셋" / "data" / "data.xls"
OUTPUT_DIR = ROOT / "outputs" / "final_model_evaluation"
MODEL_DIR = ROOT / "models" / "final_v3"

SEQUENCE_FEATURES = ["actual_d", "plan_d3", "plan_d4", "plan_d5"]
CALENDAR_FEATURES = ["dow_sin", "dow_cos", "is_weekend", "month", "days_since_start"]
MODEL_NAMES = ["XGBoost", "LightGBM", "CatBoost", "LSTM", "3-day Moving Average"]
REFERENCE_MODEL = "D+3 Plan Reference"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = float(np.abs(y_true).sum())
    variance = float(np.var(y_true))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "WAPE_pct": (
            float(np.abs(y_true - y_pred).sum() / denominator * 100)
            if denominator > 0
            else np.nan
        ),
        "Forecast_Bias": float(np.mean(y_pred - y_true)),
        "R2": (
            float(r2_score(y_true, y_pred))
            if len(y_true) >= 2 and variance > 0
            else np.nan
        ),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for values in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join([header, divider, *rows])


def load_daily_data() -> tuple[pd.DataFrame, dict[str, object]]:
    raw = pd.read_excel(DATA_PATH)
    row_id = np.arange(len(raw))

    actual_slot_sum = raw.iloc[:, 1:11].sum(axis=1)
    d1_slot_sum = raw.iloc[:, 12:22].sum(axis=1)
    d2_slot_sum = raw.iloc[:, 23:33].sum(axis=1)
    d3_slot_sum = raw.iloc[:, 34:44].sum(axis=1)
    d4_slot_sum = raw.iloc[:, 45:55].sum(axis=1)

    source_totals = {
        "D": pd.to_numeric(raw.iloc[:, 11], errors="coerce"),
        "D+1": pd.to_numeric(raw.iloc[:, 22], errors="coerce"),
        "D+2": pd.to_numeric(raw.iloc[:, 33], errors="coerce"),
        "D+3": pd.to_numeric(raw.iloc[:, 44], errors="coerce"),
        "D+4": pd.to_numeric(raw.iloc[:, 55], errors="coerce"),
    }
    slot_sums = {
        "D": actual_slot_sum,
        "D+1": d1_slot_sum,
        "D+2": d2_slot_sum,
        "D+3": d3_slot_sum,
        "D+4": d4_slot_sum,
    }
    mismatch_counts = {
        period: int((source_totals[period] != slot_sums[period]).sum())
        for period in slot_sums
    }

    selected = pd.DataFrame(
        {
            "row_id": row_id,
            "part_number": raw.iloc[:, 0].astype(str),
            "actual_d": pd.to_numeric(actual_slot_sum, errors="coerce"),
            "plan_d1": pd.to_numeric(d1_slot_sum, errors="coerce"),
            "plan_d2": pd.to_numeric(d2_slot_sum, errors="coerce"),
            "plan_d3": pd.to_numeric(d3_slot_sum, errors="coerce"),
            "plan_d4": pd.to_numeric(d4_slot_sum, errors="coerce"),
            "plan_d5": pd.to_numeric(raw.iloc[:, 56], errors="coerce"),
            "timestamp": pd.to_datetime(
                raw.iloc[:, 83].astype(str), format="%Y%m%d%H%M", errors="coerce"
            ),
        }
    )
    required = [
        "part_number",
        "actual_d",
        "plan_d1",
        "plan_d2",
        "plan_d3",
        "plan_d4",
        "plan_d5",
        "timestamp",
    ]
    missing_selected_rows = int(selected[required].isna().any(axis=1).sum())
    selected = selected.dropna(subset=required).copy()
    selected["date"] = selected["timestamp"].dt.normalize()
    selected = selected.sort_values(["part_number", "date", "timestamp", "row_id"])

    duplicate_part_timestamp_rows = int(
        selected.duplicated(["part_number", "timestamp"], keep=False).sum()
    )
    daily = (
        selected.groupby(["part_number", "date"], as_index=False, sort=False)
        .tail(1)
        .sort_values(["part_number", "date"])
        .reset_index(drop=True)
    )

    negative_quantity_rows = int(
        (daily[["actual_d", "plan_d1", "plan_d2", "plan_d3", "plan_d4", "plan_d5"]] < 0)
        .any(axis=1)
        .sum()
    )
    if negative_quantity_rows:
        raise ValueError(f"Negative quantity rows found: {negative_quantity_rows}")

    quality = {
        "source_rows": int(len(raw)),
        "source_columns": int(raw.shape[1]),
        "source_parts": int(raw.iloc[:, 0].nunique()),
        "source_missing_cells": int(raw.isna().sum().sum()),
        "source_exact_duplicate_rows": int(raw.duplicated().sum()),
        "source_start": str(selected["date"].min().date()),
        "source_end": str(selected["date"].max().date()),
        "source_dates": int(selected["date"].nunique()),
        "selected_rows_with_missing_required_values": missing_selected_rows,
        "duplicate_part_timestamp_rows": duplicate_part_timestamp_rows,
        "daily_rows_after_last_log": int(len(daily)),
        "daily_parts_after_last_log": int(daily["part_number"].nunique()),
        "negative_quantity_rows": negative_quantity_rows,
        "stored_total_mismatch_rows": mismatch_counts,
    }
    return daily, quality


def build_sequences(
    daily: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, object]]:
    daily = daily.copy()
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["dow_sin"] = np.sin(2 * np.pi * daily["day_of_week"] / 7)
    daily["dow_cos"] = np.cos(2 * np.pi * daily["day_of_week"] / 7)
    daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)
    daily["month"] = daily["date"].dt.month
    daily["days_since_start"] = (daily["date"] - daily["date"].min()).dt.days

    rows: list[dict[str, object]] = []
    sequences: list[np.ndarray] = []
    coverage_rows: list[dict[str, object]] = []

    for part_number, group in daily.groupby("part_number", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        date_lookup = group.set_index("date")
        valid_for_part: list[tuple[int, pd.Timestamp, pd.Timestamp, np.ndarray]] = []

        for end in range(SEQUENCE_LENGTH - 1, len(group)):
            start = end - SEQUENCE_LENGTH + 1
            input_window = group.iloc[start : end + 1]
            date_diffs = input_window["date"].diff().dropna().dt.days.to_numpy()
            if not np.all(date_diffs == 1):
                continue

            origin_date = pd.Timestamp(input_window.iloc[-1]["date"])
            target_date = origin_date + pd.Timedelta(days=HORIZON_DAYS)
            if target_date not in date_lookup.index:
                continue

            sequence = input_window[SEQUENCE_FEATURES].to_numpy(dtype=np.float32)
            valid_for_part.append((end, origin_date, target_date, sequence))

        coverage_rows.append(
            {
                "part_number": part_number,
                "daily_rows": int(len(group)),
                "first_date": str(group["date"].min().date()),
                "last_date": str(group["date"].max().date()),
                "valid_sequence_count": int(len(valid_for_part)),
                "excluded": part_number in EXCLUDED_PARTS,
                "exclusion_reason": (
                    f"valid_sequence_count < {MIN_SEQUENCE_COUNT}"
                    if part_number in EXCLUDED_PARTS
                    else ""
                ),
            }
        )

        if part_number in EXCLUDED_PARTS:
            continue

        for end, origin_date, target_date, sequence in valid_for_part:
            target_row = date_lookup.loc[target_date]
            if isinstance(target_row, pd.DataFrame):
                target_row = target_row.iloc[-1]
            origin_row = group.iloc[end]
            record: dict[str, object] = {
                "part_number": part_number,
                "origin_date": origin_date,
                "target_date": target_date,
                "target": float(target_row["actual_d"]),
                "moving_average_3d": float(sequence[:, 0].mean()),
                "plan_d3_reference": float(origin_row["plan_d3"]),
                **{feature: float(origin_row[feature]) for feature in CALENDAR_FEATURES},
            }
            for step_index, suffix in enumerate(["lag2", "lag1", "origin"]):
                for feature_index, feature_name in enumerate(SEQUENCE_FEATURES):
                    record[f"{feature_name}_{suffix}"] = float(
                        sequence[step_index, feature_index]
                    )
            rows.append(record)
            sequences.append(sequence)

    frame = pd.DataFrame(rows).sort_values(
        ["target_date", "part_number", "origin_date"]
    ).reset_index(drop=True)
    sequence_array = np.stack(sequences).astype(np.float32)

    # The sorted frame order must match the sequence tensor order.
    unsorted_keys = pd.DataFrame(rows)[["part_number", "origin_date", "target_date"]]
    sequence_by_key = {
        (row.part_number, row.origin_date, row.target_date): sequence
        for row, sequence in zip(unsorted_keys.itertuples(index=False), sequences)
    }
    sequence_array = np.stack(
        [
            sequence_by_key[(row.part_number, row.origin_date, row.target_date)]
            for row in frame[["part_number", "origin_date", "target_date"]].itertuples(
                index=False
            )
        ]
    ).astype(np.float32)

    coverage = pd.DataFrame(coverage_rows).sort_values(
        ["valid_sequence_count", "part_number"]
    )
    below_threshold = set(
        coverage.loc[coverage["valid_sequence_count"] < MIN_SEQUENCE_COUNT, "part_number"]
    )
    if below_threshold != EXCLUDED_PARTS:
        raise ValueError(
            f"Expected excluded parts {sorted(EXCLUDED_PARTS)}, found {sorted(below_threshold)}"
        )

    quantity_columns = [
        column
        for column in frame.columns
        if column.startswith(("actual_d_", "plan_d3_", "plan_d4_", "plan_d5_"))
    ] + ["target", "moving_average_3d", "plan_d3_reference"]
    negative_rows = int((frame[quantity_columns] < 0).any(axis=1).sum())
    if negative_rows:
        raise ValueError(f"Negative sequence quantity rows found: {negative_rows}")

    qa = {
        "sequence_length": SEQUENCE_LENGTH,
        "horizon_days": HORIZON_DAYS,
        "minimum_sequence_count": MIN_SEQUENCE_COUNT,
        "excluded_parts": sorted(EXCLUDED_PARTS),
        "eligible_rows": int(len(frame)),
        "eligible_parts": int(frame["part_number"].nunique()),
        "plan_zero_rows": int((frame["plan_d3_reference"] == 0).sum()),
        "target_zero_rows": int((frame["target"] == 0).sum()),
    }
    return frame, sequence_array, coverage, qa


def make_split_masks(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    target_dates = np.array(sorted(frame["target_date"].unique()))
    train_count = max(1, int(len(target_dates) * 0.70))
    validation_end = max(train_count + 1, int(len(target_dates) * 0.80))
    validation_end = min(validation_end, len(target_dates) - 1)

    train_end = pd.Timestamp(target_dates[train_count - 1])
    validation_end_date = pd.Timestamp(target_dates[validation_end - 1])
    train_mask = (frame["target_date"] <= train_end).to_numpy()
    validation_mask = (
        (frame["target_date"] > train_end)
        & (frame["target_date"] <= validation_end_date)
    ).to_numpy()
    test_mask = (frame["target_date"] > validation_end_date).to_numpy()

    for name, mask in {
        "train": train_mask,
        "validation": validation_mask,
        "test": test_mask,
    }.items():
        if not mask.any():
            raise ValueError(f"Empty {name} split")

    split_info = {
        "unique_target_dates": int(len(target_dates)),
        "train_target_start": str(frame.loc[train_mask, "target_date"].min().date()),
        "train_target_end": str(frame.loc[train_mask, "target_date"].max().date()),
        "validation_target_start": str(
            frame.loc[validation_mask, "target_date"].min().date()
        ),
        "validation_target_end": str(
            frame.loc[validation_mask, "target_date"].max().date()
        ),
        "test_target_start": str(frame.loc[test_mask, "target_date"].min().date()),
        "test_target_end": str(frame.loc[test_mask, "target_date"].max().date()),
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "train_parts": int(frame.loc[train_mask, "part_number"].nunique()),
        "validation_parts": int(frame.loc[validation_mask, "part_number"].nunique()),
        "test_parts": int(frame.loc[test_mask, "part_number"].nunique()),
    }
    return train_mask, validation_mask, test_mask, split_info


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


class DemandLSTM(nn.Module):
    def __init__(
        self,
        sequence_features: int,
        calendar_features: int,
        part_count: int,
        hidden_size: int = 32,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=sequence_features,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.20,
            batch_first=True,
        )
        self.part_embedding = nn.Embedding(part_count, 8)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + 8 + calendar_features, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        part_id: torch.Tensor,
        calendar: torch.Tensor,
    ) -> torch.Tensor:
        _, (hidden, _) = self.lstm(sequence)
        combined = torch.cat(
            [hidden[-1], self.part_embedding(part_id), calendar], dim=1
        )
        return self.head(combined).squeeze(1)


def train_lstm(
    sequences: np.ndarray,
    calendars: np.ndarray,
    part_ids: np.ndarray,
    targets: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    part_count: int,
) -> tuple[DemandLSTM, dict[str, object], np.ndarray]:
    sequence_scaler = StandardScaler()
    sequence_scaler.fit(sequences[train_mask].reshape(-1, sequences.shape[-1]))
    scaled_sequences = sequence_scaler.transform(
        sequences.reshape(-1, sequences.shape[-1])
    ).reshape(sequences.shape).astype(np.float32)

    calendar_scaler = StandardScaler()
    scaled_calendars = calendar_scaler.fit_transform(calendars[train_mask])
    all_scaled_calendars = calendar_scaler.transform(calendars).astype(np.float32)

    log_targets = np.log1p(targets).astype(np.float32)
    target_mean = float(log_targets[train_mask].mean())
    target_std = float(log_targets[train_mask].std())
    if target_std < 1e-8:
        target_std = 1.0
    scaled_targets = ((log_targets - target_mean) / target_std).astype(np.float32)

    model = DemandLSTM(
        sequence_features=sequences.shape[-1],
        calendar_features=calendars.shape[-1],
        part_count=part_count,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    dataset = TensorDataset(
        torch.from_numpy(scaled_sequences[train_mask]),
        torch.from_numpy(part_ids[train_mask]),
        torch.from_numpy(all_scaled_calendars[train_mask]),
        torch.from_numpy(scaled_targets[train_mask]),
    )
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, generator=generator)

    best_state = deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 1
    wait = 0
    max_epochs = 150
    patience = 18

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_sequence, batch_part, batch_calendar, batch_target in loader:
            optimizer.zero_grad()
            prediction = model(batch_sequence, batch_part, batch_calendar)
            loss = loss_fn(prediction, batch_target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = loss_fn(
                model(
                    torch.from_numpy(scaled_sequences[validation_mask]),
                    torch.from_numpy(part_ids[validation_mask]),
                    torch.from_numpy(all_scaled_calendars[validation_mask]),
                ),
                torch.from_numpy(scaled_targets[validation_mask]),
            ).item()
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scaled_prediction = model(
            torch.from_numpy(scaled_sequences),
            torch.from_numpy(part_ids),
            torch.from_numpy(all_scaled_calendars),
        ).numpy()
    predictions = np.clip(
        np.expm1(scaled_prediction * target_std + target_mean), 0, None
    )
    artifacts = {
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "sequence_scaler_mean": sequence_scaler.mean_.tolist(),
        "sequence_scaler_scale": sequence_scaler.scale_.tolist(),
        "calendar_scaler_mean": calendar_scaler.mean_.tolist(),
        "calendar_scaler_scale": calendar_scaler.scale_.tolist(),
        "target_log_mean": target_mean,
        "target_log_std": target_std,
    }
    return model, artifacts, predictions


def train_and_predict(
    frame: pd.DataFrame,
    sequences: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    test_mask: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, object], dict[str, object]]:
    flat_sequence_features = [
        f"{feature}_{suffix}"
        for suffix in ["lag2", "lag1", "origin"]
        for feature in SEQUENCE_FEATURES
    ]
    numeric_features = flat_sequence_features + CALENDAR_FEATURES
    feature_columns = ["part_number"] + numeric_features
    targets = frame["target"].to_numpy(dtype=float)

    preprocessor = make_tree_preprocessor(numeric_features)
    x_train = preprocessor.fit_transform(frame.loc[train_mask, feature_columns])
    x_validation = preprocessor.transform(frame.loc[validation_mask, feature_columns])
    x_test = preprocessor.transform(frame.loc[test_mask, feature_columns])

    xgboost = XGBRegressor(
        objective="reg:absoluteerror",
        eval_metric="mae",
        n_estimators=2000,
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
        early_stopping_rounds=60,
    )
    xgboost.fit(
        x_train,
        targets[train_mask],
        eval_set=[(x_validation, targets[validation_mask])],
        verbose=False,
    )
    print(f"XGBoost complete: {xgboost.best_iteration + 1} trees", flush=True)

    lightgbm = LGBMRegressor(
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
    lightgbm.fit(
        x_train,
        targets[train_mask],
        eval_set=[(x_validation, targets[validation_mask])],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(60, verbose=False)],
    )
    print(f"LightGBM complete: {lightgbm.best_iteration_} trees", flush=True)

    catboost = CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5.0,
        random_seed=SEED,
        thread_count=4,
        allow_writing_files=False,
        verbose=False,
    )
    catboost.fit(
        frame.loc[train_mask, feature_columns],
        targets[train_mask],
        cat_features=["part_number"],
        eval_set=(
            frame.loc[validation_mask, feature_columns],
            targets[validation_mask],
        ),
        early_stopping_rounds=60,
        verbose=False,
    )
    print(f"CatBoost complete: {catboost.get_best_iteration() + 1} trees", flush=True)

    part_names = sorted(frame.loc[train_mask, "part_number"].unique())
    part_to_id = {part: index for index, part in enumerate(part_names)}
    unknown_parts = sorted(set(frame["part_number"]) - set(part_to_id))
    if unknown_parts:
        raise ValueError(f"Parts absent from training split: {unknown_parts}")
    part_ids = frame["part_number"].map(part_to_id).to_numpy(dtype=np.int64).copy()
    calendars = frame[CALENDAR_FEATURES].to_numpy(dtype=np.float32)
    lstm, lstm_artifacts, lstm_all_predictions = train_lstm(
        sequences,
        calendars,
        part_ids,
        targets,
        train_mask,
        validation_mask,
        len(part_to_id),
    )
    print(f"LSTM complete: best epoch {lstm_artifacts['best_epoch']}", flush=True)

    test_frame = frame.loc[test_mask]
    predictions = {
        "XGBoost": np.clip(xgboost.predict(x_test), 0, None),
        "LightGBM": np.clip(lightgbm.predict(x_test), 0, None),
        "CatBoost": np.clip(
            catboost.predict(test_frame[feature_columns]), 0, None
        ),
        "LSTM": np.clip(lstm_all_predictions[test_mask], 0, None),
        "3-day Moving Average": test_frame["moving_average_3d"].to_numpy(dtype=float),
        REFERENCE_MODEL: test_frame["plan_d3_reference"].to_numpy(dtype=float),
    }
    models = {
        "tree_preprocessor": preprocessor,
        "xgboost": xgboost,
        "lightgbm": lightgbm,
        "catboost": catboost,
        "lstm": lstm,
    }
    metadata = {
        "numeric_features": numeric_features,
        "feature_columns": feature_columns,
        "part_to_id": part_to_id,
        "xgboost_best_trees": int(xgboost.best_iteration + 1),
        "lightgbm_best_trees": int(lightgbm.best_iteration_),
        "catboost_best_trees": int(catboost.get_best_iteration() + 1),
        "lstm": lstm_artifacts,
    }
    return predictions, models, metadata


def make_metrics(
    prediction_frame: pd.DataFrame, predictions: dict[str, np.ndarray]
) -> pd.DataFrame:
    y_true = prediction_frame["actual"].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for model_name in MODEL_NAMES + [REFERENCE_MODEL]:
        prediction = predictions[model_name]
        rows.append(
            {
                "Model": model_name,
                "Eligible_For_Ranking": model_name in MODEL_NAMES,
                "N": int(len(y_true)),
                "Parts": int(prediction_frame["part_number"].nunique()),
                **calculate_metrics(y_true, prediction),
            }
        )
    metrics = pd.DataFrame(rows)
    ranking_mask = metrics["Eligible_For_Ranking"]
    ranked = metrics.loc[ranking_mask].sort_values(["MAE", "RMSE"])
    rank_map = {model: rank for rank, model in enumerate(ranked["Model"], start=1)}
    metrics["MAE_Rank"] = metrics["Model"].map(rank_map)
    return metrics.sort_values(
        ["Eligible_For_Ranking", "MAE"], ascending=[False, True]
    ).reset_index(drop=True)


def save_outputs(
    frame: pd.DataFrame,
    coverage: pd.DataFrame,
    quality: dict[str, object],
    sequence_qa: dict[str, object],
    split_info: dict[str, object],
    test_mask: np.ndarray,
    predictions: dict[str, np.ndarray],
    metrics: pd.DataFrame,
    models: dict[str, object],
    model_metadata: dict[str, object],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    prediction_frame = frame.loc[
        test_mask,
        [
            "part_number",
            "origin_date",
            "target_date",
            "target",
            "plan_d3_reference",
            "moving_average_3d",
        ],
    ].copy()
    prediction_frame = prediction_frame.rename(columns={"target": "actual"})
    for model_name, prediction in predictions.items():
        prediction_frame[model_name] = prediction

    metrics.to_csv(OUTPUT_DIR / "metrics_test.csv", index=False, encoding="utf-8-sig")
    prediction_frame.to_csv(
        OUTPUT_DIR / "predictions_test.csv", index=False, encoding="utf-8-sig"
    )
    coverage.to_csv(
        OUTPUT_DIR / "part_sequence_coverage.csv", index=False, encoding="utf-8-sig"
    )

    quality_rows: list[dict[str, object]] = []
    for key, value in {**quality, **sequence_qa, **split_info}.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                quality_rows.append(
                    {"Metric": f"{key}.{nested_key}", "Value": nested_value}
                )
        else:
            quality_rows.append({"Metric": key, "Value": value})
    pd.DataFrame(quality_rows).to_csv(
        OUTPUT_DIR / "data_quality_summary.csv", index=False, encoding="utf-8-sig"
    )

    joblib.dump(models["tree_preprocessor"], MODEL_DIR / "tree_preprocessor.joblib")
    models["xgboost"].save_model(MODEL_DIR / "xgboost.json")
    (MODEL_DIR / "lightgbm.txt").write_text(
        models["lightgbm"].booster_.model_to_string(), encoding="utf-8"
    )
    models["catboost"].save_model(str(MODEL_DIR / "catboost.cbm"))
    torch.save(
        {
            "state_dict": models["lstm"].state_dict(),
            "sequence_length": SEQUENCE_LENGTH,
            "sequence_features": SEQUENCE_FEATURES,
            "calendar_features": CALENDAR_FEATURES,
            "part_to_id": model_metadata["part_to_id"],
            **model_metadata["lstm"],
        },
        MODEL_DIR / "lstm.pt",
    )

    winner = metrics.loc[metrics["MAE_Rank"] == 1].iloc[0]
    run_metadata = {
        "winner": winner["Model"],
        "winner_primary_metric": "MAE",
        "winner_mae": float(winner["MAE"]),
        "winner_rmse": float(winner["RMSE"]),
        "winner_wape_pct": float(winner["WAPE_pct"]),
        "winner_forecast_bias": float(winner["Forecast_Bias"]),
        "winner_r2": float(winner["R2"]),
        "quality": quality,
        "sequence": sequence_qa,
        "split": split_info,
        "model": model_metadata,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    display_columns = ["Model", "MAE", "RMSE", "WAPE_pct", "Forecast_Bias", "R2"]
    report_table = markdown_table(metrics[display_columns].round(4))
    report = f"""# 최종 D+3 수요예측 모델 평가

## 결론

- 최종 1위 모델은 **{winner['Model']}**이다.
- 1위 선정 기준은 공통 테스트 표본의 **MAE 최소화**이다.
- Part115는 유효한 3일 시퀀스가 {int(coverage.loc[coverage['part_number'] == 'Part 115', 'valid_sequence_count'].iloc[0])}개로 최소 기준 {MIN_SEQUENCE_COUNT}개 미만이어서 모든 모델에서 제외했다.

## 공통 테스트 성능

{report_table}

## 평가 조건

- 모델: XGBoost, LightGBM, CatBoost, LSTM, 3일 이동평균
- 대상: {sequence_qa['eligible_parts']}개 부품, 제외: {', '.join(sequence_qa['excluded_parts'])}
- 시퀀스: 연속 {SEQUENCE_LENGTH}일 입력, 마지막 입력일 기준 정확히 D+{HORIZON_DAYS} 실제 수량
- 시간 분할: 학습 70%, 검증 10%, 테스트 20%의 목표 날짜 순서
- 테스트 기간: {split_info['test_target_start']}~{split_info['test_target_end']}
- 테스트 표본: {split_info['test_rows']:,}개, 활동 부품: {split_info['test_parts']}개
- D, D+1, D+2, D+3, D+4는 시간대별 수량 합계 사용
- 계획량 0과 실제량 0은 삭제하지 않음
"""
    (OUTPUT_DIR / "FINAL_MODEL_EVALUATION_REPORT.md").write_text(
        report, encoding="utf-8"
    )


def main() -> None:
    set_seed()
    daily, quality = load_daily_data()
    frame, sequences, coverage, sequence_qa = build_sequences(daily)
    train_mask, validation_mask, test_mask, split_info = make_split_masks(frame)
    predictions, models, model_metadata = train_and_predict(
        frame, sequences, train_mask, validation_mask, test_mask
    )

    prediction_frame = frame.loc[test_mask, ["part_number", "target"]].rename(
        columns={"target": "actual"}
    )
    metrics = make_metrics(prediction_frame, predictions)
    save_outputs(
        frame,
        coverage,
        quality,
        sequence_qa,
        split_info,
        test_mask,
        predictions,
        metrics,
        models,
        model_metadata,
    )
    print("\nFinal test metrics", flush=True)
    print(metrics.to_string(index=False), flush=True)
    winner = metrics.loc[metrics["MAE_Rank"] == 1].iloc[0]
    print(
        f"\nWINNER={winner['Model']} MAE={winner['MAE']:.4f} "
        f"RMSE={winner['RMSE']:.4f} WAPE={winner['WAPE_pct']:.4f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
