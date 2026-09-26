from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "Dataset_전자부품(배터리팩) 예지보전 AI 데이터셋" / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final_track_b"
MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"
MPL_DIR = OUTPUT_DIR / ".matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42
CALIBRATION_QUANTILE = 0.999
FINAL_HOLDOUT_FRACTION = 0.20
VALIDATION_FRACTION = 0.20

RAW_COLUMNS = [
    "PageNo",
    "Speed",
    "Length",
    "RealPower",
    "SetFrequency",
    "SetDuty",
    "SetPower",
    "GateOnTime",
    "WorkingTime",
]

FEATURE_COLUMNS = [
    "PageNo",
    "Speed",
    "Length",
    "SetFrequency",
    "SetDuty",
    "SetPower",
    "GateOnTime",
    "RealPower",
    "PreviousRealPower",
    "RealPowerDelta",
    "TimeGapSeconds",
    "PageSin",
    "PageCos",
]


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_signal(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    missing_columns = sorted(set(RAW_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{path.name}: missing columns {missing_columns}")
    frame["WorkingTime"] = pd.to_datetime(frame["WorkingTime"], errors="raise")
    return frame


def add_groups(frame: pd.DataFrame, source_file: str) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    result["source_file"] = source_file
    result["source_row"] = np.arange(len(result), dtype=int)
    result["cycle_local"] = result["PageNo"].eq(1).cumsum().astype(int)
    result["group_id"] = source_file + "::" + result["cycle_local"].astype(str)
    return result


def complete_cycles(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    accepted = []
    rejected: list[dict] = []
    for group_id, group in frame.groupby("group_id", sort=False):
        ordered = group.sort_values("PageNo").copy()
        valid = len(ordered) == 39 and ordered["PageNo"].tolist() == list(range(1, 40))
        if valid:
            accepted.append(ordered)
        else:
            rejected.append(
                {
                    "group_id": str(group_id),
                    "rows": int(len(ordered)),
                    "page_min": int(ordered["PageNo"].min()),
                    "page_max": int(ordered["PageNo"].max()),
                }
            )
    if not accepted:
        raise ValueError("No complete 39-row welding cycles were found")
    return pd.concat(accepted, ignore_index=True), rejected


def audit_frame(frame: pd.DataFrame, name: str) -> dict:
    time_values = frame["WorkingTime"]
    return {
        "file": name,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "missing_cells": int(frame[RAW_COLUMNS].isna().sum().sum()),
        "duplicate_rows": int(frame[RAW_COLUMNS].duplicated().sum()),
        "invalid_time_rows": int(time_values.isna().sum()),
        "raw_time_monotonic": bool(time_values.is_monotonic_increasing),
        "nonpositive_time_diffs": int((time_values.diff().dt.total_seconds() <= 0).sum()),
        "page_min": int(frame["PageNo"].min()),
        "page_max": int(frame["PageNo"].max()),
        "cycle_count": int(frame["PageNo"].eq(1).sum()),
        "realpower_zero_rows": int(frame["RealPower"].eq(0).sum()),
        "negative_sensor_rows": int(
            (frame[["Speed", "Length", "RealPower", "SetPower", "GateOnTime"]] < 0)
            .any(axis=1)
            .sum()
        ),
    }


def split_historical_normal(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    groups = []
    for group_id, group in frame.groupby("group_id", sort=False):
        groups.append((group["WorkingTime"].min(), str(group_id), group.copy()))
    groups.sort(key=lambda item: item[0])
    cut = int(len(groups) * 0.70)
    train_groups = groups[:cut]
    calibration_groups = groups[cut:]
    train = pd.concat([item[2] for item in train_groups], ignore_index=True)
    calibration = pd.concat([item[2] for item in calibration_groups], ignore_index=True)
    return train, calibration, {
        "complete_cycles": len(groups),
        "train_cycles": len(train_groups),
        "calibration_cycles": len(calibration_groups),
        "train_rows": int(len(train)),
        "calibration_rows": int(len(calibration)),
        "train_start": str(train["WorkingTime"].min()),
        "train_end": str(train["WorkingTime"].max()),
        "calibration_start": str(calibration["WorkingTime"].min()),
        "calibration_end": str(calibration["WorkingTime"].max()),
    }


def _counts_for_stratum(size: int) -> tuple[int, int, int]:
    if size < 2:
        return size, 0, 0
    test = max(1, int(round(size * FINAL_HOLDOUT_FRACTION)))
    test = min(test, size - 1)
    remaining = size - test
    validation = 0 if remaining < 2 else max(1, int(round(size * VALIDATION_FRACTION)))
    validation = min(validation, max(0, remaining - 1))
    train = size - validation - test
    return train, validation, test


def assign_locked_splits(test_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_table = (
        test_frame.groupby(["source_file", "group_id"], as_index=False)
        .agg(group_label=("label", "max"), rows=("label", "size"), positive_rows=("label", "sum"))
    )
    assignments = []
    rng = np.random.default_rng(SEED)
    for (source_file, group_label), stratum in group_table.groupby(
        ["source_file", "group_label"], sort=True
    ):
        indices = stratum.index.to_numpy(copy=True)
        rng.shuffle(indices)
        train_count, validation_count, test_count = _counts_for_stratum(len(indices))
        split_names = (
            ["train"] * train_count
            + ["validation"] * validation_count
            + ["test"] * test_count
        )
        for index, split_name in zip(indices.tolist(), split_names):
            row = group_table.loc[index]
            assignments.append(
                {
                    "source_file": str(source_file),
                    "group_id": str(row["group_id"]),
                    "group_label": int(group_label),
                    "rows": int(row["rows"]),
                    "positive_rows": int(row["positive_rows"]),
                    "split": split_name,
                }
            )
    assignment_frame = pd.DataFrame(assignments).sort_values(["source_file", "group_id"])
    merged = test_frame.merge(assignment_frame[["group_id", "split"]], on="group_id", how="left")
    if merged["split"].isna().any():
        raise RuntimeError("Some labeled test groups were not assigned")
    return merged, assignment_frame


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy().reset_index(drop=True)
    group_key = work["group_id"]
    previous = work.groupby(group_key, sort=False)["RealPower"].shift(1)
    previous = previous.fillna(work["RealPower"])
    time_gap = work.groupby(group_key, sort=False)["WorkingTime"].diff().dt.total_seconds()
    time_gap = time_gap.clip(lower=0.0, upper=120.0).fillna(0.0)
    page_angle = 2.0 * np.pi * (work["PageNo"].astype(float) - 1.0) / 39.0
    features = pd.DataFrame(
        {
            "PageNo": work["PageNo"].astype(float),
            "Speed": work["Speed"].astype(float),
            "Length": work["Length"].astype(float),
            "SetFrequency": work["SetFrequency"].astype(float),
            "SetDuty": work["SetDuty"].astype(float),
            "SetPower": work["SetPower"].astype(float),
            "GateOnTime": work["GateOnTime"].astype(float),
            "RealPower": work["RealPower"].astype(float),
            "PreviousRealPower": previous.astype(float),
            "RealPowerDelta": work["RealPower"].astype(float) - previous.astype(float),
            "TimeGapSeconds": time_gap.astype(float),
            "PageSin": np.sin(page_angle),
            "PageCos": np.cos(page_angle),
        }
    )
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)[FEATURE_COLUMNS]


def phase_location_scale(frame: pd.DataFrame, value_column: str = "RealPower") -> tuple[pd.Series, pd.Series]:
    median = frame.groupby("PageNo")[value_column].median()
    mad = frame.groupby("PageNo")[value_column].apply(
        lambda values: float(np.median(np.abs(values - np.median(values))))
    )
    standard = frame.groupby("PageNo")[value_column].std().fillna(0.0)
    scale = pd.concat(
        [
            (1.4826 * mad).rename("mad"),
            (0.25 * standard).rename("std_floor"),
            pd.Series(1.0, index=median.index, name="unit_floor"),
        ],
        axis=1,
    ).max(axis=1)
    return median.astype(float), scale.astype(float)


def quantile_threshold(scores: np.ndarray) -> float:
    finite = np.asarray(scores, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, CALIBRATION_QUANTILE, method="higher"))


@dataclass
class ScoredModel:
    name: str
    family: str
    model: object
    threshold: float
    feature_names: list[str]
    training_seconds: float

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        if self.family == "supervised":
            return self.model.predict_proba(engineer_features(frame))[:, 1]
        if self.name == "RobustZ":
            center = frame["PageNo"].map(self.model["median"]).to_numpy(float)
            scale = frame["PageNo"].map(self.model["scale"]).to_numpy(float)
            return np.abs((frame["RealPower"].to_numpy(float) - center) / scale)
        if self.name == "LightGBMResidual":
            features = residual_features(frame, self.model["phase_median"])
            predicted = self.model["regressor"].predict(features)
            residual = np.abs(frame["RealPower"].to_numpy(float) - predicted)
            center = frame["PageNo"].map(self.model["residual_median"]).to_numpy(float)
            scale = frame["PageNo"].map(self.model["residual_scale"]).to_numpy(float)
            return np.abs(residual - center) / scale
        raise ValueError(f"Unsupported model {self.name}")

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (self.score(frame) >= self.threshold).astype(int)


def residual_features(frame: pd.DataFrame, phase_median: pd.Series) -> pd.DataFrame:
    work = frame.copy().reset_index(drop=True)
    previous = work.groupby("group_id", sort=False)["RealPower"].shift(1)
    default_previous = work["PageNo"].sub(1).replace(0, 39).map(phase_median)
    previous = previous.where(work["PageNo"].ne(1), default_previous).fillna(default_previous)
    time_gap = (
        work.groupby("group_id", sort=False)["WorkingTime"]
        .diff()
        .dt.total_seconds()
        .clip(lower=0.0, upper=120.0)
        .fillna(0.0)
    )
    page_angle = 2.0 * np.pi * (work["PageNo"].to_numpy(float) - 1.0) / 39.0
    return pd.DataFrame(
        {
            "PageNo": work["PageNo"].astype(int),
            "Speed": work["Speed"].astype(float),
            "Length": work["Length"].astype(float),
            "SetPower": work["SetPower"].astype(float),
            "GateOnTime": work["GateOnTime"].astype(float),
            "PreviousPower": previous.astype(float),
            "TimeGap": time_gap.astype(float),
            "PageSin": np.sin(page_angle),
            "PageCos": np.cos(page_angle),
        }
    )


def fit_unsupervised(train: pd.DataFrame, calibration: pd.DataFrame) -> list[ScoredModel]:
    models: list[ScoredModel] = []

    started = time.perf_counter()
    median, scale = phase_location_scale(train)
    calibration_center = calibration["PageNo"].map(median).to_numpy(float)
    calibration_scale = calibration["PageNo"].map(scale).to_numpy(float)
    calibration_scores = np.abs(
        (calibration["RealPower"].to_numpy(float) - calibration_center) / calibration_scale
    )
    models.append(
        ScoredModel(
            name="RobustZ",
            family="unsupervised",
            model={"median": median, "scale": scale},
            threshold=quantile_threshold(calibration_scores),
            feature_names=["PageNo", "RealPower"],
            training_seconds=time.perf_counter() - started,
        )
    )

    started = time.perf_counter()
    phase_median, _ = phase_location_scale(train)
    features = residual_features(train, phase_median)
    target = train["RealPower"].to_numpy(float)
    internal_cut = int(len(train) * 0.85)
    regressor = lgb.LGBMRegressor(
        objective="huber",
        n_estimators=500,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    regressor.fit(
        features.iloc[:internal_cut],
        target[:internal_cut],
        eval_X=features.iloc[internal_cut:],
        eval_y=target[internal_cut:],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    train_residual = np.abs(target - regressor.predict(features))
    residual_frame = train[["PageNo"]].copy()
    residual_frame["Residual"] = train_residual
    residual_median, residual_scale = phase_location_scale(residual_frame, "Residual")
    payload = {
        "phase_median": phase_median,
        "regressor": regressor,
        "residual_median": residual_median,
        "residual_scale": residual_scale,
    }
    temporary = ScoredModel(
        name="LightGBMResidual",
        family="unsupervised",
        model=payload,
        threshold=0.0,
        feature_names=list(features.columns),
        training_seconds=0.0,
    )
    temporary.threshold = quantile_threshold(temporary.score(calibration))
    temporary.training_seconds = time.perf_counter() - started
    models.append(temporary)
    return models


def contiguous_regions(values: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(values, dtype=int)
    padded = np.pad(values, (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def event_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    work = frame.copy().reset_index(drop=True)
    work["prediction"] = prediction
    event_delays_rows: list[int] = []
    event_delays_seconds: list[float] = []
    event_total = 0
    event_detected = 0
    false_alarm_events = 0
    for _, group in work.groupby("group_id", sort=False):
        group = group.sort_values("PageNo").reset_index(drop=True)
        truth = group["label"].to_numpy(int)
        predicted = group["prediction"].to_numpy(int)
        for start, end in contiguous_regions(truth):
            event_total += 1
            detected_offsets = np.flatnonzero(predicted[start : end + 1] == 1)
            if len(detected_offsets):
                event_detected += 1
                detected_index = start + int(detected_offsets[0])
                event_delays_rows.append(detected_index - start)
                delay_seconds = (
                    group.loc[detected_index, "WorkingTime"] - group.loc[start, "WorkingTime"]
                ).total_seconds()
                event_delays_seconds.append(float(max(0.0, delay_seconds)))
        normal_positive = ((truth == 0) & (predicted == 1)).astype(int)
        false_alarm_events += len(contiguous_regions(normal_positive))
    return {
        "event_total": event_total,
        "event_detected": event_detected,
        "event_recall": float(event_detected / event_total) if event_total else 0.0,
        "missed_events": event_total - event_detected,
        "false_alarm_events": false_alarm_events,
        "mean_detection_delay_rows": float(np.mean(event_delays_rows)) if event_delays_rows else None,
        "max_detection_delay_rows": int(max(event_delays_rows)) if event_delays_rows else None,
        "mean_detection_delay_seconds": float(np.mean(event_delays_seconds)) if event_delays_seconds else None,
        "max_detection_delay_seconds": float(max(event_delays_seconds)) if event_delays_seconds else None,
    }


def binary_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> tuple[dict, dict]:
    truth = frame["label"].to_numpy(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, prediction, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    result = {
        "rows": int(len(frame)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        **event_metrics(frame, prediction),
    }
    report = classification_report(
        truth,
        prediction,
        labels=[0, 1],
        target_names=["normal", "anomaly"],
        output_dict=True,
        zero_division=0,
    )
    return result, report


def threshold_candidates(scores: np.ndarray) -> np.ndarray:
    quantiles = np.linspace(0.01, 0.99, 199)
    candidates = np.unique(np.quantile(scores, quantiles))
    return np.unique(np.r_[0.0, candidates, 0.5, 1.0])


def choose_supervised_threshold(validation: pd.DataFrame, scores: np.ndarray) -> float:
    best_threshold = 0.5
    best_key = None
    for threshold in threshold_candidates(scores):
        prediction = (scores >= threshold).astype(int)
        metrics, _ = binary_metrics(validation, prediction)
        key = (
            -metrics["missed_events"],
            -metrics["fn"],
            -metrics["false_alarm_events"],
            metrics["f1"],
            -metrics["fp"],
            float(threshold),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def fit_supervised(
    historical_train: pd.DataFrame,
    labeled_train: pd.DataFrame,
    validation: pd.DataFrame,
) -> list[ScoredModel]:
    normal_history = historical_train.copy()
    normal_history["label"] = 0
    training = pd.concat([normal_history, labeled_train], ignore_index=True)
    x_train = engineer_features(training)
    y_train = training["label"].to_numpy(int)
    x_validation = engineer_features(validation)
    models: list[ScoredModel] = []

    started = time.perf_counter()
    logistic = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=SEED,
                    solver="liblinear",
                ),
            ),
        ]
    )
    logistic.fit(x_train, y_train)
    threshold = choose_supervised_threshold(validation, logistic.predict_proba(x_validation)[:, 1])
    models.append(
        ScoredModel(
            name="LogisticClassifier",
            family="supervised",
            model=logistic,
            threshold=threshold,
            feature_names=FEATURE_COLUMNS,
            training_seconds=time.perf_counter() - started,
        )
    )

    started = time.perf_counter()
    forest = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )
    forest.fit(x_train, y_train)
    threshold = choose_supervised_threshold(validation, forest.predict_proba(x_validation)[:, 1])
    models.append(
        ScoredModel(
            name="RandomForestClassifier",
            family="supervised",
            model=forest,
            threshold=threshold,
            feature_names=FEATURE_COLUMNS,
            training_seconds=time.perf_counter() - started,
        )
    )
    return models


def feature_importance(models: list[ScoredModel]) -> pd.DataFrame:
    rows = []
    for model in models:
        if model.name == "LogisticClassifier":
            values = np.abs(model.model.named_steps["classifier"].coef_[0])
        elif model.name == "RandomForestClassifier":
            values = model.model.feature_importances_
        else:
            continue
        total = float(np.sum(values)) or 1.0
        for feature, value in zip(model.feature_names, values):
            rows.append(
                {
                    "model": model.name,
                    "feature": feature,
                    "importance": float(value),
                    "normalized_importance": float(value / total),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "normalized_importance"], ascending=[True, False])


def evaluate_models(
    models: list[ScoredModel],
    validation: pd.DataFrame,
    holdout: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    metric_rows = []
    prediction_rows = []
    reports: dict[str, dict] = {}
    for model in models:
        reports[model.name] = {}
        for split_name, frame in [("validation", validation), ("test", holdout)]:
            scores = model.score(frame)
            predictions = (scores >= model.threshold).astype(int)
            metrics, report = binary_metrics(frame, predictions)
            reports[model.name][split_name] = report
            metric_rows.append(
                {
                    "model": model.name,
                    "family": model.family,
                    "split": split_name,
                    "threshold": model.threshold,
                    "training_seconds": model.training_seconds,
                    **metrics,
                }
            )
            block = frame[
                ["source_file", "source_row", "group_id", "PageNo", "WorkingTime", "label"]
            ].copy()
            block.insert(0, "split", split_name)
            block.insert(0, "model", model.name)
            block["score"] = scores
            block["threshold"] = model.threshold
            block["prediction"] = predictions
            prediction_rows.append(block)
    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True), reports


def select_winner(metrics: pd.DataFrame) -> str:
    validation = metrics[metrics["split"].eq("validation")].copy()
    validation = validation.sort_values(
        ["missed_events", "fn", "false_alarm_events", "f1", "fp", "training_seconds"],
        ascending=[True, True, True, False, True, True],
        kind="stable",
    )
    return str(validation.iloc[0]["model"])


def plot_confusion(metrics: pd.DataFrame) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for row in metrics[metrics["split"].eq("test")].itertuples():
        matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]])
        fig, ax = plt.subplots(figsize=(5.6, 4.6))
        image = ax.imshow(matrix, cmap="Blues")
        for (y, x), value in np.ndenumerate(matrix):
            ax.text(x, y, f"{value:,}", ha="center", va="center", fontsize=13)
        ax.set_xticks([0, 1], ["Pred normal", "Pred anomaly"])
        ax.set_yticks([0, 1], ["True normal", "True anomaly"])
        ax.set_title(f"{row.model} - locked holdout")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"confusion_{row.model}.png", dpi=180)
        plt.close(fig)


def plot_comparison(metrics: pd.DataFrame) -> None:
    table = metrics[metrics["split"].eq("test")].reset_index(drop=True)
    x = np.arange(len(table))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 6))
    for offset, column in enumerate(["precision", "recall", "f1"]):
        ax.bar(x + (offset - 1) * width, table[column], width, label=column.title())
    ax.set_xticks(x, table["model"], rotation=15, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Track B models on the same locked holdout")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "model_comparison_locked_holdout.png", dpi=180)
    plt.close(fig)


def plot_feature_importance(importance: pd.DataFrame) -> None:
    for model, group in importance.groupby("model"):
        top = group.nlargest(12, "normalized_importance").sort_values("normalized_importance")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top["feature"], top["normalized_importance"], color="#2c7fb8")
        ax.set_xlabel("Normalized importance")
        ax.set_title(f"{model} feature importance")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"feature_importance_{model}.png", dpi=180)
        plt.close(fig)


def plot_score_timelines(predictions: pd.DataFrame) -> None:
    test_predictions = predictions[predictions["split"].eq("test")].copy()
    for (model, source_file), group in test_predictions.groupby(["model", "source_file"]):
        group = group.sort_values("source_row")
        threshold = float(group["threshold"].iloc[0])
        fig, ax = plt.subplots(figsize=(11, 4.5))
        score_label_used = False
        truth_label_used = False
        for _, cycle in group.groupby("group_id", sort=False):
            cycle = cycle.sort_values("source_row")
            x = cycle["source_row"].to_numpy()
            score = cycle["score"].to_numpy(float)
            truth = cycle["label"].to_numpy(int)
            ax.plot(
                x,
                score,
                color="#225ea8",
                linewidth=1.2,
                label="Anomaly score" if not score_label_used else None,
            )
            score_label_used = True
            if truth.any():
                ymax = max(float(np.nanmax(score)), threshold) * 1.05
                ax.fill_between(
                    x,
                    0,
                    ymax,
                    where=truth.astype(bool),
                    color="#fb6a4a",
                    alpha=0.22,
                    label="True anomaly" if not truth_label_used else None,
                )
                truth_label_used = True
        ax.axhline(threshold, color="#d7301f", linestyle="--", label=f"Threshold {threshold:.4g}")
        ax.set_title(f"{model} - {source_file} locked cycles")
        ax.set_xlabel("Original file row")
        ax.set_ylabel("Score")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"score_{model}_{source_file}.png", dpi=180)
        plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in selected.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append("" if pd.isna(value) else f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def write_report(
    metrics: pd.DataFrame,
    winner: str,
    split_summary: dict,
    assignment_frame: pd.DataFrame,
    importance: pd.DataFrame,
) -> None:
    test_metrics = metrics[metrics["split"].eq("test")].copy()
    validation_metrics = metrics[metrics["split"].eq("validation")].copy()
    winner_test = test_metrics[test_metrics["model"].eq(winner)].iloc[0]
    split_counts = (
        assignment_frame.groupby(["split", "group_label"], as_index=False)
        .agg(groups=("group_id", "size"), rows=("rows", "sum"), positive_rows=("positive_rows", "sum"))
    )
    lines = [
        "# 트랙 B 용접 설비 예지보전 최종 검증 보고서",
        "",
        f"최종 선정 모델: **{winner}**",
        "",
        "## 고정 평가 원칙",
        "",
        "- 데이터 행을 먼저 섞지 않고 39행 용접 사이클을 그룹으로 만든 뒤 train/validation/test를 분할했다.",
        "- test 그룹은 잠금 홀드아웃이며 모델 적합, 임계값 결정, 모델 선택에 사용하지 않았다.",
        "- 지도 모델 임계값과 최종 모델 선택은 validation에서만 결정했다.",
        "- 비지도 모델은 기존 정상 Training_Data의 앞 70%로 학습하고 뒤 30% 정상 구간으로 임계값을 고정했다.",
        "- 모든 모델은 동일한 test 행과 동일한 라벨로 평가했다.",
        "- 선택 순서는 이벤트 누락, FN, 오경보 이벤트, F1, FP, 학습시간 순이다.",
        "",
        "## 분할 요약",
        "",
        markdown_table(split_counts, ["split", "group_label", "groups", "rows", "positive_rows"]),
        "",
        "## 모델 선택용 validation 결과",
        "",
        markdown_table(
            validation_metrics,
            ["model", "family", "precision", "recall", "f1", "missed_events", "false_alarm_events"],
        ),
        "",
        "## 잠금 홀드아웃 결과",
        "",
        markdown_table(
            test_metrics,
            [
                "model",
                "family",
                "precision",
                "recall",
                "f1",
                "tn",
                "fp",
                "fn",
                "tp",
                "event_recall",
                "mean_detection_delay_seconds",
            ],
        ),
        "",
        "## 최종 모델 해석",
        "",
        f"- 잠금 홀드아웃 Precision: {winner_test['precision']:.4f}",
        f"- 잠금 홀드아웃 Recall: {winner_test['recall']:.4f}",
        f"- 잠금 홀드아웃 F1: {winner_test['f1']:.4f}",
        f"- 이벤트 탐지: {int(winner_test['event_detected'])}/{int(winner_test['event_total'])}",
        f"- 오경보 이벤트: {int(winner_test['false_alarm_events'])}",
        "",
        "## 주요 특징",
        "",
    ]
    for model, group in importance.groupby("model"):
        top = group.nlargest(5, "normalized_importance")
        features = ", ".join(
            f"{row.feature} ({row.normalized_importance:.3f})" for row in top.itertuples()
        )
        lines.append(f"- {model}: {features}")
    lines.extend(
        [
            "",
            "## 한계와 운영 주의사항",
            "",
            "- 현재 네 시험 파일은 이전 실험에서 이미 사용됐으므로 이번 잠금 홀드아웃은 앞으로의 변경을 통제하는 기준이지 완전히 새로운 외부 검증 데이터는 아니다.",
            "- NG 유형은 03 파일의 고립 이상과 04 파일의 연속 이상 두 유형뿐이다. 새로운 고장 모드에 대한 성능은 보장하지 않는다.",
            "- 용접 사이클 단위 분할로 행 누수는 막았지만 동일 파일의 다른 사이클이 여러 세트에 포함되므로 파일 고유 분포의 영향이 남을 수 있다.",
            "- 현장 배포 전에는 신규 시점에서 수집한 OK/NG 파일로 완전 외부 검증하고 점검 비용에 맞춰 임계값을 재승인해야 한다.",
            "- 기존 N-HiTS 가이드북 재현 결과는 별도 legacy 실험이며 이 최종 모델 선택에는 사용하지 않았다.",
            "",
            "## 학습 데이터 분할",
            "",
            f"- 정상 학습 사이클: {split_summary['train_cycles']}",
            f"- 정상 임계값 보정 사이클: {split_summary['calibration_cycles']}",
        ]
    )
    (OUTPUT_DIR / "final_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_data_quality_report(
    audit: pd.DataFrame,
    historical_split: dict,
    historical_rejected: list[dict],
    assignment_frame: pd.DataFrame,
) -> None:
    split_summary = (
        assignment_frame.groupby(["split", "group_label"], as_index=False)
        .agg(groups=("group_id", "size"), rows=("rows", "sum"), positive_rows=("positive_rows", "sum"))
    )
    lines = [
        "# 트랙 B 데이터 품질 및 분할 보고서",
        "",
        "## 원본 파일 검사",
        "",
        markdown_table(
            audit,
            [
                "file",
                "rows",
                "missing_cells",
                "duplicate_rows",
                "nonpositive_time_diffs",
                "cycle_count",
                "realpower_zero_rows",
                "negative_sensor_rows",
            ],
        ),
        "",
        "## 처리 정책",
        "",
        "- WorkingTime은 엄격한 날짜시간 변환을 통과해야 한다.",
        "- PageNo 1~39가 정확히 한 번씩 존재하는 완전한 용접 사이클만 사용한다.",
        "- 원본 행 순서가 시간순이 아니어도 사이클 시작시각으로 그룹을 정렬하되 행을 서로 다른 사이클에 섞지 않는다.",
        "- 결측치, 완전 중복행, 음수 센서값은 현재 파일에서 발견되지 않았다.",
        "- RealPower가 0인 39행 사이클 1개는 임의 삭제하지 않고 품질 플래그로 기록해 모델 강건성 검증에 포함했다.",
        "- SetFrequency와 SetDuty는 상수열이지만 데이터 사전 보존과 지도 모델 비교를 위해 입력 후보에 유지했다. 중요도는 별도 기록한다.",
        "",
        "## 처리 결과",
        "",
        f"- 정상 이력 완전 사이클: {historical_split['complete_cycles']}",
        f"- 정상 학습 사이클: {historical_split['train_cycles']}",
        f"- 정상 임계값 보정 사이클: {historical_split['calibration_cycles']}",
        f"- 불완전하여 제외한 정상 사이클: {len(historical_rejected)}",
        "",
        "## 지도학습 그룹 분할",
        "",
        markdown_table(split_summary, ["split", "group_label", "groups", "rows", "positive_rows"]),
        "",
        "group_label 0은 정상 사이클, 1은 한 행 이상 이상 라벨이 포함된 사이클이다.",
    ]
    (OUTPUT_DIR / "data_quality_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    data_files = [
        DATA_ROOT / "raw_data" / "train" / "Training_Data.csv",
        *sorted((DATA_ROOT / "raw_data" / "test").glob("*.csv")),
        *sorted((DATA_ROOT / "preprocessed" / "test").glob("*Label.csv")),
    ]

    audit_rows = []
    historical_raw = read_signal(data_files[0])
    audit_rows.append(audit_frame(historical_raw, data_files[0].name))
    historical = add_groups(historical_raw, "Training_Data")
    historical, historical_rejected = complete_cycles(historical)
    historical_train, historical_calibration, historical_split = split_historical_normal(historical)

    test_blocks = []
    for path in sorted((DATA_ROOT / "raw_data" / "test").glob("*.csv")):
        raw = read_signal(path)
        audit_rows.append(audit_frame(raw, path.name))
        block = add_groups(raw, path.stem)
        if path.stem.endswith("NG"):
            label_path = DATA_ROOT / "preprocessed" / "test" / f"{path.stem}_Label.csv"
            labels = pd.read_csv(label_path)["label"].to_numpy(int)
        else:
            labels = np.zeros(len(block), dtype=int)
        if len(labels) != len(block):
            raise ValueError(f"{path.stem}: label length mismatch")
        block["label"] = labels
        complete, rejected = complete_cycles(block)
        if rejected:
            raise ValueError(f"{path.stem}: incomplete test cycles: {rejected}")
        test_blocks.append(complete)
    labeled = pd.concat(test_blocks, ignore_index=True)
    labeled, assignment_frame = assign_locked_splits(labeled)
    labeled_train = labeled[labeled["split"].eq("train")].copy()
    validation = labeled[labeled["split"].eq("validation")].copy()
    holdout = labeled[labeled["split"].eq("test")].copy()
    if validation["label"].nunique() != 2 or holdout["label"].nunique() != 2:
        raise RuntimeError("Validation and test must each contain normal and anomaly rows")

    unsupervised_models = fit_unsupervised(historical_train, historical_calibration)
    supervised_models = fit_supervised(historical_train, labeled_train, validation)
    models = [*supervised_models, *unsupervised_models]
    metrics, predictions, reports = evaluate_models(models, validation, holdout)
    winner = select_winner(metrics)
    importance = feature_importance(supervised_models)

    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False, encoding="utf-8-sig")
    assignment_frame.to_csv(OUTPUT_DIR / "split_manifest.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")
    audit_frame_result = pd.DataFrame(audit_rows)
    audit_frame_result.to_csv(OUTPUT_DIR / "data_quality.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "classification_reports.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for model in models:
        joblib.dump(
            {
                "name": model.name,
                "family": model.family,
                "model": model.model,
                "threshold": model.threshold,
                "feature_names": model.feature_names,
            },
            MODEL_DIR / f"{model.name}.joblib",
        )

    plot_confusion(metrics)
    plot_comparison(metrics)
    plot_feature_importance(importance)
    plot_score_timelines(predictions)
    write_report(metrics, winner, historical_split, assignment_frame, importance)
    write_data_quality_report(
        audit_frame_result,
        historical_split,
        historical_rejected,
        assignment_frame,
    )

    test_metrics = metrics[metrics["split"].eq("test")].set_index("model")
    manifest = {
        "pipeline_version": "track-b-final-v1",
        "seed": SEED,
        "winner_selected_on_validation": winner,
        "winner_locked_holdout_metrics": test_metrics.loc[winner].to_dict(),
        "historical_split": historical_split,
        "historical_rejected_cycles": historical_rejected,
        "split_policy": {
            "unit": "39-row welding cycle",
            "train_fraction_approx": 0.60,
            "validation_fraction_approx": VALIDATION_FRACTION,
            "locked_test_fraction_approx": FINAL_HOLDOUT_FRACTION,
            "stratification": "source_file and cycle-level anomaly presence",
            "selection_data": "validation only",
            "test_usage": "final reporting only",
        },
        "threshold_policy": {
            "supervised": "validation: missed events, FN, false alarms, F1, FP",
            "unsupervised": f"normal calibration {CALIBRATION_QUANTILE:.3%} quantile",
        },
        "data_sha256": {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in data_files},
        "pipeline_sha256": sha256(Path(__file__)),
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(f"Winner selected on validation: {winner}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
