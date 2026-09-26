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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "track_b_final_v2"
MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"
MPL_DIR = OUTPUT_DIR / ".matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


SEED = 42
NORMAL_CALIBRATION_QUANTILE = 0.999
DEVELOPMENT_TRAIN_FRACTION = 0.70

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

CURRENT_FEATURES = [
    "PageNo",
    "PageSin",
    "PageCos",
    "Speed",
    "Length",
    "SetPower",
    "GateOnTime",
    "RealPower",
    "PreviousRealPower",
    "RealPowerDelta",
    "PhaseSignedZ",
    "PhaseAbsZ",
    "RelativePowerError",
    "TimeGapSeconds",
]

HISTORY_FEATURES = [
    "PageNo",
    "PageSin",
    "PageCos",
    "SetPower",
    "GateOnTime",
    "PreviousRealPower",
    "PreviousPhaseSignedZ",
    "PreviousPhaseAbsZ",
    "TimeGapSeconds",
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
    missing = sorted(set(RAW_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    frame["WorkingTime"] = pd.to_datetime(frame["WorkingTime"], errors="raise")
    return frame


def add_identity(frame: pd.DataFrame, source_file: str) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    result["source_file"] = source_file
    result["source_row"] = np.arange(len(result), dtype=int)
    result["cycle_local"] = result["PageNo"].eq(1).cumsum().astype(int)
    result["group_id"] = source_file + "::cycle_" + result["cycle_local"].astype(str)
    return result


def require_complete_cycles(frame: pd.DataFrame) -> pd.DataFrame:
    accepted = []
    for group_id, group in frame.groupby("group_id", sort=False):
        ordered = group.sort_values("source_row")
        valid = len(ordered) == 39 and ordered["PageNo"].tolist() == list(range(1, 40))
        if not valid:
            raise ValueError(f"Incomplete welding cycle: {group_id}")
        accepted.append(ordered)
    return pd.concat(accepted, ignore_index=True)


def load_test_file(name: str) -> pd.DataFrame:
    path = DATA_ROOT / "raw_data" / "test" / f"{name}.csv"
    frame = require_complete_cycles(add_identity(read_signal(path), name))
    if name.endswith("NG"):
        label_path = DATA_ROOT / "preprocessed" / "test" / f"{name}_Label.csv"
        labels = pd.read_csv(label_path)["label"].to_numpy(dtype=int)
    else:
        labels = np.zeros(len(frame), dtype=int)
    if len(labels) != len(frame):
        raise ValueError(f"{name}: label length mismatch")
    frame["label"] = labels
    return frame


def split_historical_normal(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    groups = [group.copy() for _, group in frame.groupby("group_id", sort=False)]
    cut = int(len(groups) * 0.70)
    train = pd.concat(groups[:cut], ignore_index=True)
    calibration = pd.concat(groups[cut:], ignore_index=True)
    return train, calibration, {
        "complete_cycles": len(groups),
        "train_cycles": cut,
        "calibration_cycles": len(groups) - cut,
        "train_rows": int(len(train)),
        "calibration_rows": int(len(calibration)),
    }


def split_development_by_cycle(
    frames: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    manifest_rows: list[dict] = []
    for frame in frames:
        source_file = str(frame["source_file"].iloc[0])
        groups = [group.copy() for _, group in frame.groupby("group_id", sort=False)]
        cut = max(1, min(len(groups) - 1, int(len(groups) * DEVELOPMENT_TRAIN_FRACTION)))
        for index, group in enumerate(groups):
            split = "development_train" if index < cut else "development_validation"
            (train_parts if index < cut else validation_parts).append(group)
            manifest_rows.append(
                {
                    "source_file": source_file,
                    "group_id": str(group["group_id"].iloc[0]),
                    "split": split,
                    "rows": int(len(group)),
                    "positive_rows": int(group["label"].sum()),
                    "group_label": int(group["label"].max()),
                }
            )
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(validation_parts, ignore_index=True),
        pd.DataFrame(manifest_rows),
    )


def phase_reference(frame: pd.DataFrame) -> dict[str, pd.Series]:
    median = frame.groupby("PageNo")["RealPower"].median().astype(float)
    mad = frame.groupby("PageNo")["RealPower"].apply(
        lambda values: float(np.median(np.abs(values - np.median(values))))
    )
    std = frame.groupby("PageNo")["RealPower"].std().fillna(0.0)
    scale = pd.concat(
        [
            (1.4826 * mad).rename("mad"),
            (0.25 * std).rename("std_floor"),
            pd.Series(1.0, index=median.index, name="unit_floor"),
        ],
        axis=1,
    ).max(axis=1)
    return {"median": median, "scale": scale.astype(float)}


def make_features(frame: pd.DataFrame, reference: dict[str, pd.Series]) -> pd.DataFrame:
    work = frame.copy().reset_index(drop=True)
    phase_median = work["PageNo"].map(reference["median"]).astype(float)
    phase_scale = work["PageNo"].map(reference["scale"]).astype(float)
    signed_z = (work["RealPower"].astype(float) - phase_median) / phase_scale

    previous_power = work.groupby("group_id", sort=False)["RealPower"].shift(1)
    previous_page = work["PageNo"].sub(1).where(work["PageNo"].gt(1), 39)
    previous_default = previous_page.map(reference["median"]).astype(float)
    previous_power = previous_power.fillna(previous_default).astype(float)
    previous_center = previous_page.map(reference["median"]).astype(float)
    previous_scale = previous_page.map(reference["scale"]).astype(float)
    previous_z = (previous_power - previous_center) / previous_scale

    time_gap = (
        work.groupby("group_id", sort=False)["WorkingTime"]
        .diff()
        .dt.total_seconds()
        .clip(lower=0.0, upper=120.0)
        .fillna(0.0)
    )
    angle = 2.0 * np.pi * (work["PageNo"].to_numpy(float) - 1.0) / 39.0
    features = pd.DataFrame(
        {
            "PageNo": work["PageNo"].astype(float),
            "PageSin": np.sin(angle),
            "PageCos": np.cos(angle),
            "Speed": work["Speed"].astype(float),
            "Length": work["Length"].astype(float),
            "SetPower": work["SetPower"].astype(float),
            "GateOnTime": work["GateOnTime"].astype(float),
            "RealPower": work["RealPower"].astype(float),
            "PreviousRealPower": previous_power,
            "RealPowerDelta": work["RealPower"].astype(float) - previous_power,
            "PhaseSignedZ": signed_z,
            "PhaseAbsZ": signed_z.abs(),
            "RelativePowerError": (
                (work["RealPower"].astype(float) - phase_median).abs()
                / phase_median.abs().clip(lower=1.0)
            ),
            "PreviousPhaseSignedZ": previous_z,
            "PreviousPhaseAbsZ": previous_z.abs(),
            "TimeGapSeconds": time_gap.astype(float),
        }
    )
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def contiguous_regions(values: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(values, dtype=int), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def event_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    work = frame.copy().reset_index(drop=True)
    work["prediction"] = np.asarray(prediction, dtype=int)
    total = detected = false_alarm_events = 0
    delays_rows: list[int] = []
    delays_seconds: list[float] = []
    for _, file_frame in work.groupby("source_file", sort=False):
        ordered = file_frame.sort_values("source_row").reset_index(drop=True)
        truth = ordered["label"].to_numpy(dtype=int)
        predicted = ordered["prediction"].to_numpy(dtype=int)
        for start, end in contiguous_regions(truth):
            total += 1
            offsets = np.flatnonzero(predicted[start : end + 1] == 1)
            if len(offsets):
                detected += 1
                first = start + int(offsets[0])
                delays_rows.append(first - start)
                seconds = (ordered.loc[first, "WorkingTime"] - ordered.loc[start, "WorkingTime"]).total_seconds()
                delays_seconds.append(float(max(0.0, seconds)))
        false_alarm_mask = ((truth == 0) & (predicted == 1)).astype(int)
        false_alarm_events += len(contiguous_regions(false_alarm_mask))
    return {
        "event_total": total,
        "event_detected": detected,
        "event_recall": float(detected / total) if total else 0.0,
        "missed_events": total - detected,
        "false_alarm_events": false_alarm_events,
        "mean_detection_delay_rows": float(np.mean(delays_rows)) if delays_rows else None,
        "max_detection_delay_rows": int(max(delays_rows)) if delays_rows else None,
        "mean_detection_delay_seconds": float(np.mean(delays_seconds)) if delays_seconds else None,
    }


def calculate_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> tuple[dict, dict]:
    truth = frame["label"].to_numpy(dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, prediction, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    metrics = {
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
    return metrics, report


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    quantiles = np.linspace(0.001, 0.999, 501)
    return np.unique(np.r_[scores.min() - 1e-12, np.quantile(scores, quantiles), 0.5])


def choose_threshold(validation: pd.DataFrame, scores: np.ndarray) -> tuple[float, dict]:
    rows = []
    for threshold in candidate_thresholds(scores):
        prediction = (scores >= threshold).astype(int)
        metrics, _ = calculate_metrics(validation, prediction)
        rows.append((float(threshold), metrics))
    feasible = [
        item
        for item in rows
        if item[1]["event_recall"] >= 1.0 and item[1]["false_positive_rate"] <= 0.01
    ]
    pool = feasible if feasible else rows
    threshold, metrics = max(
        pool,
        key=lambda item: (
            item[1]["event_recall"],
            item[1]["recall"],
            item[1]["f1"],
            item[1]["precision"],
            -item[1]["false_alarm_events"],
            -item[1]["fp"],
            item[0],
        ),
    )
    return threshold, {
        "feasibility_rule_met": bool(feasible),
        "rule": "event_recall=1 and normal FPR<=1%; then recall, F1, precision, false alarms",
        "selected_validation_metrics": metrics,
    }


@dataclass
class Detector:
    name: str
    family: str
    model: object
    threshold: float
    feature_names: list[str]
    reference: dict[str, pd.Series]
    threshold_source: str
    training_seconds: float

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        features = make_features(frame, self.reference)
        if self.name == "RobustPhaseZ":
            return features["PhaseAbsZ"].to_numpy(dtype=float)
        if self.name == "IsolationForestNormal":
            return -self.model.decision_function(features[self.feature_names])
        return self.model.predict_proba(features[self.feature_names])[:, 1]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (self.score(frame) >= self.threshold).astype(int)


def fit_models(
    historical_train: pd.DataFrame,
    historical_calibration: pd.DataFrame,
    development_train: pd.DataFrame,
    development_validation: pd.DataFrame,
) -> tuple[list[Detector], dict]:
    reference = phase_reference(historical_train)
    x_train = make_features(development_train, reference)
    y_train = development_train["label"].to_numpy(dtype=int)
    models: list[Detector] = []
    threshold_audit: dict = {}

    supervised_specs = [
        (
            "LogisticCurrent",
            CURRENT_FEATURES,
            Pipeline(
                [
                    ("scale", RobustScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=3000,
                            class_weight="balanced",
                            C=0.5,
                            random_state=SEED,
                        ),
                    ),
                ]
            ),
        ),
        (
            "RandomForestCurrent",
            CURRENT_FEATURES,
            RandomForestClassifier(
                n_estimators=600,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=SEED,
                n_jobs=-1,
            ),
        ),
        (
            "LogisticHistoryOnly",
            HISTORY_FEATURES,
            Pipeline(
                [
                    ("scale", RobustScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=3000,
                            class_weight="balanced",
                            C=0.5,
                            random_state=SEED,
                        ),
                    ),
                ]
            ),
        ),
    ]
    for name, columns, estimator in supervised_specs:
        started = time.perf_counter()
        estimator.fit(x_train[columns], y_train)
        temporary = Detector(
            name=name,
            family="supervised",
            model=estimator,
            threshold=0.5,
            feature_names=list(columns),
            reference=reference,
            threshold_source="development_validation_labels",
            training_seconds=time.perf_counter() - started,
        )
        validation_scores = temporary.score(development_validation)
        threshold, audit = choose_threshold(development_validation, validation_scores)
        temporary.threshold = threshold
        threshold_audit[name] = audit
        models.append(temporary)

    calibration_features = make_features(historical_calibration, reference)
    robust_threshold = float(
        np.quantile(
            calibration_features["PhaseAbsZ"].to_numpy(dtype=float),
            NORMAL_CALIBRATION_QUANTILE,
            method="higher",
        )
    )
    models.append(
        Detector(
            name="RobustPhaseZ",
            family="unsupervised",
            model={"median": reference["median"], "scale": reference["scale"]},
            threshold=robust_threshold,
            feature_names=["PageNo", "RealPower", "PhaseAbsZ"],
            reference=reference,
            threshold_source="historical_normal_calibration_99.9pct",
            training_seconds=0.0,
        )
    )

    isolation_features = [
        "PageSin",
        "PageCos",
        "Speed",
        "Length",
        "SetPower",
        "GateOnTime",
        "RealPower",
        "PreviousRealPower",
        "RealPowerDelta",
        "PhaseSignedZ",
        "PhaseAbsZ",
    ]
    historical_features = make_features(historical_train, reference)
    started = time.perf_counter()
    isolation = IsolationForest(
        n_estimators=500,
        max_samples=min(8192, len(historical_features)),
        contamination="auto",
        random_state=SEED,
        n_jobs=-1,
    )
    isolation.fit(historical_features[isolation_features])
    calibration_scores = -isolation.decision_function(calibration_features[isolation_features])
    isolation_threshold = float(
        np.quantile(calibration_scores, NORMAL_CALIBRATION_QUANTILE, method="higher")
    )
    models.append(
        Detector(
            name="IsolationForestNormal",
            family="unsupervised",
            model=isolation,
            threshold=isolation_threshold,
            feature_names=isolation_features,
            reference=reference,
            threshold_source="historical_normal_calibration_99.9pct",
            training_seconds=time.perf_counter() - started,
        )
    )
    return models, threshold_audit


def evaluate_models(
    models: list[Detector], validation: pd.DataFrame, locked_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    metric_rows = []
    prediction_rows = []
    per_file_rows = []
    reports: dict = {}
    for detector in models:
        for split_name, frame in [("validation", validation), ("locked_test", locked_test)]:
            score = detector.score(frame)
            prediction = (score >= detector.threshold).astype(int)
            metrics, report = calculate_metrics(frame, prediction)
            metric_rows.append(
                {
                    "model": detector.name,
                    "family": detector.family,
                    "split": split_name,
                    "threshold": detector.threshold,
                    "threshold_source": detector.threshold_source,
                    "training_seconds": detector.training_seconds,
                    **metrics,
                }
            )
            reports[f"{detector.name}::{split_name}"] = report
            block = frame[
                [
                    "source_file",
                    "source_row",
                    "group_id",
                    "PageNo",
                    "WorkingTime",
                    "RealPower",
                    "label",
                ]
            ].copy()
            block["model"] = detector.name
            block["family"] = detector.family
            block["split"] = split_name
            block["score"] = score
            block["threshold"] = detector.threshold
            block["prediction"] = prediction
            prediction_rows.append(block)
            for source_file, file_frame in block.groupby("source_file", sort=False):
                source_original = frame[frame["source_file"].eq(source_file)].copy()
                file_metrics, _ = calculate_metrics(
                    source_original, file_frame["prediction"].to_numpy(dtype=int)
                )
                per_file_rows.append(
                    {
                        "model": detector.name,
                        "family": detector.family,
                        "split": split_name,
                        "source_file": source_file,
                        **file_metrics,
                    }
                )
    return (
        pd.DataFrame(metric_rows),
        pd.concat(prediction_rows, ignore_index=True),
        pd.DataFrame(per_file_rows),
        reports,
    )


def choose_winner(metrics: pd.DataFrame) -> str:
    validation = metrics[metrics["split"].eq("validation")].copy()
    validation = validation.sort_values(
        [
            "missed_events",
            "fn",
            "false_alarm_events",
            "f1",
            "fp",
            "training_seconds",
            "model",
        ],
        ascending=[True, True, True, False, True, True, True],
    )
    return str(validation.iloc[0]["model"])


def feature_importance(models: list[Detector]) -> pd.DataFrame:
    rows = []
    for detector in models:
        if detector.family != "supervised":
            continue
        estimator = detector.model
        if isinstance(estimator, Pipeline):
            values = np.abs(estimator.named_steps["model"].coef_[0])
        else:
            values = estimator.feature_importances_
        total = float(np.sum(values)) or 1.0
        for feature, value in zip(detector.feature_names, values):
            rows.append(
                {
                    "model": detector.name,
                    "feature": feature,
                    "importance": float(value),
                    "normalized_importance": float(value / total),
                }
            )
    return pd.DataFrame(rows)


def data_audit(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in frames.items():
        rows.append(
            {
                "file": name,
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
                "missing_cells": int(frame[RAW_COLUMNS].isna().sum().sum()),
                "duplicate_rows": int(frame[RAW_COLUMNS].duplicated().sum()),
                "nonpositive_time_diffs": int(
                    (frame["WorkingTime"].diff().dt.total_seconds() <= 0).sum()
                ),
                "complete_cycles": int(frame["group_id"].nunique()),
                "positive_rows": int(frame.get("label", pd.Series(dtype=int)).sum()),
                "realpower_zero_rows": int(frame["RealPower"].eq(0).sum()),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append("" if pd.isna(value) else f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def plot_results(metrics: pd.DataFrame, predictions: pd.DataFrame, importance: pd.DataFrame) -> None:
    test = metrics[metrics["split"].eq("locked_test")].reset_index(drop=True)
    x = np.arange(len(test))
    width = 0.24
    fig, ax = plt.subplots(figsize=(12, 6))
    for offset, column in enumerate(["precision", "recall", "f1"]):
        ax.bar(x + (offset - 1) * width, test[column], width, label=column.title())
    ax.set_xticks(x, test["model"], rotation=18, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Track B - independent file holdout")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "locked_test_model_comparison.png", dpi=180)
    plt.close(fig)

    for row in test.itertuples():
        matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]])
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        image = ax.imshow(matrix, cmap="Blues")
        for (y, x_pos), value in np.ndenumerate(matrix):
            ax.text(x_pos, y, f"{value:,}", ha="center", va="center", fontsize=13)
        ax.set_xticks([0, 1], ["Pred normal", "Pred anomaly"])
        ax.set_yticks([0, 1], ["True normal", "True anomaly"])
        ax.set_title(f"{row.model} - locked file holdout")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"confusion_{row.model}.png", dpi=180)
        plt.close(fig)

    for model, group in importance.groupby("model"):
        top = group.nlargest(12, "normalized_importance").sort_values("normalized_importance")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top["feature"], top["normalized_importance"], color="#2c7fb8")
        ax.set_xlabel("Normalized importance")
        ax.set_title(f"{model} feature importance")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"feature_importance_{model}.png", dpi=180)
        plt.close(fig)

    locked = predictions[predictions["split"].eq("locked_test")]
    for (model, source_file), group in locked.groupby(["model", "source_file"], sort=False):
        group = group.sort_values("source_row")
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(group["source_row"], group["score"], color="#225ea8", label="Anomaly score")
        threshold = float(group["threshold"].iloc[0])
        ax.axhline(threshold, color="#d7301f", linestyle="--", label="Threshold")
        if group["label"].any():
            ymax = max(float(group["score"].max()), threshold) * 1.05
            ax.fill_between(
                group["source_row"],
                0,
                ymax,
                where=group["label"].astype(bool),
                color="#fb6a4a",
                alpha=0.22,
                label="True anomaly",
            )
        ax.set_title(f"{model} - {source_file}")
        ax.set_xlabel("Original file row")
        ax.set_ylabel("Score")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"score_{model}_{source_file}.png", dpi=180)
        plt.close(fig)


def write_plan() -> None:
    text = """# 트랙 B 최종 학습 계획 v2

## 평가 목표

- 지도 분류 모델 2종 이상과 정상 패턴 기반 비지도 모델 1종 이상을 동일한 독립 시험 세트에서 비교한다.
- 불량 행 Recall과 물리 이벤트 Recall을 우선하고, 정상 오경보율과 오경보 이벤트 수를 함께 제한한다.
- 높은 정확도가 동일 파일 분할이나 현재 라벨의 직접 누수에서 나오지 않도록 파일 역할을 먼저 잠근다.

## 데이터 역할

- 정상 기준 학습/보정: `Training_Data.csv`의 앞 70%/뒤 30% 완전 용접 사이클.
- 지도 개발: `WeldingTest_01_OK`와 `WeldingTest_03_NG`; 파일 내부는 39행 완전 사이클을 유지한 시간순 70%/30% train/validation.
- 잠금 시험: `WeldingTest_02_OK`와 `WeldingTest_04_NG` 전체 파일. 적합, 임계값 선택, 모델 선택에 사용하지 않는다.

## 후보 모델

- 지도: LogisticCurrent, RandomForestCurrent.
- 현재값 의존성 진단: LogisticHistoryOnly. 현재 RealPower/Speed/Length를 제외하여 사전 징후 성능을 확인한다.
- 비지도: RobustPhaseZ, IsolationForestNormal. 정상 이력만 학습하며 임계값은 정상 보정 구간 99.9 분위수로 고정한다.

## 선택과 검증

- 지도 임계값은 validation에서 이벤트 Recall 100%, 정상 FPR 1% 이하를 우선한다.
- 최종 모델은 validation에서 이벤트 누락, FN, 오경보 이벤트, F1, FP 순으로 선택한다.
- 모델 선택 뒤 잠금 시험을 한 번 평가한다. 모든 모델은 동일 행과 동일 라벨을 사용한다.
- 이벤트는 cycle별로 잘게 세지 않고 원본 파일에서 연속된 라벨 구간으로 정의한다.

## 해석 경계

- Current 모델은 현재 시점 이상 탐지기이며 미래 고장 예측기로 과장하지 않는다.
- HistoryOnly는 현재 측정값 없이 선행 탐지가 가능한지 확인하는 진단 모델이다.
- NG04는 하나의 장기 이벤트뿐이므로 이벤트 Recall 100%만으로 일반화를 단정하지 않는다.
"""
    (OUTPUT_DIR / "training_plan.md").write_text(text, encoding="utf-8")


def write_data_quality_report(
    audit: pd.DataFrame, historical_split: dict, split_manifest: pd.DataFrame
) -> None:
    split_summary = (
        split_manifest.groupby("split", as_index=False)
        .agg(
            cycles=("group_id", "size"),
            rows=("rows", "sum"),
            positive_rows=("positive_rows", "sum"),
        )
        .copy()
    )
    split_summary["positive_rate"] = (
        split_summary["positive_rows"] / split_summary["rows"]
    )
    lines = [
        "# 트랙 B 데이터 품질 및 분할 보고서 v2",
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
                "complete_cycles",
                "positive_rows",
                "realpower_zero_rows",
            ],
        ),
        "",
        "## 처리 규칙",
        "",
        "- 필수 9개 열과 WorkingTime 날짜시간 변환을 검사했다.",
        "- PageNo 1~39가 순서대로 존재하는 완전한 39행 용접 사이클만 허용했다.",
        "- 결측·중복·음수값을 임의 보간하거나 삭제하지 않고 발견 수를 기록했다.",
        "- NG04의 RealPower=0인 39행은 라벨이 있는 실제 이상 구간이므로 제거하지 않았다.",
        "- 개발·검증은 행을 섞기 전에 cycle을 시간순으로 배정했다.",
        "- 잠금 시험은 개발과 다른 파일 전체를 사용했다.",
        "",
        "## 정상 이력 분할",
        "",
        f"- 완전 사이클: {historical_split['complete_cycles']}",
        f"- 정상 기준 학습: {historical_split['train_cycles']} cycles / {historical_split['train_rows']} rows",
        f"- 정상 임계값 보정: {historical_split['calibration_cycles']} cycles / {historical_split['calibration_rows']} rows",
        "",
        "## 지도 개발 클래스 비율",
        "",
        markdown_table(
            split_summary,
            ["split", "cycles", "rows", "positive_rows", "positive_rate"],
        ),
        "",
        "지도 모델은 희소한 이상 행을 보완하기 위해 class_weight='balanced' 또는 "
        "class_weight='balanced_subsample'을 사용했다.",
    ]
    (OUTPUT_DIR / "data_quality_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(
    metrics: pd.DataFrame,
    per_file: pd.DataFrame,
    reverse_stress: pd.DataFrame,
    winner: str,
    historical_split: dict,
    manifest: pd.DataFrame,
    importance: pd.DataFrame,
) -> None:
    validation = metrics[metrics["split"].eq("validation")]
    test = metrics[metrics["split"].eq("locked_test")]
    winner_test = test[test["model"].eq(winner)].iloc[0]
    locked_by_file = per_file[
        (per_file["split"].eq("locked_test")) & (per_file["model"].eq(winner))
    ]
    top_lines = []
    for model, group in importance.groupby("model"):
        features = ", ".join(
            f"{row.feature} ({row.normalized_importance:.3f})"
            for row in group.nlargest(5, "normalized_importance").itertuples()
        )
        top_lines.append(f"- {model}: {features}")
    lines = [
        "# 트랙 B 지도·비지도 최종 검증 보고서 v2",
        "",
        f"검증 세트에서 선택한 최종 모델: **{winner}**",
        "",
        "## 평가 설계",
        "",
        "- 개발 파일: WeldingTest_01_OK, WeldingTest_03_NG.",
        "- 독립 잠금 시험 파일: WeldingTest_02_OK, WeldingTest_04_NG.",
        "- 잠금 시험 파일은 학습, 임계값 결정, 모델 선택에 사용하지 않았다.",
        "- 개발 파일 내부도 39행 용접 사이클을 유지하고 시간순으로 분할했다.",
        "- 비지도 모델은 Training_Data 정상 구간만 학습하고 정상 보정 구간으로 임계값을 고정했다.",
        "- 지도 모델은 희소한 이상 행에 balanced class weight를 적용했다.",
        "- 이벤트는 원본 파일의 연속 이상 구간으로 계산하여 NG04의 한 이벤트를 여러 cycle 이벤트로 부풀리지 않았다.",
        "",
        "## 개발 validation 결과",
        "",
        markdown_table(
            validation,
            [
                "model",
                "family",
                "precision",
                "recall",
                "f1",
                "false_positive_rate",
                "event_recall",
                "false_alarm_events",
            ],
        ),
        "",
        "## 독립 파일 잠금 시험 결과",
        "",
        markdown_table(
            test,
            [
                "model",
                "family",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "tn",
                "fp",
                "fn",
                "tp",
                "event_recall",
                "false_alarm_events",
                "mean_detection_delay_rows",
            ],
        ),
        "",
        "## 최종 모델의 파일별 성능",
        "",
        markdown_table(
            locked_by_file,
            [
                "source_file",
                "rows",
                "precision",
                "recall",
                "f1",
                "tn",
                "fp",
                "fn",
                "tp",
            ],
        ),
        "",
        "## 반대 방향 파일 스트레스 테스트",
        "",
        "이 표는 02_OK+04_NG로 개발하고 01_OK+03_NG 전체를 시험한 민감도 분석이다. "
        "NG04에 독립된 두 번째 이상 이벤트가 없어 내부 validation이 같은 장기 이벤트를 나누므로 "
        "최종 모델 선택에는 사용하지 않았다.",
        "",
        markdown_table(
            reverse_stress,
            [
                "model",
                "family",
                "precision",
                "recall",
                "f1",
                "fn",
                "fp",
                "event_recall",
                "false_alarm_events",
            ],
        ),
        "",
        "## 성공 기준 판정",
        "",
        f"- 불량 행 Recall >= 0.90: {'PASS' if winner_test['recall'] >= 0.90 else 'FAIL'} ({winner_test['recall']:.4f})",
        f"- 물리 이벤트 Recall = 1.00: {'PASS' if winner_test['event_recall'] >= 1.0 else 'FAIL'} ({winner_test['event_recall']:.4f})",
        f"- 정상 행 FPR <= 0.01: {'PASS' if winner_test['false_positive_rate'] <= 0.01 else 'FAIL'} ({winner_test['false_positive_rate']:.4f})",
        f"- 독립 시험 FN: {int(winner_test['fn'])}, FP: {int(winner_test['fp'])}",
        "",
        "## 높은 성능에 대한 점검",
        "",
        "- 잠금 시험은 학습과 다른 파일 및 다른 NG 유형 전체를 사용하므로 이전 cycle 혼합 평가보다 강하다.",
        "- Current 모델은 현재 RealPower를 사용하므로 현재 이상 감지 성능이다. 미래 고장 예측 성능으로 해석하지 않는다.",
        "- HistoryOnly 결과를 함께 제시하여 현재 측정값을 제거했을 때의 성능 저하를 확인한다.",
        "- 비지도 RobustPhaseZ가 높은 성능을 보이면 정상 PageNo별 RealPower 범위와 NG04의 분리가 매우 크다는 데이터 특성 때문이다.",
        "",
        "## 주요 특징",
        "",
        *top_lines,
        "",
        "## 데이터 규모",
        "",
        f"- 정상 기준 학습: {historical_split['train_cycles']} cycles / {historical_split['train_rows']} rows",
        f"- 정상 임계값 보정: {historical_split['calibration_cycles']} cycles / {historical_split['calibration_rows']} rows",
        f"- 지도 개발 train: {int((manifest['split'] == 'development_train').sum())} cycles",
        f"- 지도 개발 validation: {int((manifest['split'] == 'development_validation').sum())} cycles",
        "- 잠금 시험: WeldingTest_02_OK 48 cycles + WeldingTest_04_NG 9 cycles.",
        "",
        "## 한계",
        "",
        "- 독립 시험 NG 파일이 하나이며 NG04에는 장기 이상 이벤트가 한 건뿐이다.",
        "- NG03과 NG04 외의 새로운 고장 모드에 대한 외부 타당성은 확인되지 않았다.",
        "- 현 데이터에는 고장 이전을 명시하는 horizon 라벨이 없어 사전 예측 시간을 직접 평가할 수 없다.",
        "- 신규 시점·신규 설비 파일을 추가 확보해 파일 단위 반복 검증을 해야 한다.",
    ]
    (OUTPUT_DIR / "final_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    write_plan()

    historical_path = DATA_ROOT / "raw_data" / "train" / "Training_Data.csv"
    historical = require_complete_cycles(add_identity(read_signal(historical_path), "Training_Data"))
    historical_train, historical_calibration, historical_split = split_historical_normal(historical)

    names = [
        "WeldingTest_01_OK",
        "WeldingTest_02_OK",
        "WeldingTest_03_NG",
        "WeldingTest_04_NG",
    ]
    tests = {name: load_test_file(name) for name in names}
    development_train, development_validation, split_manifest = split_development_by_cycle(
        [tests["WeldingTest_01_OK"], tests["WeldingTest_03_NG"]]
    )
    locked_test = pd.concat(
        [tests["WeldingTest_02_OK"], tests["WeldingTest_04_NG"]], ignore_index=True
    )

    if set(development_train["source_file"]) & set(locked_test["source_file"]):
        raise RuntimeError("Development and locked test files overlap")
    if development_validation["label"].nunique() != 2 or locked_test["label"].nunique() != 2:
        raise RuntimeError("Validation and locked test must both contain two classes")

    models, threshold_audit = fit_models(
        historical_train,
        historical_calibration,
        development_train,
        development_validation,
    )
    metrics, predictions, per_file, reports = evaluate_models(
        models, development_validation, locked_test
    )
    winner = choose_winner(metrics)
    importance = feature_importance(models)

    reverse_train, reverse_validation, _ = split_development_by_cycle(
        [tests["WeldingTest_02_OK"], tests["WeldingTest_04_NG"]]
    )
    reverse_models, _ = fit_models(
        historical_train,
        historical_calibration,
        reverse_train,
        reverse_validation,
    )
    reverse_test = pd.concat(
        [tests["WeldingTest_01_OK"], tests["WeldingTest_03_NG"]], ignore_index=True
    )
    reverse_metrics_all, _, _, _ = evaluate_models(
        reverse_models, reverse_validation, reverse_test
    )
    reverse_stress = reverse_metrics_all[
        reverse_metrics_all["split"].eq("locked_test")
    ].copy()
    reverse_stress["split"] = "reverse_file_stress"

    audit_frames = {"Training_Data": historical, **tests}
    audit = data_audit(audit_frames)
    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False, encoding="utf-8-sig")
    per_file.to_csv(OUTPUT_DIR / "per_file_metrics.csv", index=False, encoding="utf-8-sig")
    reverse_stress.to_csv(
        OUTPUT_DIR / "reverse_file_stress_metrics.csv", index=False, encoding="utf-8-sig"
    )
    split_manifest.to_csv(OUTPUT_DIR / "split_manifest.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(OUTPUT_DIR / "data_quality.csv", index=False, encoding="utf-8-sig")
    write_data_quality_report(audit, historical_split, split_manifest)
    (OUTPUT_DIR / "classification_reports.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for detector in models:
        joblib.dump(
            {
                "pipeline_version": "track-b-final-v2",
                "name": detector.name,
                "family": detector.family,
                "model": detector.model,
                "threshold": detector.threshold,
                "threshold_source": detector.threshold_source,
                "feature_names": detector.feature_names,
                "reference": detector.reference,
            },
            MODEL_DIR / f"{detector.name}.joblib",
        )

    data_paths = [
        historical_path,
        *[DATA_ROOT / "raw_data" / "test" / f"{name}.csv" for name in names],
        DATA_ROOT / "preprocessed" / "test" / "WeldingTest_03_NG_Label.csv",
        DATA_ROOT / "preprocessed" / "test" / "WeldingTest_04_NG_Label.csv",
    ]
    manifest = {
        "pipeline_version": "track-b-final-v2",
        "seed": SEED,
        "evaluation_protocol": "independent_file_holdout",
        "development_files": ["WeldingTest_01_OK", "WeldingTest_03_NG"],
        "locked_test_files": ["WeldingTest_02_OK", "WeldingTest_04_NG"],
        "winner_selected_on_validation": winner,
        "locked_test_used_for_selection": False,
        "reverse_stress_used_for_selection": False,
        "reverse_stress_limitation": "NG04 has one continuous event, so its inner development split is not event-independent",
        "event_definition": "contiguous anomaly labels in original file order",
        "historical_split": historical_split,
        "threshold_audit": threshold_audit,
        "data_sha256": {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in data_paths},
        "pipeline_sha256": sha256(Path(__file__)),
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    plot_results(metrics, predictions, importance)
    write_report(
        metrics,
        per_file,
        reverse_stress,
        winner,
        historical_split,
        split_manifest,
        importance,
    )

    print(metrics.to_string(index=False))
    print(f"Winner selected on validation: {winner}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
