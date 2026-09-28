from __future__ import annotations

import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
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
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pdm_contract import validate_signals
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
PIPELINE_VERSION = "track-b-audited-v3"
COMPLEXITY_ORDER = ['RobustPhaseZ', 'LogisticCurrent', 'LightGBMResidual',
                    'RandomForestCurrent', 'IsolationForestNormal', 'LogisticHistoryOnly']
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

LIGHTGBM_REGRESSION_FEATURES = [
    "PageNo",
    "PageSin",
    "PageCos",
    "Speed",
    "Length",
    "SetPower",
    "GateOnTime",
    "PreviousRealPower",
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
    return validate_signals(pd.read_csv(path))


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
        label_frame = pd.read_csv(label_path)
        if ('label' not in label_frame or not label_frame['label'].isin([0, 1]).all()
                or not np.array_equal(label_frame.iloc[:, 0], np.arange(len(frame)))):
            raise ValueError(f'{name}: invalid labels or row alignment')
        labels = label_frame['label'].to_numpy(dtype=int)
    else:
        labels = np.zeros(len(frame), dtype=int)
    if len(labels) != len(frame):
        raise ValueError(f"{name}: label length mismatch")
    frame["label"] = labels
    return frame


def split_historical_normal(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    all_groups = [group.copy() for _, group in frame.groupby("group_id", sort=False)]
    contaminated_groups = [
        group for group in all_groups if group["RealPower"].eq(0).any()
    ]
    time_invalid = [g for g in all_groups if (g.WorkingTime.diff().dt.total_seconds().dropna() <= 0).any()]
    excluded = {str(g.group_id.iloc[0]) for g in contaminated_groups + time_invalid}
    groups = sorted([g for g in all_groups if str(g.group_id.iloc[0]) not in excluded],
                    key=lambda g: g.WorkingTime.iloc[0])
    if not groups:
        raise RuntimeError("No clean historical cycles remain after zero-power filtering")
    cut = int(len(groups) * 0.70)
    train = pd.concat(groups[:cut], ignore_index=True)
    calibration = pd.concat(groups[cut:], ignore_index=True)
    if train.WorkingTime.max() >= calibration.WorkingTime.min():
        raise ValueError('Historical cycle intervals overlap the fit/calibration boundary')
    return train, calibration, {
        "complete_cycles": len(all_groups),
        "clean_cycles": len(groups),
        "excluded_zero_power_cycles": len(contaminated_groups),
        "excluded_zero_power_rows": int(sum(len(group) for group in contaminated_groups)),
        "excluded_time_cycles": len(time_invalid),
        "excluded_time_rows": int(sum(len(g) for g in time_invalid)),
        "excluded_time_group_ids": [str(g.group_id.iloc[0]) for g in time_invalid],
        "order": "cycle_start_time; original source_row and within-cycle order preserved",
        "fit_end": str(train.WorkingTime.max()),
        "calibration_start": str(calibration.WorkingTime.min()),
        "excluded_group_ids": [
            str(group["group_id"].iloc[0]) for group in contaminated_groups
        ],
        "train_cycles": cut,
        "calibration_cycles": len(groups) - cut,
        "train_rows": int(len(train)),
        "calibration_rows": int(len(calibration)),
    }


def split_development_by_cycle(
    frames: list[pd.DataFrame],
    fraction: float = DEVELOPMENT_TRAIN_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    manifest_rows: list[dict] = []
    for frame in frames:
        source_file = str(frame["source_file"].iloc[0])
        groups = [group.copy() for _, group in frame.groupby("group_id", sort=False)]
        cut = max(1, min(len(groups) - 1, int(len(groups) * fraction)))
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
    work = validate_signals(frame).reset_index(drop=True)
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

    time_gap = work.groupby('group_id', sort=False).WorkingTime.diff().dt.total_seconds()
    if (time_gap.dropna() <= 0).any():
        raise ValueError('입력 오류: 사이클 내부 시간 역전 또는 중복 시각입니다.')
    time_gap = time_gap.clip(upper=120.0).fillna(0.0)
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
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError('입력 오류: 모델 기준에 없는 공정 또는 유효하지 않은 특징값입니다.')
    return features


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
        if self.name == "LightGBMResidual":
            prediction = self.model["estimator"].predict(features[self.feature_names])
            residual = np.abs(frame["RealPower"].to_numpy(dtype=float) - prediction)
            center = frame["PageNo"].map(self.model["residual_median"]).to_numpy(dtype=float)
            scale = frame["PageNo"].map(self.model["residual_scale"]).to_numpy(dtype=float)
            return np.abs(residual - center) / scale
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
    include_unsupervised: bool = True,
) -> tuple[list[Detector], dict]:
    started_reference = time.perf_counter()
    reference = phase_reference(historical_train)
    reference_seconds = time.perf_counter() - started_reference
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

    if not include_unsupervised:
        return models, threshold_audit
    started_robust = time.perf_counter()
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
            training_seconds=reference_seconds + time.perf_counter() - started_robust,
        )
    )

    historical_features = make_features(historical_train, reference)
    historical_groups = historical_train["group_id"].drop_duplicates().tolist()
    lightgbm_validation_start = int(len(historical_groups) * 0.85)
    lightgbm_train_groups = set(historical_groups[:lightgbm_validation_start])
    lightgbm_train_mask = historical_train["group_id"].isin(lightgbm_train_groups)
    lightgbm_target = historical_train["RealPower"].to_numpy(dtype=float)
    started = time.perf_counter()
    lightgbm = lgb.LGBMRegressor(
        objective="huber",
        alpha=50.0,
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
    lightgbm.fit(
        historical_features.loc[lightgbm_train_mask, LIGHTGBM_REGRESSION_FEATURES],
        lightgbm_target[lightgbm_train_mask.to_numpy()],
        eval_X=historical_features.loc[
            ~lightgbm_train_mask, LIGHTGBM_REGRESSION_FEATURES
        ],
        eval_y=lightgbm_target[(~lightgbm_train_mask).to_numpy()],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    historical_prediction = lightgbm.predict(
        historical_features[LIGHTGBM_REGRESSION_FEATURES]
    )
    residual_frame = pd.DataFrame(
        {
            "PageNo": historical_train["PageNo"].to_numpy(dtype=int),
            "RealPower": np.abs(lightgbm_target - historical_prediction),
        }
    )
    residual_reference = phase_reference(residual_frame)
    lightgbm_payload = {
        "estimator": lightgbm,
        "residual_median": residual_reference["median"],
        "residual_scale": residual_reference["scale"],
        "early_stopping_train_cycles": len(lightgbm_train_groups),
        "early_stopping_validation_cycles": len(historical_groups)
        - len(lightgbm_train_groups),
    }
    lightgbm_detector = Detector(
        name="LightGBMResidual",
        family="unsupervised",
        model=lightgbm_payload,
        threshold=0.0,
        feature_names=list(LIGHTGBM_REGRESSION_FEATURES),
        reference=reference,
        threshold_source="historical_normal_calibration_99.9pct",
        training_seconds=time.perf_counter() - started,
    )
    lightgbm_scores = lightgbm_detector.score(historical_calibration)
    lightgbm_detector.threshold = float(
        np.quantile(lightgbm_scores, NORMAL_CALIBRATION_QUANTILE, method="higher")
    )
    models.append(lightgbm_detector)

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
    models: list[Detector], validation: pd.DataFrame, locked_test: pd.DataFrame | None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    metric_rows = []
    prediction_rows = []
    per_file_rows = []
    reports: dict = {}
    for detector in models:
        for split_name, frame in [("validation", validation), ("locked_test", locked_test)]:
            if frame is None:
                continue
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
    validation['complexity_rank'] = validation.model.map({n: i for i, n in enumerate(COMPLEXITY_ORDER)})
    validation = validation.sort_values(
        [
            "missed_events",
            "fn",
            "false_alarm_events",
            "f1",
            "fp",
            "complexity_rank",
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
    ax.set_title("Track B - retrospective file holdout")
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


def main() -> None:
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    historical_path = DATA_ROOT / "raw_data" / "train" / "Training_Data.csv"
    historical = require_complete_cycles(add_identity(read_signal(historical_path), "Training_Data"))
    historical_train, historical_calibration, historical_split = split_historical_normal(historical)
    fit_ids, cal_ids = set(historical_train.group_id), set(historical_calibration.group_id)
    history_groups = []
    for group_id, group in historical.groupby('group_id', sort=False):
        reasons = []
        if group.RealPower.eq(0).any():
            reasons.append('zero_power')
        if (group.WorkingTime.diff().dt.total_seconds().dropna() <= 0).any():
            reasons.append('within_cycle_time_error')
        history_groups.append(dict(group_id=group_id, original_start_row=int(group.source_row.min()),
            start=str(group.WorkingTime.iloc[0]), end=str(group.WorkingTime.iloc[-1]),
            split='fit' if group_id in fit_ids else 'calibration' if group_id in cal_ids else 'quarantine',
            reason=';'.join(reasons)))
    pd.DataFrame(history_groups).to_csv(OUTPUT_DIR/'historical_split_manifest.csv', index=False, encoding='utf-8-sig')

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
    validation_metrics, _, _, _ = evaluate_models(models, development_validation, None)
    winner = choose_winner(validation_metrics)
    selected_supervised = choose_winner(validation_metrics[validation_metrics.family.eq('supervised')])
    selected_unsupervised = choose_winner(validation_metrics[validation_metrics.family.eq('unsupervised')])
    selection = {'winner': winner, 'supervised': selected_supervised, 'unsupervised': selected_unsupervised,
                 'selected_at': datetime.now(timezone.utc).isoformat(),
                 'rule': 'validation missed_events, fn, false_alarm_events, -f1, fp, fixed complexity order',
                 'complexity_order': COMPLEXITY_ORDER, 'test_evaluated': False}
    (OUTPUT_DIR/'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    test_evaluation_started_at = datetime.now(timezone.utc).isoformat()
    metrics, predictions, per_file, reports = evaluate_models(
        models, development_validation, locked_test
    )
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
    (OUTPUT_DIR / "classification_reports.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for detector in models:
        joblib.dump(
            {
                "pipeline_version": PIPELINE_VERSION,
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
        "pipeline_version": PIPELINE_VERSION,
        "seed": SEED,
        "evaluation_protocol": "retrospective_reused_file_holdout",
        "external_validation_completed": False,
        "selection": selection,
        "test_evaluation_started_at": test_evaluation_started_at,
        "default_supervised": selected_supervised,
        "default_unsupervised": selected_unsupervised,
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
    import importlib.metadata
    import platform
    import pdm_evaluation as evidence
    manifest['environment'] = {'python': platform.python_version(), **{
        name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'joblib']}}
    manifest['source_sha256'] = {str(p.relative_to(PROJECT_ROOT.parent)):sha256(p) for p in [
        Path(__file__), PROJECT_ROOT/'src/pdm_contract.py', PROJECT_ROOT/'src/pdm_evaluation.py',
        PROJECT_ROOT.parent/'backend/maintenance_policy.py', PROJECT_ROOT/'docs/TRAINING_PLAN_V3.md']}
    manifest['model_sha256'] = {p.name:sha256(p) for p in MODEL_DIR.glob('*.joblib')}
    manifest['alarm_policy'] = evidence.POLICY
    operating = evidence.all_operational(models, development_validation, locked_test)
    operating.to_csv(OUTPUT_DIR/'operational_metrics.csv', index=False, encoding='utf-8-sig')
    sensitivity, forward, forward_split = evidence.supplementary(
        __import__(__name__), models, historical, historical_train, historical_calibration, tests, locked_test)
    # When executed as a script, __import__('__main__') is this fitted pipeline.
    sensitivity.to_csv(OUTPUT_DIR/'sensitivity_metrics.csv', index=False, encoding='utf-8-sig')
    forward.to_csv(OUTPUT_DIR/'forward_time_metrics.csv', index=False, encoding='utf-8-sig')
    manifest['forward_time_split'] = forward_split
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    plot_results(metrics, predictions, importance)
    evidence.write_report(__import__(__name__), manifest, metrics, operating, sensitivity, forward)

    print(metrics.to_string(index=False))
    print(f"Winner selected on validation: {winner}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
