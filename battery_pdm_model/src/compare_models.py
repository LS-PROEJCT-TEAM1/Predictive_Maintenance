from __future__ import annotations

import io
import json
import math
import os
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "model_comparison"
MPL_CONFIG_DIR = OUTPUT_DIR / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
CALIBRATION_QUANTILE = 0.999
CONTEXT_LENGTH = 20
PREDICTION_LENGTH = 10

DATA_ROOT = PROJECT_ROOT / "Dataset_전자부품(배터리팩) 예지보전 AI 데이터셋" / "data"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


def read_signal(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    frame["WorkingTime"] = pd.to_datetime(frame["WorkingTime"], errors="raise")
    return frame


def add_cycle_id(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["cycle_id"] = result["PageNo"].eq(1).cumsum().astype(int)
    return result


def split_training_cycles(frame: pd.DataFrame, train_fraction: float = 0.70) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    cycled = add_cycle_id(frame)
    groups: List[Tuple[pd.Timestamp, pd.DataFrame]] = []
    rejected = []
    for cycle_id, group in cycled.groupby("cycle_id", sort=False):
        group = group.sort_values("PageNo").copy()
        complete = len(group) == 39 and group["PageNo"].tolist() == list(range(1, 40))
        contains_zero = bool(group["RealPower"].eq(0).any())
        if not complete or contains_zero:
            rejected.append({
                "cycle_id": int(cycle_id),
                "complete": complete,
                "contains_zero": contains_zero,
            })
            continue
        groups.append((group["WorkingTime"].min(), group))

    groups.sort(key=lambda item: item[0])
    split_at = int(len(groups) * train_fraction)
    train_groups = groups[:split_at]
    calibration_groups = groups[split_at:]

    train = pd.concat([group for _, group in train_groups], ignore_index=True)
    calibration = pd.concat([group for _, group in calibration_groups], ignore_index=True)
    info = {
        "accepted_cycles": len(groups),
        "training_cycles": len(train_groups),
        "calibration_cycles": len(calibration_groups),
        "rejected_cycles": rejected,
        "training_start": str(train["WorkingTime"].min()),
        "training_end": str(train["WorkingTime"].max()),
        "calibration_start": str(calibration["WorkingTime"].min()),
        "calibration_end": str(calibration["WorkingTime"].max()),
    }
    return train, calibration, info


def quantile_threshold(scores: np.ndarray, quantile: float = CALIBRATION_QUANTILE) -> float:
    finite = np.asarray(scores, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, quantile, method="higher"))


def phase_location_scale(frame: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    median = frame.groupby("PageNo")["RealPower"].median()
    mad = frame.groupby("PageNo")["RealPower"].apply(
        lambda values: float(np.median(np.abs(values - np.median(values))))
    )
    robust = 1.4826 * mad
    standard = frame.groupby("PageNo")["RealPower"].std().fillna(0.0)
    scale = pd.concat(
        [robust.rename("mad"), (0.25 * standard).rename("std_floor"), pd.Series(1.0, index=median.index, name="unit_floor")],
        axis=1,
    ).max(axis=1)
    return median.astype(float), scale.astype(float)


def phase_signed_z(frame: pd.DataFrame, median: pd.Series, scale: pd.Series) -> np.ndarray:
    center = frame["PageNo"].map(median).to_numpy(dtype=float)
    spread = frame["PageNo"].map(scale).to_numpy(dtype=float)
    return (frame["RealPower"].to_numpy(dtype=float) - center) / spread


def serialized_kb(value) -> float:
    try:
        return len(pickle.dumps(value)) / 1024.0
    except Exception:
        return float("nan")


class BaseDetector:
    name = "Base"

    def __init__(self) -> None:
        self.threshold = float("nan")
        self.training_seconds = float("nan")
        self.parameter_count = 0
        self.model_size_kb = float("nan")

    def fit(self, train: pd.DataFrame, calibration: pd.DataFrame) -> "BaseDetector":
        raise NotImplementedError

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (self.score(frame) > self.threshold).astype(int)


class RobustZDetector(BaseDetector):
    name = "RobustZ"

    def fit(self, train: pd.DataFrame, calibration: pd.DataFrame) -> "RobustZDetector":
        started = time.perf_counter()
        self.median, self.scale = phase_location_scale(train)
        self.threshold = quantile_threshold(np.abs(phase_signed_z(calibration, self.median, self.scale)))
        self.training_seconds = time.perf_counter() - started
        self.parameter_count = 78
        self.model_size_kb = serialized_kb((self.median, self.scale, self.threshold))
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return np.abs(phase_signed_z(frame, self.median, self.scale))


class EWMACUSUMDetector(BaseDetector):
    name = "EWMA_CUSUM"

    def __init__(self, alpha: float = 0.35, drift: float = 0.50, decay: float = 0.85) -> None:
        super().__init__()
        self.alpha = alpha
        self.drift = drift
        self.decay = decay

    def _sequential_score(self, signed_z: np.ndarray) -> np.ndarray:
        clipped = np.clip(signed_z, -25.0, 25.0)
        result = np.zeros(len(clipped), dtype=float)
        ewma = 0.0
        positive = 0.0
        negative = 0.0
        for index, value in enumerate(clipped):
            ewma = self.alpha * value + (1.0 - self.alpha) * ewma
            positive = max(0.0, self.decay * positive + value - self.drift)
            negative = max(0.0, self.decay * negative - value - self.drift)
            result[index] = max(abs(ewma), positive, negative)
        return result

    def fit(self, train: pd.DataFrame, calibration: pd.DataFrame) -> "EWMACUSUMDetector":
        started = time.perf_counter()
        self.median, self.scale = phase_location_scale(train)
        calibration_scores = self._sequential_score(phase_signed_z(calibration, self.median, self.scale))
        self.threshold = quantile_threshold(calibration_scores)
        self.training_seconds = time.perf_counter() - started
        self.parameter_count = 81
        self.model_size_kb = serialized_kb(
            (self.median, self.scale, self.threshold, self.alpha, self.drift, self.decay)
        )
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return self._sequential_score(phase_signed_z(frame, self.median, self.scale))


def regression_features(frame: pd.DataFrame, phase_median: pd.Series) -> pd.DataFrame:
    work = frame.reset_index(drop=True)
    timestamps = work["WorkingTime"]
    previous_power = work["RealPower"].shift(1)
    default_previous = work["PageNo"].sub(1).replace(0, 39).map(phase_median)
    previous_power = previous_power.where(work["PageNo"].ne(1), default_previous)
    previous_power = previous_power.fillna(default_previous).astype(float)
    time_gap = timestamps.diff().dt.total_seconds().clip(lower=0.0, upper=120.0).fillna(0.0)
    page_angle = 2.0 * np.pi * (work["PageNo"].to_numpy(dtype=float) - 1.0) / 39.0
    return pd.DataFrame(
        {
            "PageNo": work["PageNo"].astype(int),
            "Speed": work["Speed"].astype(float),
            "Length": work["Length"].astype(float),
            "SetPower": work["SetPower"].astype(float),
            "GateOnTime": work["GateOnTime"].astype(float),
            "PreviousPower": previous_power,
            "TimeGap": time_gap,
            "PageSin": np.sin(page_angle),
            "PageCos": np.cos(page_angle),
        }
    )


class LightGBMDetector(BaseDetector):
    name = "LightGBM"

    def fit(self, train: pd.DataFrame, calibration: pd.DataFrame) -> "LightGBMDetector":
        started = time.perf_counter()
        self.phase_median, _ = phase_location_scale(train)
        features = regression_features(train, self.phase_median)
        target = train["RealPower"].to_numpy(dtype=float)
        validation_start = int(len(train) * 0.85)
        self.model = lgb.LGBMRegressor(
            objective="huber",
            n_estimators=500,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=40,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
        )
        self.model.fit(
            features.iloc[:validation_start],
            target[:validation_start],
            eval_set=[(features.iloc[validation_start:], target[validation_start:])],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        train_prediction = self.model.predict(features)
        residual_frame = pd.DataFrame(
            {
                "PageNo": train["PageNo"].to_numpy(),
                "RealPower": np.abs(target - train_prediction),
            }
        )
        self.residual_median, self.residual_scale = phase_location_scale(residual_frame)
        calibration_scores = self.score(calibration)
        self.threshold = quantile_threshold(calibration_scores)
        self.training_seconds = time.perf_counter() - started
        booster = self.model.booster_
        self.parameter_count = int(sum(tree.get("num_leaves", 0) for tree in booster.dump_model()["tree_info"]))
        self.model_size_kb = serialized_kb(self.model)
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        features = regression_features(frame, self.phase_median)
        prediction = self.model.predict(features)
        residual = np.abs(frame["RealPower"].to_numpy(dtype=float) - prediction)
        center = frame["PageNo"].map(self.residual_median).to_numpy(dtype=float)
        scale = frame["PageNo"].map(self.residual_scale).to_numpy(dtype=float)
        return np.abs(residual - center) / scale


class NHiTSBlock(nn.Module):
    def __init__(self, context_length: int, prediction_length: int, pool_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.pool_size = pool_size
        pooled_length = math.ceil(context_length / pool_size)
        coarse_forecast_length = math.ceil(prediction_length / pool_size)
        self.coarse_forecast_length = coarse_forecast_length
        self.network = nn.Sequential(
            nn.Linear(pooled_length, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, context_length + coarse_forecast_length),
        )

    def forward(self, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = F.avg_pool1d(
            values.unsqueeze(1),
            kernel_size=self.pool_size,
            stride=self.pool_size,
            ceil_mode=True,
            count_include_pad=False,
        ).squeeze(1)
        theta = self.network(pooled)
        backcast = theta[:, : self.context_length]
        coarse = theta[:, self.context_length :].unsqueeze(1)
        forecast = F.interpolate(
            coarse,
            size=self.prediction_length,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        return backcast, forecast


class CompactNHiTS(nn.Module):
    def __init__(self, context_length: int = CONTEXT_LENGTH, prediction_length: int = PREDICTION_LENGTH) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [NHiTSBlock(context_length, prediction_length, pool_size) for pool_size in (1, 2, 4)]
        )
        self.prediction_length = prediction_length

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        forecast = torch.zeros((len(values), self.prediction_length), dtype=values.dtype, device=values.device)
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        return forecast


def adjusted_power(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.where(values < 1300.0, values + 930.5, values)


def make_forecast_windows(sequence: np.ndarray, context_length: int, prediction_length: int) -> Tuple[np.ndarray, np.ndarray]:
    window = context_length + prediction_length
    view = np.lib.stride_tricks.sliding_window_view(sequence, window)
    return view[:, :context_length].copy(), view[:, context_length:].copy()


class NHiTSZDetector(BaseDetector):
    name = "NHiTS_Z4"

    def __init__(self) -> None:
        super().__init__()
        self.model = CompactNHiTS()

    def fit(self, train: pd.DataFrame, calibration: pd.DataFrame) -> "NHiTSZDetector":
        started = time.perf_counter()
        train_values = adjusted_power(train["RealPower"].to_numpy())
        self.value_mean = float(train_values.mean())
        self.value_std = float(train_values.std())
        normalized = (train_values - self.value_mean) / self.value_std
        x, y = make_forecast_windows(normalized, CONTEXT_LENGTH, PREDICTION_LENGTH)
        if len(x) > 80000:
            rng = np.random.default_rng(SEED)
            selected = np.sort(rng.choice(len(x), size=80000, replace=False))
            x, y = x[selected], y[selected]
        validation_start = int(len(x) * 0.90)
        train_dataset = TensorDataset(torch.from_numpy(x[:validation_start]), torch.from_numpy(y[:validation_start]))
        validation_x = torch.from_numpy(x[validation_start:])
        validation_y = torch.from_numpy(y[validation_start:])
        loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, generator=torch.Generator().manual_seed(SEED))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        best_state = None
        best_loss = float("inf")
        patience = 0
        for _ in range(18):
            self.model.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = F.l1_loss(self.model(batch_x), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
            self.model.eval()
            with torch.no_grad():
                validation_loss = float(F.l1_loss(self.model(validation_x), validation_y))
            if validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_state = {key: value.detach().clone() for key, value in self.model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 4:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        calibration_errors = self._prediction_errors(calibration)
        finite = calibration_errors[np.isfinite(calibration_errors)]
        self.error_mean = float(finite.mean())
        self.error_std = float(max(finite.std(), 1e-6))
        self.threshold = 4.0
        self.training_seconds = time.perf_counter() - started
        self.parameter_count = int(sum(parameter.numel() for parameter in self.model.parameters()))
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        self.model_size_kb = len(buffer.getvalue()) / 1024.0
        return self

    def _predict_next(self, contexts: np.ndarray) -> np.ndarray:
        self.model.eval()
        results = []
        with torch.no_grad():
            for start in range(0, len(contexts), 2048):
                batch = torch.from_numpy(contexts[start : start + 2048])
                results.append(self.model(batch)[:, 0].cpu().numpy())
        return np.concatenate(results) if results else np.empty(0, dtype=float)

    def _prediction_errors(self, frame: pd.DataFrame) -> np.ndarray:
        values = adjusted_power(frame["RealPower"].to_numpy())
        normalized = (values - self.value_mean) / self.value_std
        if len(normalized) <= CONTEXT_LENGTH:
            return np.full(len(normalized), np.nan)
        contexts = np.lib.stride_tricks.sliding_window_view(normalized, CONTEXT_LENGTH)[:-1].copy()
        prediction = self._predict_next(contexts)
        actual = normalized[CONTEXT_LENGTH:]
        errors = np.full(len(normalized), np.nan, dtype=float)
        errors[CONTEXT_LENGTH:] = prediction - actual
        return errors

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        errors = self._prediction_errors(frame)
        scores = np.zeros(len(frame), dtype=float)
        valid = np.isfinite(errors)
        scores[valid] = np.abs((errors[valid] - self.error_mean) / self.error_std)
        return scores


class CycleAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(39, 20),
            nn.ReLU(),
            nn.Linear(20, 8),
            nn.ReLU(),
            nn.Linear(8, 20),
            nn.ReLU(),
            nn.Linear(20, 39),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class CycleAutoencoderDetector(BaseDetector):
    name = "Cycle_AE"

    def __init__(self) -> None:
        super().__init__()
        self.model = CycleAutoencoder()

    def _cycle_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        signed = phase_signed_z(frame, self.median, self.scale)
        if len(signed) % 39 != 0:
            raise ValueError("Cycle autoencoder requires complete 39-row cycles")
        return signed.reshape(-1, 39).astype(np.float32)

    def fit(self, train: pd.DataFrame, calibration: pd.DataFrame) -> "CycleAutoencoderDetector":
        started = time.perf_counter()
        self.median, self.scale = phase_location_scale(train)
        matrix = self._cycle_matrix(train)
        validation_start = int(len(matrix) * 0.90)
        dataset = TensorDataset(torch.from_numpy(matrix[:validation_start]))
        loader = DataLoader(dataset, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(SEED))
        validation = torch.from_numpy(matrix[validation_start:])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        best_state = None
        best_loss = float("inf")
        patience = 0
        for _ in range(100):
            self.model.train()
            for (batch,) in loader:
                optimizer.zero_grad(set_to_none=True)
                reconstruction = self.model(batch)
                loss = F.mse_loss(reconstruction, batch)
                loss.backward()
                optimizer.step()
            self.model.eval()
            with torch.no_grad():
                validation_loss = float(F.mse_loss(self.model(validation), validation))
            if validation_loss < best_loss - 1e-6:
                best_loss = validation_loss
                best_state = {key: value.detach().clone() for key, value in self.model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 8:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.threshold = quantile_threshold(self.score(calibration))
        self.training_seconds = time.perf_counter() - started
        self.parameter_count = int(sum(parameter.numel() for parameter in self.model.parameters()))
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        self.model_size_kb = len(buffer.getvalue()) / 1024.0
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self._cycle_matrix(frame)
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(torch.from_numpy(matrix)).cpu().numpy()
        return np.abs(matrix - reconstructed).reshape(-1).astype(float)


def event_intervals(labels: np.ndarray) -> List[Tuple[int, int]]:
    labels = np.asarray(labels, dtype=int)
    starts = np.where((labels == 1) & (np.r_[0, labels[:-1]] == 0))[0]
    ends = np.where((labels == 1) & (np.r_[labels[1:], 0] == 0))[0]
    return list(zip(starts.tolist(), ends.tolist()))


def count_predicted_events(prediction: np.ndarray) -> int:
    return len(event_intervals(np.asarray(prediction, dtype=int)))


def evaluate_detector(
    detector: BaseDetector,
    test_frames: Dict[str, pd.DataFrame],
    labels: Dict[str, np.ndarray],
) -> Tuple[dict, pd.DataFrame, dict]:
    prediction_rows = []
    all_true = []
    all_prediction = []
    per_file = {}

    inference_started = time.perf_counter()
    cached_scores = {name: detector.score(frame) for name, frame in test_frames.items()}
    inference_seconds = time.perf_counter() - inference_started

    for file_name, frame in test_frames.items():
        truth = labels[file_name].astype(int)
        score = cached_scores[file_name]
        prediction = (score > detector.threshold).astype(int)
        all_true.append(truth)
        all_prediction.append(prediction)
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, prediction, average="binary", zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
        per_file[file_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        }
        for row_index, (actual, predicted, anomaly_score) in enumerate(zip(truth, prediction, score)):
            prediction_rows.append(
                {
                    "model": detector.name,
                    "file": file_name,
                    "row_index": row_index,
                    "actual": int(actual),
                    "predicted": int(predicted),
                    "score": float(anomaly_score),
                    "threshold": float(detector.threshold),
                }
            )

    pooled_true = np.concatenate(all_true)
    pooled_prediction = np.concatenate(all_prediction)
    precision, recall, f1, _ = precision_recall_fscore_support(
        pooled_true, pooled_prediction, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(pooled_true, pooled_prediction, labels=[0, 1]).ravel()

    abnormal_files = ["WeldingTest_03_NG", "WeldingTest_04_NG"]
    macro_f1 = float(np.mean([per_file[name]["f1"] for name in abnormal_files]))
    total_events = 0
    missed_events = 0
    delays = []
    event_details = {}
    for name in abnormal_files:
        truth = labels[name]
        prediction = (cached_scores[name] > detector.threshold).astype(int)
        details = []
        for start, end in event_intervals(truth):
            total_events += 1
            hits = np.where(prediction[start : end + 1] == 1)[0]
            if len(hits) == 0:
                missed_events += 1
                details.append({"start": start, "end": end, "detected": False, "delay": None})
            else:
                delay = int(hits[0])
                delays.append(delay)
                details.append({"start": start, "end": end, "detected": True, "delay": delay})
        event_details[name] = details

    normal_names = ["WeldingTest_01_OK", "WeldingTest_02_OK"]
    normal_truth = np.concatenate([labels[name] for name in normal_names])
    normal_prediction = np.concatenate(
        [(cached_scores[name] > detector.threshold).astype(int) for name in normal_names]
    )
    normal_fp = int(normal_prediction.sum())
    normal_fpr = float(normal_fp / len(normal_truth))
    normal_false_alarm_events = int(
        sum(count_predicted_events((cached_scores[name] > detector.threshold).astype(int)) for name in normal_names)
    )

    metrics = {
        "model": detector.name,
        "threshold": float(detector.threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "macro_f1_03_04": macro_f1,
        "total_true_events": total_events,
        "missed_events": missed_events,
        "event_recall": float((total_events - missed_events) / total_events),
        "mean_detection_delay_rows": float(np.mean(delays)) if delays else float("inf"),
        "max_detection_delay_rows": int(max(delays)) if delays else None,
        "normal_false_positive_rows": normal_fp,
        "normal_false_alarm_events": normal_false_alarm_events,
        "normal_file_fpr": normal_fpr,
        "training_seconds": float(detector.training_seconds),
        "inference_ms_per_1000_rows": float(inference_seconds * 1000.0 / len(pooled_true) * 1000.0),
        "parameter_count": int(detector.parameter_count),
        "model_size_kb": float(detector.model_size_kb),
    }
    detail = {"per_file": per_file, "events": event_details}
    return metrics, pd.DataFrame(prediction_rows), detail


def plot_model_metrics(metrics: dict, output_path: Path) -> None:
    values = [metrics["precision"], metrics["recall"], metrics["f1"], metrics["fpr"]]
    labels = ["Precision", "Recall", "F1", "False positive rate"]
    colors = ["#2E86AB", "#4CAF50", "#F2A541", "#D1495B"]
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.25, 1.0]})
    bars = axes[0].bar(labels, values, color=colors)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Score")
    axes[0].set_title(f"{metrics['model']} evaluation metrics")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, min(value + 0.025, 1.01), f"{value:.4f}", ha="center")

    image = axes[1].imshow(matrix, cmap="Blues")
    axes[1].set_xticks([0, 1], labels=["Pred normal", "Pred anomaly"])
    axes[1].set_yticks([0, 1], labels=["True normal", "True anomaly"])
    axes[1].set_title("Confusion matrix")
    limit = matrix.max() / 2.0
    for row in range(2):
        for column in range(2):
            axes[1].text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > limit else "black",
                fontsize=12,
                fontweight="bold",
            )
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Macro F1 (03/04): {metrics['macro_f1_03_04']:.4f} | "
        f"Missed events: {metrics['missed_events']}/{metrics['total_true_events']} | "
        f"Normal false alarms: {metrics['normal_false_alarm_events']}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_combined_metrics(results: pd.DataFrame, output_path: Path) -> None:
    metric_columns = ["precision", "recall", "f1", "fpr"]
    labels = ["Precision", "Recall", "F1", "False positive rate"]
    x = np.arange(len(results))
    width = 0.18
    fig, ax = plt.subplots(figsize=(13, 6))
    for offset, (column, label) in enumerate(zip(metric_columns, labels)):
        ax.bar(x + (offset - 1.5) * width, results[column], width, label=label)
    ax.set_xticks(x, results["model"], rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Five-model predictive-maintenance anomaly detection comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    def format_value(value) -> str:
        if isinstance(value, (float, np.floating)):
            if math.isinf(float(value)):
                return "inf"
            return f"{float(value):.4f}"
        return str(value)

    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(format_value(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(results: pd.DataFrame, training_info: dict, winner: str, output_path: Path) -> None:
    display_columns = [
        "model",
        "missed_events",
        "normal_false_alarm_events",
        "macro_f1_03_04",
        "mean_detection_delay_rows",
        "training_seconds",
        "inference_ms_per_1000_rows",
        "parameter_count",
        "model_size_kb",
    ]
    visual_columns = ["model", "precision", "recall", "f1", "tn", "fp", "fn", "tp", "fpr"]
    lines = [
        "# 배터리팩 용접 이상탐지 모델 비교",
        "",
        f"최종 선정 모델: **{winner}**",
        "",
        "## 평가 원칙",
        "",
        "- 학습 데이터의 정상 39행 사이클을 시작 시각순으로 정렬했다.",
        f"- 앞 {training_info['training_cycles']}개 사이클은 모델 적합, 뒤 {training_info['calibration_cycles']}개 사이클은 임계값 보정에 사용했다.",
        "- 전 구간 RealPower가 0인 사이클은 정상 학습에서 제외했다.",
        "- 두 OK 파일과 두 NG 파일은 최종 평가에만 사용했다.",
        "- 모델 선정 우선순위는 이벤트 누락, 정상 파일 오경보 이벤트, 03/04 Macro F1, 탐지 지연, 추론 속도 순이다.",
        "",
        "## 모델 선정 지표",
        "",
        dataframe_to_markdown(results[display_columns]),
        "",
        "## 시각화 대상 지표",
        "",
        dataframe_to_markdown(results[visual_columns]),
        "",
        "## 해석 시 주의사항",
        "",
        "- Precision, Recall, F1과 혼동행렬은 OK 2개 및 NG 2개 파일을 합친 행 단위 결과다.",
        "- Macro F1은 03번과 04번 F1의 단순 평균으로, 긴 04번 이상 구간이 결과를 독점하지 않게 한다.",
        "- 정상 파일 오경보 이벤트는 연속된 오경보 구간을 한 번의 경보로 계산한다.",
        "- NHiTS_Z4는 가이드북의 20행 입력, 10행 예측, 64 hidden size, 저출력 구간 +930.5 변환 및 |Z|>4 규칙을 따른 경량 N-HiTS 재현 모델이다.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_full = read_signal(DATA_ROOT / "raw_data" / "train" / "Training_Data.csv")
    train, calibration, training_info = split_training_cycles(train_full)

    test_paths = {
        "WeldingTest_01_OK": DATA_ROOT / "raw_data" / "test" / "WeldingTest_01_OK.csv",
        "WeldingTest_02_OK": DATA_ROOT / "raw_data" / "test" / "WeldingTest_02_OK.csv",
        "WeldingTest_03_NG": DATA_ROOT / "raw_data" / "test" / "WeldingTest_03_NG.csv",
        "WeldingTest_04_NG": DATA_ROOT / "raw_data" / "test" / "WeldingTest_04_NG.csv",
    }
    test_frames = {name: read_signal(path) for name, path in test_paths.items()}
    labels = {
        "WeldingTest_01_OK": np.zeros(len(test_frames["WeldingTest_01_OK"]), dtype=int),
        "WeldingTest_02_OK": np.zeros(len(test_frames["WeldingTest_02_OK"]), dtype=int),
        "WeldingTest_03_NG": pd.read_csv(
            DATA_ROOT / "preprocessed" / "test" / "WeldingTest_03_NG_Label.csv"
        )["label"].to_numpy(dtype=int),
        "WeldingTest_04_NG": pd.read_csv(
            DATA_ROOT / "preprocessed" / "test" / "WeldingTest_04_NG_Label.csv"
        )["label"].to_numpy(dtype=int),
    }

    detectors: List[BaseDetector] = [
        NHiTSZDetector(),
        RobustZDetector(),
        LightGBMDetector(),
        EWMACUSUMDetector(),
        CycleAutoencoderDetector(),
    ]
    all_metrics = []
    all_predictions = []
    details = {}

    for detector in detectors:
        print(f"Fitting {detector.name}...", flush=True)
        detector.fit(train, calibration)
        metrics, predictions, detector_detail = evaluate_detector(detector, test_frames, labels)
        all_metrics.append(metrics)
        all_predictions.append(predictions)
        details[detector.name] = detector_detail
        plot_model_metrics(metrics, OUTPUT_DIR / f"{detector.name}_metrics.png")
        print(
            f"{detector.name}: F1={metrics['f1']:.4f}, macroF1={metrics['macro_f1_03_04']:.4f}, "
            f"missed_events={metrics['missed_events']}, normal_false_alarms={metrics['normal_false_alarm_events']}",
            flush=True,
        )

    results = pd.DataFrame(all_metrics)
    results["selection_delay"] = results["mean_detection_delay_rows"].replace([np.inf], 10**9)
    results = results.sort_values(
        by=[
            "missed_events",
            "normal_false_alarm_events",
            "macro_f1_03_04",
            "selection_delay",
            "inference_ms_per_1000_rows",
        ],
        ascending=[True, True, False, True, True],
        kind="stable",
    ).drop(columns=["selection_delay"]).reset_index(drop=True)
    results.insert(0, "rank", np.arange(1, len(results) + 1))
    winner = str(results.iloc[0]["model"])

    results.to_csv(OUTPUT_DIR / "model_comparison_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_predictions, ignore_index=True).to_csv(
        OUTPUT_DIR / "model_predictions.csv", index=False, encoding="utf-8-sig"
    )
    plot_combined_metrics(results, OUTPUT_DIR / "all_models_comparison.png")
    write_report(results, training_info, winner, OUTPUT_DIR / "model_comparison_report.md")
    payload = {
        "winner": winner,
        "training_split": training_info,
        "selection_priority": [
            "missed_events ascending",
            "normal_false_alarm_events ascending",
            "macro_f1_03_04 descending",
            "mean_detection_delay_rows ascending",
            "inference_ms_per_1000_rows ascending",
        ],
        "results": results.replace([np.inf, -np.inf], None).to_dict(orient="records"),
        "details": details,
    }
    (OUTPUT_DIR / "model_comparison_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Winner: {winner}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
