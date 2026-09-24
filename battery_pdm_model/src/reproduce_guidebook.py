from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "guidebook_reproduction"
MPL_CONFIG_DIR = OUTPUT_DIR / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import NHiTS, TimeSeriesDataSet
from pytorch_forecasting.metrics import MQF2DistributionLoss
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


DATA_ROOT = PROJECT_ROOT / "Dataset_전자부품(배터리팩) 예지보전 AI 데이터셋" / "data"
SEED = 42
CONTEXT_LENGTH = 20
PREDICTION_LENGTH = 10
THRESHOLD = 4.0


def read_raw(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    return frame


def guidebook_series(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce guidebook codes 31-32 without the unused synthetic AR values."""
    values = frame["RealPower"].to_numpy(dtype=np.float64).copy()
    values[values < 1300.0] += 930.5
    return pd.DataFrame(
        {
            "series": np.zeros(len(values), dtype=np.int64),
            "time_idx": np.arange(len(values), dtype=np.int64),
            "value": values,
        }
    )


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "evaluated_rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def as_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "output"):
        return as_tensor(value.output)
    if isinstance(value, (tuple, list)):
        return as_tensor(value[0])
    raise TypeError(f"Unsupported prediction container: {type(value)!r}")


def make_dataset(data: pd.DataFrame, cutoff: int) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    base = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= cutoff],
        time_idx="time_idx",
        target="value",
        group_ids=["series"],
        time_varying_unknown_reals=["value"],
        max_encoder_length=CONTEXT_LENGTH,
        max_prediction_length=PREDICTION_LENGTH,
    )
    future = TimeSeriesDataSet.from_dataset(base, data, min_prediction_idx=cutoff + 1)
    return base, future


def train_guidebook_nhits(train_frame: pd.DataFrame) -> tuple[NHiTS, dict]:
    data = guidebook_series(train_frame)
    training_cutoff = int(data["time_idx"].max() - 100 * PREDICTION_LENGTH)
    training, validation = make_dataset(data, training_cutoff)
    train_loader = training.to_dataloader(train=True, batch_size=128, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=128, num_workers=0)

    checkpoint = ModelCheckpoint(
        dirpath=OUTPUT_DIR / "checkpoints",
        filename="guidebook-nhits-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stop = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
    )
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="cpu",
        devices=1,
        enable_model_summary=True,
        gradient_clip_val=1.0,
        callbacks=[early_stop, checkpoint],
        limit_train_batches=30,
        enable_checkpointing=True,
        logger=False,
        deterministic=True,
        default_root_dir=OUTPUT_DIR,
    )
    model = NHiTS.from_dataset(
        training,
        learning_rate=0.04,
        log_interval=10,
        log_val_interval=1,
        weight_decay=1e-2,
        backcast_loss_ratio=0.0,
        hidden_size=64,
        loss=MQF2DistributionLoss(prediction_length=PREDICTION_LENGTH),
    )

    started = time.perf_counter()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    elapsed = time.perf_counter() - started
    best_model = NHiTS.load_from_checkpoint(checkpoint.best_model_path)
    return best_model, {
        "training_cutoff": training_cutoff,
        "epochs_completed": int(trainer.current_epoch),
        "training_seconds": elapsed,
        "best_model_path": str(Path(checkpoint.best_model_path).resolve()),
        "guidebook_config": {
            "context_length": CONTEXT_LENGTH,
            "prediction_length": PREDICTION_LENGTH,
            "batch_size": 128,
            "max_epochs": 30,
            "limit_train_batches": 30,
            "learning_rate": 0.04,
            "weight_decay": 0.01,
            "hidden_size": 64,
            "loss": "MQF2DistributionLoss",
            "threshold": THRESHOLD,
        },
    }


def first_horizon_error_series(
    model: NHiTS, frame: pd.DataFrame, testing_cutoff: int
) -> tuple[np.ndarray, np.ndarray]:
    data = guidebook_series(frame)
    _, testing = make_dataset(data, testing_cutoff)
    loader = testing.to_dataloader(train=False, batch_size=128, num_workers=0)
    actuals = torch.cat([y[0] for _, y in iter(loader)]).detach().cpu()
    result = model.predict(
        loader,
        return_index=True,
        trainer_kwargs={
            "accelerator": "cpu",
            "devices": 1,
            "logger": False,
            "enable_progress_bar": False,
        },
    )
    predicted = as_tensor(result).detach().cpu()
    time_indices = result.index["time_idx"].to_numpy(dtype=int)
    errors = predicted[:, 0].numpy() - actuals[:, 0].numpy()
    return time_indices, errors


def first_horizon_errors(model: NHiTS, frame: pd.DataFrame, testing_cutoff: int) -> np.ndarray:
    _, errors = first_horizon_error_series(model, frame, testing_cutoff)
    return errors


def guidebook_nhits_predictions(
    model: NHiTS,
    frame: pd.DataFrame,
    testing_cutoff: int,
    output_length: int,
    shift: int = 12,
) -> tuple[np.ndarray, dict]:
    errors = first_horizon_errors(model, frame, testing_cutoff)
    error_mean = float(np.mean(errors))
    error_std = float(np.std(errors))
    z_scores = (errors - error_mean) / max(error_std, 1e-12)
    prediction = np.zeros(output_length, dtype=int)
    usable = min(len(errors), max(0, output_length - shift))
    prediction[shift : shift + usable] = (np.abs(z_scores[:usable]) > THRESHOLD).astype(int)
    return prediction, {
        "prediction_error_count": int(len(errors)),
        "alignment_shift": shift,
        "test_error_mean": error_mean,
        "test_error_std": error_std,
        "note": "The guidebook standardizes with each test file's own errors in codes 99/107.",
    }


def guidebook_two_band_z_fit(train_frame: pd.DataFrame) -> dict:
    values = train_frame["RealPower"].to_numpy(dtype=float)[:-20]
    low = values[values < 1200.0]
    high = values[values >= 1200.0]
    return {
        "low_mean": float(low.mean()),
        "low_std": float(low.std()),
        "high_mean": float(high.mean()),
        "high_std": float(high.std()),
    }


def guidebook_two_band_z_predict(frame: pd.DataFrame, fitted: dict, truncate_last_20: bool) -> np.ndarray:
    values = frame["RealPower"].to_numpy(dtype=float)
    if truncate_last_20:
        values = values[:-20]
    low_z = np.abs(values - fitted["low_mean"]) / fitted["low_std"]
    high_z = np.abs(values - fitted["high_mean"]) / fitted["high_std"]
    return (np.minimum(low_z, high_z) >= THRESHOLD).astype(int)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(SEED, workers=True)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    torch.set_float32_matmul_precision("medium")

    train = read_raw(DATA_ROOT / "raw_data" / "train" / "Training_Data.csv")
    test_paths = {
        "WeldingTest_01_OK": DATA_ROOT / "raw_data" / "test" / "WeldingTest_01_OK.csv",
        "WeldingTest_02_OK": DATA_ROOT / "raw_data" / "test" / "WeldingTest_02_OK.csv",
        "WeldingTest_03_NG": DATA_ROOT / "raw_data" / "test" / "WeldingTest_03_NG.csv",
        "WeldingTest_04_NG": DATA_ROOT / "raw_data" / "test" / "WeldingTest_04_NG.csv",
    }
    tests = {name: read_raw(path) for name, path in test_paths.items()}
    labels = {
        "WeldingTest_01_OK": np.zeros(len(tests["WeldingTest_01_OK"]), dtype=int),
        "WeldingTest_02_OK": np.zeros(len(tests["WeldingTest_02_OK"]), dtype=int),
        "WeldingTest_03_NG": pd.read_csv(
            DATA_ROOT / "preprocessed" / "test" / "WeldingTest_03_NG_Label.csv"
        )["label"].to_numpy(dtype=int),
        "WeldingTest_04_NG": pd.read_csv(
            DATA_ROOT / "preprocessed" / "test" / "WeldingTest_04_NG_Label.csv"
        )["label"].to_numpy(dtype=int),
    }

    nhits, training_info = train_guidebook_nhits(train)
    nhits_metrics = {}
    nhits_details = {}
    for name, frame in tests.items():
        cutoff = 100 if name.endswith("OK") else 50
        prediction, detail = guidebook_nhits_predictions(
            nhits, frame, testing_cutoff=cutoff, output_length=len(frame), shift=12
        )
        nhits_metrics[name] = binary_metrics(labels[name], prediction)
        nhits_details[name] = detail

    z_fitted = guidebook_two_band_z_fit(train)
    z_metrics_full = {}
    for name, frame in tests.items():
        prediction = guidebook_two_band_z_predict(frame, z_fitted, truncate_last_20=False)
        z_metrics_full[name] = binary_metrics(labels[name], prediction)

    guidebook_04_prediction = guidebook_two_band_z_predict(
        tests["WeldingTest_04_NG"], z_fitted, truncate_last_20=True
    )
    guidebook_04_metrics = binary_metrics(
        labels["WeldingTest_04_NG"][: len(guidebook_04_prediction)], guidebook_04_prediction
    )

    result = {
        "training": training_info,
        "guidebook_nhits_z4": {
            "per_file": nhits_metrics,
            "details": nhits_details,
            "published_reference_03": {
                "accuracy": 0.9960,
                "precision": 0.8947,
                "recall": 0.8947,
                "f1": 0.8947,
            },
        },
        "guidebook_two_band_z4": {
            "fitted": z_fitted,
            "per_file_full_length": z_metrics_full,
            "published_protocol_04_first_n_minus_20_rows": guidebook_04_metrics,
            "published_reference_04": {
                "accuracy": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
            },
        },
        "methodology_notes": [
            "Guidebook N-HiTS uses official pytorch_forecasting.NHiTS with MQF2DistributionLoss.",
            "Guidebook code standardizes N-HiTS errors using each test file's own mean and standard deviation.",
            "Guidebook code evaluates the two-band Z method on WeldingTest_04_NG after dropping the final 20 rows.",
            "These published-protocol results must be separated from leakage-free common-protocol comparison.",
        ],
    }
    output_path = OUTPUT_DIR / "guidebook_reproduction_results.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
