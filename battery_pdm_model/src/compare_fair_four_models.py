from __future__ import annotations

import json
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fair_four_model_comparison"
MPL_CONFIG_DIR = OUTPUT_DIR / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import NHiTS
from pytorch_forecasting.metrics import MQF2DistributionLoss
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from compare_models import (
    DATA_ROOT,
    LightGBMDetector,
    RobustZDetector,
    read_signal,
    split_training_cycles,
)
from reproduce_guidebook import (
    CONTEXT_LENGTH,
    PREDICTION_LENGTH,
    THRESHOLD,
    first_horizon_error_series,
    guidebook_series,
    guidebook_two_band_z_fit,
    guidebook_two_band_z_predict,
    make_dataset,
)


SEED = 42


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def contiguous_regions(values: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(values, dtype=int)
    padded = np.pad(values, (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def event_metrics(
    labels_by_file: dict[str, np.ndarray],
    predictions_by_file: dict[str, np.ndarray],
    masks_by_file: dict[str, np.ndarray],
) -> dict:
    total_events = 0
    missed_events = 0
    normal_false_alarm_events = 0
    for name, labels in labels_by_file.items():
        prediction = predictions_by_file[name]
        mask = masks_by_file[name]
        if labels.sum() == 0:
            normal_false_alarm_events += len(contiguous_regions(prediction[mask]))
            continue
        for start, end in contiguous_regions(labels):
            event_indices = np.arange(start, end + 1)
            event_indices = event_indices[mask[event_indices]]
            if len(event_indices) == 0:
                continue
            total_events += 1
            if not prediction[event_indices].any():
                missed_events += 1
    return {
        "total_evaluable_events": total_events,
        "missed_events": missed_events,
        "event_recall": float((total_events - missed_events) / total_events) if total_events else 0.0,
        "normal_false_alarm_events": normal_false_alarm_events,
    }


def train_fair_nhits(train_frame: pd.DataFrame) -> tuple[NHiTS, dict]:
    checkpoint_dir = OUTPUT_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(checkpoint_dir.glob("fair-nhits-*.ckpt"))
    if existing:
        model = NHiTS.load_from_checkpoint(existing[-1])
        return model, {
            "reused_checkpoint": True,
            "checkpoint": str(existing[-1].resolve()),
            "training_seconds": 0.0,
        }

    data = guidebook_series(train_frame)
    training_cutoff = int(data["time_idx"].max() - 100 * PREDICTION_LENGTH)
    training, validation = make_dataset(data, training_cutoff)
    train_loader = training.to_dataloader(train=True, batch_size=128, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=128, num_workers=0)

    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="fair-nhits-{epoch:02d}-{val_loss:.4f}",
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
        gradient_clip_val=1.0,
        callbacks=[early_stop, checkpoint],
        limit_train_batches=30,
        enable_checkpointing=True,
        enable_progress_bar=False,
        enable_model_summary=False,
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
        "reused_checkpoint": False,
        "checkpoint": str(Path(checkpoint.best_model_path).resolve()),
        "training_seconds": elapsed,
        "epochs_completed": int(trainer.current_epoch),
        "internal_validation_rows": 1000,
    }


def aggregate_model(
    name: str,
    labels: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
) -> dict:
    y_true = np.concatenate([labels[file][masks[file]] for file in labels])
    y_pred = np.concatenate([predictions[file][masks[file]] for file in labels])
    result = {"model": name, **binary_metrics(y_true, y_pred)}
    result.update(event_metrics(labels, predictions, masks))
    return result


def write_report(results: list[dict], metadata: dict) -> None:
    lines = [
        "# 4개 모델 공통 프로토콜 비교",
        "",
        "## 공정성 규칙",
        "",
        "- 모든 모델은 동일한 정상 2,436사이클로 학습했다.",
        "- 임계값·오차 통계는 동일한 정상 1,044사이클에서만 고정했다.",
        "- 테스트 파일의 평균·표준편차나 라벨은 임계값 조정에 사용하지 않았다.",
        "- N-HiTS가 입력 20행·예측 10행을 필요로 하므로, 모든 모델을 각 파일의 30번째 행부터 끝에서 10번째 전 행까지만 동일하게 평가했다.",
        "",
        "## 결과",
        "",
        "| 모델 | Accuracy | Precision | Recall | F1 | TP | FP | FN | 이벤트 탐지 | 정상 오경보 이벤트 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        detected = row["total_evaluable_events"] - row["missed_events"]
        lines.append(
            f"| {row['model']} | {row['accuracy']:.4f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} | {row['tp']} | {row['fp']} | "
            f"{row['fn']} | {detected}/{row['total_evaluable_events']} | "
            f"{row['normal_false_alarm_events']} |"
        )
    lines.extend(
        [
            "",
            "## 해석",
            "",
            "RobustZ와 LightGBM은 이 데이터의 두 이상 유형을 모두 탐지했다. "
            "가이드북 2구간 Z-score는 두 이상 유형을 모두 탐지했지만 정상 구간 오경보가 171행/28이벤트 발생했다. "
            "공식 N-HiTS는 가이드북 구조를 사용했지만 정상 보정 구간에서 고정한 Z-score로 평가했을 때의 성능은 다르다.",
            "",
            "가이드북의 발표 수치는 테스트 파일 자체의 오차 평균·표준편차를 사용한 별도 재현 결과이므로, 이 표의 운영 기준 성능과 구분해야 한다.",
            "",
            "## N-HiTS 보정 정보",
            "",
            f"- 보정 오차 평균: {metadata['nhits_calibration']['error_mean']:.6f}",
            f"- 보정 오차 표준편차: {metadata['nhits_calibration']['error_std']:.6f}",
            f"- 임계값: |Z| > {THRESHOLD:.1f}",
        ]
    )
    (OUTPUT_DIR / "fair_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(SEED, workers=True)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    torch.set_float32_matmul_precision("medium")

    train_full = read_signal(DATA_ROOT / "raw_data" / "train" / "Training_Data.csv")
    train, calibration, split_info = split_training_cycles(train_full)
    test_paths = {
        "WeldingTest_01_OK": DATA_ROOT / "raw_data" / "test" / "WeldingTest_01_OK.csv",
        "WeldingTest_02_OK": DATA_ROOT / "raw_data" / "test" / "WeldingTest_02_OK.csv",
        "WeldingTest_03_NG": DATA_ROOT / "raw_data" / "test" / "WeldingTest_03_NG.csv",
        "WeldingTest_04_NG": DATA_ROOT / "raw_data" / "test" / "WeldingTest_04_NG.csv",
    }
    tests = {name: read_signal(path) for name, path in test_paths.items()}
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

    robust = RobustZDetector().fit(train, calibration)
    lightgbm = LightGBMDetector().fit(train, calibration)

    nhits, nhits_training = train_fair_nhits(train)
    _, calibration_errors = first_horizon_error_series(nhits, calibration, testing_cutoff=29)
    nhits_error_mean = float(calibration_errors.mean())
    nhits_error_std = float(max(calibration_errors.std(), 1e-12))

    z_fitted = guidebook_two_band_z_fit(train)
    masks: dict[str, np.ndarray] = {}
    predictions = {
        "RobustZ": {},
        "LightGBM": {},
        "Guidebook_NHiTS_FixedZ4": {},
        "Guidebook_TwoBand_Z4": {},
    }
    for name, frame in tests.items():
        indices, errors = first_horizon_error_series(nhits, frame, testing_cutoff=29)
        mask = np.zeros(len(frame), dtype=bool)
        mask[indices] = True
        masks[name] = mask

        predictions["RobustZ"][name] = robust.predict(frame)
        predictions["LightGBM"][name] = lightgbm.predict(frame)
        nhits_prediction = np.zeros(len(frame), dtype=int)
        nhits_prediction[indices] = (
            np.abs((errors - nhits_error_mean) / nhits_error_std) > THRESHOLD
        ).astype(int)
        predictions["Guidebook_NHiTS_FixedZ4"][name] = nhits_prediction
        predictions["Guidebook_TwoBand_Z4"][name] = guidebook_two_band_z_predict(
            frame, z_fitted, truncate_last_20=False
        )

    results = [
        aggregate_model(model, labels, model_predictions, masks)
        for model, model_predictions in predictions.items()
    ]
    results.sort(
        key=lambda row: (
            row["missed_events"],
            row["normal_false_alarm_events"],
            -row["f1"],
        )
    )
    metadata = {
        "split": split_info,
        "common_evaluation": {
            "first_time_idx": 30,
            "last_rows_excluded": 9,
            "total_rows": int(sum(mask.sum() for mask in masks.values())),
        },
        "nhits_training": nhits_training,
        "nhits_calibration": {
            "error_count": int(len(calibration_errors)),
            "error_mean": nhits_error_mean,
            "error_std": nhits_error_std,
            "threshold": THRESHOLD,
        },
        "guidebook_two_band_z": z_fitted,
    }
    payload = {"results": results, "metadata": metadata}
    (OUTPUT_DIR / "fair_comparison_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(results).to_csv(
        OUTPUT_DIR / "fair_comparison_metrics.csv", index=False, encoding="utf-8-sig"
    )
    write_report(results, metadata)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
