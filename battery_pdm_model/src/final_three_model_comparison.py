from __future__ import annotations

import json
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final_three_model_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import NHiTS
from pytorch_forecasting.metrics import MQF2DistributionLoss
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from compare_models import DATA_ROOT, LightGBMDetector, RobustZDetector, read_signal, add_cycle_id
from reproduce_guidebook import (
    CONTEXT_LENGTH,
    PREDICTION_LENGTH,
    first_horizon_error_series,
    guidebook_series,
    make_dataset,
)


SEED = 42
CALIBRATION_QUANTILE = 0.999
THRESHOLD_LABEL = "normal calibration score 99.9th percentile"


def split_all_cycles(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cycled = add_cycle_id(frame)
    cycles = []
    for cycle_id, group in cycled.groupby("cycle_id", sort=False):
        group = group.sort_values("PageNo").copy()
        if len(group) != 39 or group["PageNo"].tolist() != list(range(1, 40)):
            raise ValueError(f"Cycle {cycle_id} is incomplete; guidebook comparison keeps raw rows.")
        cycles.append((group["WorkingTime"].min(), group))
    cycles.sort(key=lambda item: item[0])
    cut = int(len(cycles) * 0.70)
    train = pd.concat([group for _, group in cycles[:cut]], ignore_index=True)
    calibration = pd.concat([group for _, group in cycles[cut:]], ignore_index=True)
    info = {
        "all_complete_cycles_kept": len(cycles),
        "train_cycles": cut,
        "calibration_cycles": len(cycles) - cut,
        "train_rows": len(train),
        "calibration_rows": len(calibration),
        "zero_power_cycles_kept": int(
            sum(group["RealPower"].eq(0).any() for _, group in cycles)
        ),
        "train_start": str(train["WorkingTime"].min()),
        "train_end": str(train["WorkingTime"].max()),
        "calibration_start": str(calibration["WorkingTime"].min()),
        "calibration_end": str(calibration["WorkingTime"].max()),
    }
    return train, calibration, info


def train_nhits(train_frame: pd.DataFrame) -> tuple[NHiTS, dict]:
    checkpoint_dir = OUTPUT_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    old_checkpoints = sorted(checkpoint_dir.glob("three-model-nhits-*.ckpt"))
    if old_checkpoints:
        path = old_checkpoints[-1]
        return NHiTS.load_from_checkpoint(path), {
            "checkpoint_reused": True,
            "checkpoint": str(path.resolve()),
            "training_seconds": 0.0,
        }

    series = guidebook_series(train_frame)
    training_cutoff = int(series["time_idx"].max() - 100 * PREDICTION_LENGTH)
    training, validation = make_dataset(series, training_cutoff)
    train_loader = training.to_dataloader(train=True, batch_size=128, num_workers=0)
    validation_loader = validation.to_dataloader(train=False, batch_size=128, num_workers=0)
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="three-model-nhits-{epoch:02d}-{val_loss:.4f}",
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
    start = time.perf_counter()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
    elapsed = time.perf_counter() - start
    best = NHiTS.load_from_checkpoint(checkpoint.best_model_path)
    return best, {
        "checkpoint_reused": False,
        "checkpoint": str(Path(checkpoint.best_model_path).resolve()),
        "training_seconds": elapsed,
        "epochs_completed": int(trainer.current_epoch),
        "inner_validation_rows": 1000,
        "guidebook_model_parameters": {
            "context_length": CONTEXT_LENGTH,
            "prediction_length": PREDICTION_LENGTH,
            "batch_size": 128,
            "max_epochs": 30,
            "limit_train_batches": 30,
            "learning_rate": 0.04,
            "weight_decay": 0.01,
            "hidden_size": 64,
            "loss": "MQF2DistributionLoss",
            "power_transform": "RealPower < 1300 => +930.5",
        },
    }


def calc_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def regions(values: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(values, dtype=int)
    padded = np.pad(values, (1, 1))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def event_summary(labels: dict[str, np.ndarray], preds: dict[str, np.ndarray]) -> tuple[int, int, int]:
    total, detected, false_alarm_events = 0, 0, 0
    for name, truth in labels.items():
        prediction = preds[name]
        if truth.sum() == 0:
            false_alarm_events += len(regions(prediction))
        for start, end in regions(truth):
            total += 1
            if prediction[start : end + 1].any():
                detected += 1
        normal_positive = ((truth == 0) & (prediction == 1)).astype(int)
        false_alarm_events += len(regions(normal_positive)) if truth.sum() else 0
    return total, detected, false_alarm_events


def main() -> None:
    pl.seed_everything(SEED, workers=True)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    torch.set_float32_matmul_precision("medium")

    full_train = read_signal(DATA_ROOT / "raw_data" / "train" / "Training_Data.csv")
    train, calibration, split_info = split_all_cycles(full_train)
    file_names = [
        "WeldingTest_01_OK",
        "WeldingTest_02_OK",
        "WeldingTest_03_NG",
        "WeldingTest_04_NG",
    ]
    tests = {name: read_signal(DATA_ROOT / "raw_data" / "test" / f"{name}.csv") for name in file_names}
    labels = {
        name: (
            pd.read_csv(DATA_ROOT / "preprocessed" / "test" / f"{name}_Label.csv")["label"].to_numpy(dtype=int)
            if name.endswith("NG")
            else np.zeros(len(tests[name]), dtype=int)
        )
        for name in file_names
    }

    robust = RobustZDetector().fit(train, calibration)
    lightgbm = LightGBMDetector().fit(train, calibration)
    nhits, nhits_fit = train_nhits(train)

    _, calibration_errors = first_horizon_error_series(
        nhits, calibration, testing_cutoff=CONTEXT_LENGTH + PREDICTION_LENGTH - 1
    )
    nhits_error_mean = float(np.mean(calibration_errors))
    nhits_error_std = float(max(np.std(calibration_errors), 1e-12))
    nhits_cal_scores = np.abs((calibration_errors - nhits_error_mean) / nhits_error_std)
    nhits_threshold = float(np.quantile(nhits_cal_scores, CALIBRATION_QUANTILE, method="higher"))

    preds: dict[str, dict[str, np.ndarray]] = {"RobustZ": {}, "LightGBM": {}, "N-HiTS": {}}
    nhits_coverage = {}
    for name, frame in tests.items():
        preds["RobustZ"][name] = robust.predict(frame)
        preds["LightGBM"][name] = lightgbm.predict(frame)
        indices, errors = first_horizon_error_series(
            nhits, frame, testing_cutoff=CONTEXT_LENGTH + PREDICTION_LENGTH - 1
        )
        prediction = np.zeros(len(frame), dtype=int)
        z = np.abs((errors - nhits_error_mean) / nhits_error_std)
        prediction[indices] = (z > nhits_threshold).astype(int)
        preds["N-HiTS"][name] = prediction
        nhits_coverage[name] = {
            "total_rows": int(len(frame)),
            "scored_rows": int(len(indices)),
            "unscored_rows_assumed_normal": int(len(frame) - len(indices)),
            "first_scored_index": int(indices.min()) if len(indices) else None,
            "last_scored_index": int(indices.max()) if len(indices) else None,
        }

    y_all = np.concatenate([labels[n] for n in file_names])
    rows = []
    for model_name in ["RobustZ", "LightGBM", "N-HiTS"]:
        pred_all = np.concatenate([preds[model_name][n] for n in file_names])
        metrics = calc_binary_metrics(y_all, pred_all)
        event_total, event_detected, false_alarm_events = event_summary(
            labels, preds[model_name]
        )
        rows.append({
            "model": model_name,
            **metrics,
            "event_detected": event_detected,
            "event_total": event_total,
            "normal_false_alarm_events": false_alarm_events,
        })

    thresholds = {
        "calibration_quantile": CALIBRATION_QUANTILE,
        "rule": THRESHOLD_LABEL,
        "RobustZ": float(robust.threshold),
        "LightGBM": float(lightgbm.threshold),
        "N-HiTS": nhits_threshold,
        "N-HiTS_calibration_error_mean": nhits_error_mean,
        "N-HiTS_calibration_error_std": nhits_error_std,
        "N-HiTS_calibration_scored_rows": int(len(nhits_cal_scores)),
    }
    payload = {
        "protocol": {
            "training_split": "chronological cycle split: first 70% train, last 30% normal calibration",
            "all_training_rows_kept": True,
            "zero_power_cycle_kept": True,
            "iqr": "guidebook c=4 check had 0 rows; no rows removed",
            "common_threshold_rule": THRESHOLD_LABEL,
            "threshold_quantile": CALIBRATION_QUANTILE,
            "test_files": file_names,
            "test_rows_evaluated": int(len(y_all)),
            "N-HiTS_unscored_rule": "keep every test row; N-HiTS rows without a forecast score are predicted normal",
            "event_rule": "an anomaly event is detected if at least one row within its contiguous labeled interval is predicted anomalous",
            "false_alarm_event_rule": "count each contiguous predicted-positive run on label-0 rows, including normal portions of NG files",
        },
        "split": split_info,
        "thresholds": thresholds,
        "nhits_training": nhits_fit,
        "nhits_test_coverage": nhits_coverage,
        "results": rows,
    }
    (OUTPUT_DIR / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "metrics.csv", index=False, encoding="utf-8-sig")
    guidebook_path = PROJECT_ROOT / "outputs" / "guidebook_reproduction" / "guidebook_reproduction_results.json"
    guidebook_result = json.loads(guidebook_path.read_text(encoding="utf-8"))
    nhits_03 = guidebook_result["guidebook_nhits_z4"]["per_file"]["WeldingTest_03_NG"]
    z04 = guidebook_result["guidebook_two_band_z4"]["published_protocol_04_first_n_minus_20_rows"]
    lines = [
        "# 가이드북 재현 확인 및 3개 모델 공통 비교",
        "",
        "## 가이드북 재현 확인",
        "",
        "| 가이드북 결과 | 가이드북 기재 | 동일 설정 재학습 결과 | 재현 여부 |",
        "| --- | ---: | ---: | --- |",
        f"| N-HiTS + Z-score, WeldingTest_03_NG F1 | {guidebook_result['guidebook_nhits_z4']['published_reference_03']['f1']:.4f} | {nhits_03['f1']:.4f} | 미재현 |",
        f"| 2구간 Z-score, WeldingTest_04_NG F1 | {guidebook_result['guidebook_two_band_z4']['published_reference_04']['f1']:.4f} | {z04['f1']:.4f} | 재현 |",
        "",
        "가이드북 N-HiTS 모델 구조와 주요 학습값은 적용했지만 03번의 기재 성능은 재현되지 않았다. 03번 재현 결과는 Accuracy "
        f"{nhits_03['accuracy']:.4f}, Precision {nhits_03['precision']:.4f}, Recall {nhits_03['recall']:.4f}, F1 {nhits_03['f1']:.4f} "
        f"(TN {nhits_03['tn']}, FP {nhits_03['fp']}, FN {nhits_03['fn']}, TP {nhits_03['tp']})다. "
        "가이드북의 코드 흐름은 파일별 테스트 오차 평균·표준편차를 사용하고 12행 이동하는 반면, 설명문에는 학습 오차 표준편차를 쓰라고 적혀 있어 두 설명 사이에 차이가 있다.",
        "",
        "## 3개 모델 공통 비교",
        "",
        f"- 학습 데이터: 시간 순서 기준 앞 70% ({split_info['train_cycles']:,}개 사이클, {split_info['train_rows']:,}행). RealPower=0 사이클을 포함했다.",
        f"- 정상 보정 데이터: 뒤 30% ({split_info['calibration_cycles']:,}개 사이클, {split_info['calibration_rows']:,}행).",
        "- 임계값: 각 모델의 정상 보정 점수 상위 0.1%(99.9 백분위). 테스트 라벨로 임계값을 바꾸지 않았다.",
        f"- 평가: 네 테스트 파일 전체 {len(y_all):,}행. N-HiTS가 점수를 만들지 못하는 행도 제거하지 않고 정상 판정으로 두었다.",
        "- 이벤트 탐지: 정답 이상 구간 안에서 한 행 이상 이상으로 판정하면 해당 이벤트를 탐지한 것으로 계산했다.",
        "- 정상 오경보: 실제 정상 라벨인 구간에서 연속으로 발생한 오경보 묶음 수다. FP는 오경보 행 수다.",
        "",
        "| 모델 | Accuracy | Precision | Recall | F1 | 이벤트 탐지 | TN | FP | FN | TP | 정상 오경보(이벤트) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['event_detected']}/{row['event_total']} | {row['tn']} | "
            f"{row['fp']} | {row['fn']} | {row['tp']} | {row['normal_false_alarm_events']} |"
        )
    lines.extend([
        "",
        "## 해석",
        "",
        "RobustZ와 LightGBM은 Accuracy·Precision·Recall·F1 및 혼동행렬에서 같은 결과를 냈다. 이 데이터에서는 LightGBM이 RobustZ보다 나은 성능을 보였다고 할 근거가 없다.",
        "N-HiTS는 행 단위 F1이 높지만, 20개 이상 이벤트 중 긴 04번 이벤트 한 개만 탐지하고 03번의 고립 이벤트 19개는 탐지하지 못했다. 따라서 F1만 보면 이벤트 탐지 능력을 과대평가할 수 있다.",
        "가이드북 재현 지표와 공통 비교 지표는 학습·임계값·정렬 규칙이 다르므로 서로 직접 순위를 매기지 않는다. 가이드북 재현을 먼저 보고하고, 세 모델 공통 비교는 별도 실험 결과로 제시한다.",
        "",
    ])
    (OUTPUT_DIR / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
