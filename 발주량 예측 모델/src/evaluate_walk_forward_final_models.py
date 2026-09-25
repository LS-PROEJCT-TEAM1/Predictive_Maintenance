from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd

from train_evaluate_final_models import (
    MODEL_NAMES,
    REFERENCE_MODEL,
    build_sequences,
    calculate_metrics,
    load_daily_data,
    set_seed,
    train_and_predict,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "final_walk_forward"
FOLD_COUNT = 5
INITIAL_HISTORY_DATES = 9


def make_expanding_folds(frame: pd.DataFrame) -> list[dict[str, object]]:
    dates = np.array(sorted(pd.to_datetime(frame["target_date"]).unique()))
    if len(dates) < INITIAL_HISTORY_DATES + FOLD_COUNT:
        raise ValueError("Not enough target dates for five expanding walk-forward folds.")

    test_blocks = np.array_split(dates[INITIAL_HISTORY_DATES:], FOLD_COUNT)
    folds: list[dict[str, object]] = []
    for fold_number, test_dates in enumerate(test_blocks, start=1):
        test_start = pd.Timestamp(test_dates[0])
        test_end = pd.Timestamp(test_dates[-1])
        history_dates = dates[dates < np.datetime64(test_start)]
        validation_count = max(1, int(round(len(history_dates) * 0.20)))
        validation_count = min(validation_count, max(1, len(history_dates) - 2))
        train_dates = history_dates[:-validation_count]
        validation_dates = history_dates[-validation_count:]
        if len(train_dates) < 2:
            raise ValueError(f"Fold {fold_number} has too few training dates.")

        train_candidate = frame[frame["target_date"].isin(train_dates)]
        learned_parts = set(train_candidate["part_number"].unique())
        raw_test = frame[frame["target_date"].isin(test_dates)]
        cold_start = raw_test[~raw_test["part_number"].isin(learned_parts)]

        fold_frame = frame[
            frame["target_date"].isin(np.concatenate([train_dates, validation_dates, test_dates]))
            & frame["part_number"].isin(learned_parts)
        ].copy()
        folds.append(
            {
                "fold": fold_number,
                "frame": fold_frame,
                "train_dates": train_dates,
                "validation_dates": validation_dates,
                "test_dates": test_dates,
                "test_start": test_start,
                "test_end": test_end,
                "cold_start_rows": int(len(cold_start)),
                "cold_start_parts": sorted(cold_start["part_number"].unique()),
            }
        )
    return folds


def metrics_for_predictions(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    y_true = prediction_frame["actual"].to_numpy(dtype=float)
    for model in MODEL_NAMES + [REFERENCE_MODEL]:
        rows.append(
            {
                "Model": model,
                "N": int(len(prediction_frame)),
                "Parts": int(prediction_frame["part_number"].nunique()),
                **calculate_metrics(y_true, prediction_frame[model].to_numpy(dtype=float)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed()
    daily, _ = load_daily_data()
    frame, sequences, _, _ = build_sequences(daily)

    sequence_lookup = {
        (row.part_number, pd.Timestamp(row.origin_date), pd.Timestamp(row.target_date)): sequence
        for row, sequence in zip(
            frame[["part_number", "origin_date", "target_date"]].itertuples(index=False),
            sequences,
        )
    }

    prediction_frames: list[pd.DataFrame] = []
    fold_metrics: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []

    for fold_info in make_expanding_folds(frame):
        fold = int(fold_info["fold"])
        fold_frame = fold_info["frame"].sort_values(
            ["target_date", "part_number", "origin_date"]
        ).reset_index(drop=True)
        fold_sequences = np.stack(
            [
                sequence_lookup[
                    (row.part_number, pd.Timestamp(row.origin_date), pd.Timestamp(row.target_date))
                ]
                for row in fold_frame[["part_number", "origin_date", "target_date"]].itertuples(
                    index=False
                )
            ]
        ).astype(np.float32)

        train_mask = fold_frame["target_date"].isin(fold_info["train_dates"]).to_numpy()
        validation_mask = fold_frame["target_date"].isin(
            fold_info["validation_dates"]
        ).to_numpy()
        test_mask = fold_frame["target_date"].isin(fold_info["test_dates"]).to_numpy()
        if not train_mask.any() or not validation_mask.any() or not test_mask.any():
            raise ValueError(f"Fold {fold} contains an empty split.")

        set_seed()
        predictions, models, metadata = train_and_predict(
            fold_frame,
            fold_sequences,
            train_mask,
            validation_mask,
            test_mask,
        )
        test_frame = fold_frame.loc[
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
        test_frame = test_frame.rename(columns={"target": "actual"})
        test_frame.insert(0, "Fold", fold)
        for model, values in predictions.items():
            test_frame[model] = values
        prediction_frames.append(test_frame)

        current_metrics = metrics_for_predictions(test_frame)
        current_metrics.insert(0, "Fold", fold)
        current_metrics.insert(1, "Train_Start", str(fold_frame.loc[train_mask, "target_date"].min().date()))
        current_metrics.insert(2, "Train_End", str(fold_frame.loc[train_mask, "target_date"].max().date()))
        current_metrics.insert(3, "Validation_Start", str(fold_frame.loc[validation_mask, "target_date"].min().date()))
        current_metrics.insert(4, "Validation_End", str(fold_frame.loc[validation_mask, "target_date"].max().date()))
        current_metrics.insert(5, "Test_Start", str(fold_frame.loc[test_mask, "target_date"].min().date()))
        current_metrics.insert(6, "Test_End", str(fold_frame.loc[test_mask, "target_date"].max().date()))
        fold_metrics.append(current_metrics)

        coverage_rows.append(
            {
                "Fold": fold,
                "Train_Rows": int(train_mask.sum()),
                "Train_Parts": int(fold_frame.loc[train_mask, "part_number"].nunique()),
                "Validation_Rows": int(validation_mask.sum()),
                "Validation_Parts": int(
                    fold_frame.loc[validation_mask, "part_number"].nunique()
                ),
                "Test_Rows": int(test_mask.sum()),
                "Test_Parts": int(fold_frame.loc[test_mask, "part_number"].nunique()),
                "Excluded_Cold_Start_Rows": int(fold_info["cold_start_rows"]),
                "Excluded_Cold_Start_Parts": len(fold_info["cold_start_parts"]),
                "Excluded_Cold_Start_Part_List": ", ".join(fold_info["cold_start_parts"]),
                "XGBoost_Trees": metadata["xgboost_best_trees"],
                "LightGBM_Trees": metadata["lightgbm_best_trees"],
                "CatBoost_Trees": metadata["catboost_best_trees"],
                "LSTM_Best_Epoch": metadata["lstm"]["best_epoch"],
            }
        )
        print(f"Walk-forward fold {fold}/{FOLD_COUNT} complete", flush=True)
        del models
        gc.collect()

    predictions_all = pd.concat(prediction_frames, ignore_index=True)
    metrics_by_fold = pd.concat(fold_metrics, ignore_index=True)
    pooled_metrics = metrics_for_predictions(predictions_all).sort_values("MAE")
    pooled_metrics["MAE_Rank"] = np.arange(1, len(pooled_metrics) + 1)
    pooled_metrics["Eligible_For_Ranking"] = pooled_metrics["Model"].isin(MODEL_NAMES)
    eligible_models = pooled_metrics.loc[
        pooled_metrics["Eligible_For_Ranking"], "Model"
    ].tolist()
    eligible_rank = {model: rank for rank, model in enumerate(eligible_models, start=1)}
    pooled_metrics["Eligible_MAE_Rank"] = pooled_metrics["Model"].map(eligible_rank)
    eligible = pooled_metrics[pooled_metrics["Eligible_For_Ranking"]].copy()

    predictions_all.to_csv(
        OUTPUT_DIR / "predictions_walk_forward.csv", index=False, encoding="utf-8-sig"
    )
    metrics_by_fold.to_csv(
        OUTPUT_DIR / "metrics_by_fold.csv", index=False, encoding="utf-8-sig"
    )
    pooled_metrics.to_csv(
        OUTPUT_DIR / "metrics_pooled.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(coverage_rows).to_csv(
        OUTPUT_DIR / "fold_coverage.csv", index=False, encoding="utf-8-sig"
    )

    winner = eligible.iloc[0]
    report = f"""# 최종 5모델 expanding walk-forward 평가

- 5개 시간 구간을 순서대로 검증했으며 미래 데이터를 과거 학습에 사용하지 않았다.
- LSTM과 공정하게 비교하기 위해 각 fold의 학습 구간에 한 번도 없던 부품은 해당 fold의 모든 모델 평가에서 제외했다.
- 평가 대상 모델: {', '.join(MODEL_NAMES)}
- pooled MAE 기준 1위: **{winner['Model']}** ({winner['MAE']:.4f})
- pooled 테스트 표본: {len(predictions_all):,}건

단일 holdout 결과와 함께 보며, 약 50일의 짧은 수집 기간 때문에 장기 계절성 검증으로 해석하지 않는다.
"""
    (OUTPUT_DIR / "WALK_FORWARD_REPORT.md").write_text(report, encoding="utf-8")
    print(pooled_metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
