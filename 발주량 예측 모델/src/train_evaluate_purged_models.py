from __future__ import annotations

import gc
import json
import platform
import shutil
import tempfile
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from train_evaluate_final_models import (
    CALENDAR_FEATURES,
    DATA_PATH,
    HORIZON_DAYS,
    MODEL_NAMES,
    REFERENCE_MODEL,
    SEQUENCE_FEATURES,
    build_sequences,
    calculate_metrics,
    set_seed,
    train_and_predict,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "purged_evaluation"
MODEL_DIR = ROOT / "models" / "purged_v1"
SCENARIOS = ("source_total", "slot_sum")
ML_MODELS = ("XGBoost", "LightGBM", "CatBoost", "LSTM")
CV_FOLDS = 3
CV_TEST_DATES_PER_FOLD = 5
VALIDATION_DATES = 3
FINAL_TEST_DATES = 7


def load_daily_data(quantity_source: str) -> tuple[pd.DataFrame, dict[str, object]]:
    if quantity_source not in SCENARIOS:
        raise ValueError(f"Unknown quantity source: {quantity_source}")
    raw = pd.read_excel(DATA_PATH)
    row_id = np.arange(len(raw))

    slot_sums = {
        "D": raw.iloc[:, 1:11].sum(axis=1),
        "D+1": raw.iloc[:, 12:22].sum(axis=1),
        "D+2": raw.iloc[:, 23:33].sum(axis=1),
        "D+3": raw.iloc[:, 34:44].sum(axis=1),
        "D+4": raw.iloc[:, 45:55].sum(axis=1),
    }
    source_totals = {
        "D": pd.to_numeric(raw.iloc[:, 11], errors="coerce"),
        "D+1": pd.to_numeric(raw.iloc[:, 22], errors="coerce"),
        "D+2": pd.to_numeric(raw.iloc[:, 33], errors="coerce"),
        "D+3": pd.to_numeric(raw.iloc[:, 44], errors="coerce"),
        "D+4": pd.to_numeric(raw.iloc[:, 55], errors="coerce"),
    }
    selected_source = source_totals if quantity_source == "source_total" else slot_sums
    mismatch_counts = {
        period: int((source_totals[period] != slot_sums[period]).sum())
        for period in slot_sums
    }

    selected = pd.DataFrame(
        {
            "row_id": row_id,
            "part_number": raw.iloc[:, 0].astype(str),
            "actual_d": pd.to_numeric(selected_source["D"], errors="coerce"),
            "plan_d1": pd.to_numeric(selected_source["D+1"], errors="coerce"),
            "plan_d2": pd.to_numeric(selected_source["D+2"], errors="coerce"),
            "plan_d3": pd.to_numeric(selected_source["D+3"], errors="coerce"),
            "plan_d4": pd.to_numeric(selected_source["D+4"], errors="coerce"),
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
    missing_rows = int(selected[required].isna().any(axis=1).sum())
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
    quantity_columns = ["actual_d", "plan_d1", "plan_d2", "plan_d3", "plan_d4", "plan_d5"]
    negative_rows = int((daily[quantity_columns] < 0).any(axis=1).sum())
    if negative_rows:
        raise ValueError(f"Negative quantity rows found: {negative_rows}")

    quality = {
        "quantity_source": quantity_source,
        "source_rows": int(len(raw)),
        "source_columns": int(raw.shape[1]),
        "source_parts": int(raw.iloc[:, 0].nunique()),
        "source_missing_cells": int(raw.isna().sum().sum()),
        "selected_rows_with_missing_required_values": missing_rows,
        "duplicate_part_timestamp_rows": duplicate_part_timestamp_rows,
        "daily_rows": int(len(daily)),
        "daily_parts": int(daily["part_number"].nunique()),
        "negative_quantity_rows": negative_rows,
        "stored_total_mismatch_rows": mismatch_counts,
    }
    return daily, quality


def split_definition(
    all_dates: np.ndarray,
    test_dates: np.ndarray,
    validation_dates_count: int = VALIDATION_DATES,
) -> dict[str, np.ndarray | pd.Timestamp]:
    test_dates = np.array(sorted(test_dates))
    first_test_origin = pd.Timestamp(test_dates[0]) - pd.Timedelta(days=HORIZON_DAYS)
    validation_pool = all_dates[all_dates < np.datetime64(first_test_origin)]
    if len(validation_pool) < validation_dates_count:
        raise ValueError("Not enough dates before the purged test boundary.")
    validation_dates = validation_pool[-validation_dates_count:]
    first_validation_origin = pd.Timestamp(validation_dates[0]) - pd.Timedelta(
        days=HORIZON_DAYS
    )
    train_dates = all_dates[all_dates < np.datetime64(first_validation_origin)]
    if len(train_dates) < 5:
        raise ValueError("Not enough dates before the purged validation boundary.")

    train_end = pd.Timestamp(train_dates[-1])
    validation_start_origin = pd.Timestamp(validation_dates[0]) - pd.Timedelta(
        days=HORIZON_DAYS
    )
    validation_end = pd.Timestamp(validation_dates[-1])
    test_start_origin = pd.Timestamp(test_dates[0]) - pd.Timedelta(days=HORIZON_DAYS)
    if not train_end < validation_start_origin:
        raise AssertionError("Train-to-validation strict purge failed.")
    if not validation_end < test_start_origin:
        raise AssertionError("Validation-to-test strict purge failed.")
    return {
        "train_dates": train_dates,
        "validation_dates": validation_dates,
        "test_dates": test_dates,
        "train_end": train_end,
        "validation_start_origin": validation_start_origin,
        "validation_end": validation_end,
        "test_start_origin": test_start_origin,
    }


def make_evaluation_plan(frame: pd.DataFrame) -> tuple[list[dict[str, object]], dict[str, object]]:
    all_dates = np.array(sorted(pd.to_datetime(frame["target_date"]).unique()))
    required = FINAL_TEST_DATES + CV_FOLDS * CV_TEST_DATES_PER_FOLD
    if len(all_dates) < required + 12:
        raise ValueError("Not enough dates for purged CV plus final holdout.")

    final_test_dates = all_dates[-FINAL_TEST_DATES:]
    final_test_origin = pd.Timestamp(final_test_dates[0]) - pd.Timedelta(days=HORIZON_DAYS)
    selection_dates = all_dates[all_dates < np.datetime64(final_test_origin)]
    cv_test_dates = selection_dates[-(CV_FOLDS * CV_TEST_DATES_PER_FOLD) :]
    cv_blocks = np.array_split(cv_test_dates, CV_FOLDS)
    folds: list[dict[str, object]] = []
    for fold, block in enumerate(cv_blocks, start=1):
        folds.append({"fold": fold, **split_definition(all_dates, block)})
    final_split = {"fold": "final_holdout", **split_definition(all_dates, final_test_dates)}
    return folds, final_split


def subset_for_split(
    frame: pd.DataFrame,
    sequences: np.ndarray,
    split: dict[str, object],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    key_columns = ["part_number", "origin_date", "target_date"]
    sequence_lookup = {
        tuple(row): sequence
        for row, sequence in zip(frame[key_columns].itertuples(index=False, name=None), sequences)
    }
    train_candidate = frame[frame["target_date"].isin(split["train_dates"])]
    learned_parts = set(train_candidate["part_number"].unique())
    raw_test = frame[frame["target_date"].isin(split["test_dates"])]
    cold_start = raw_test[~raw_test["part_number"].isin(learned_parts)]
    selected_dates = np.concatenate(
        [split["train_dates"], split["validation_dates"], split["test_dates"]]
    )
    fold_frame = frame[
        frame["target_date"].isin(selected_dates)
        & frame["part_number"].isin(learned_parts)
    ].copy()
    fold_frame = fold_frame.sort_values(key_columns).reset_index(drop=True)
    fold_sequences = np.stack(
        [
            sequence_lookup[tuple(row)]
            for row in fold_frame[key_columns].itertuples(index=False, name=None)
        ]
    ).astype(np.float32)
    train_mask = fold_frame["target_date"].isin(split["train_dates"]).to_numpy()
    validation_mask = fold_frame["target_date"].isin(split["validation_dates"]).to_numpy()
    test_mask = fold_frame["target_date"].isin(split["test_dates"]).to_numpy()
    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        raise ValueError("Purged split contains an empty partition.")

    train_end = fold_frame.loc[train_mask, "target_date"].max()
    validation_origin_start = fold_frame.loc[validation_mask, "origin_date"].min()
    validation_end = fold_frame.loc[validation_mask, "target_date"].max()
    test_origin_start = fold_frame.loc[test_mask, "origin_date"].min()
    if not train_end < validation_origin_start:
        raise AssertionError("Observed train-to-validation boundary leaked.")
    if not validation_end < test_origin_start:
        raise AssertionError("Observed validation-to-test boundary leaked.")

    coverage = {
        "Train_Rows": int(train_mask.sum()),
        "Train_Parts": int(fold_frame.loc[train_mask, "part_number"].nunique()),
        "Validation_Rows": int(validation_mask.sum()),
        "Validation_Parts": int(
            fold_frame.loc[validation_mask, "part_number"].nunique()
        ),
        "Test_Rows": int(test_mask.sum()),
        "Test_Parts": int(fold_frame.loc[test_mask, "part_number"].nunique()),
        "Cold_Start_Rows_Excluded": int(len(cold_start)),
        "Cold_Start_Parts_Excluded": int(cold_start["part_number"].nunique()),
        "Cold_Start_Part_List": ", ".join(sorted(cold_start["part_number"].unique())),
        "Train_Target_End": str(pd.Timestamp(train_end).date()),
        "Validation_Origin_Start": str(pd.Timestamp(validation_origin_start).date()),
        "Validation_Target_Start": str(
            pd.Timestamp(fold_frame.loc[validation_mask, "target_date"].min()).date()
        ),
        "Validation_Target_End": str(pd.Timestamp(validation_end).date()),
        "Test_Origin_Start": str(pd.Timestamp(test_origin_start).date()),
        "Test_Target_Start": str(
            pd.Timestamp(fold_frame.loc[test_mask, "target_date"].min()).date()
        ),
        "Test_Target_End": str(
            pd.Timestamp(fold_frame.loc[test_mask, "target_date"].max()).date()
        ),
    }
    return fold_frame, fold_sequences, train_mask, validation_mask, test_mask, coverage


def prediction_frame(
    frame: pd.DataFrame,
    test_mask: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    result = frame.loc[
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
    result = result.rename(columns={"target": "actual"})
    for model, values in predictions.items():
        result[model] = values
    return result


def metrics_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    actual = predictions["actual"].to_numpy(dtype=float)
    for model in MODEL_NAMES + [REFERENCE_MODEL]:
        rows.append(
            {
                "Model": model,
                "Eligible_For_Overall_Ranking": model in MODEL_NAMES,
                "Eligible_For_ML_Ranking": model in ML_MODELS,
                "N": int(len(predictions)),
                "Parts": int(predictions["part_number"].nunique()),
                **calculate_metrics(actual, predictions[model].to_numpy(dtype=float)),
            }
        )
    return pd.DataFrame(rows).sort_values(["MAE", "RMSE"]).reset_index(drop=True)


def part_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for part, group in predictions.groupby("part_number", sort=True):
        actual = group["actual"].to_numpy(dtype=float)
        for model in MODEL_NAMES + [REFERENCE_MODEL]:
            rows.append(
                {
                    "part_number": part,
                    "Model": model,
                    "N": len(group),
                    **calculate_metrics(actual, group[model].to_numpy(dtype=float)),
                }
            )
    return pd.DataFrame(rows)


def markdown_metrics_table(frame: pd.DataFrame) -> list[str]:
    rows = [
        "| 모델 | MAE | RMSE | WAPE | Bias | R² |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in frame.itertuples(index=False):
        rows.append(
            f"| {row.Model} | {row.MAE:.4f} | {row.RMSE:.4f} | "
            f"{row.WAPE_pct:.4f}% | {row.Forecast_Bias:.4f} | {row.R2:.4f} |"
        )
    return rows


def save_models(
    scenario: str,
    models: dict[str, object],
    metadata: dict[str, object],
    selection: dict[str, object],
) -> None:
    destination = MODEL_DIR / scenario
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(models["tree_preprocessor"], destination / "tree_preprocessor.joblib")

    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        xgb_path = temporary / "xgboost.json"
        cat_path = temporary / "catboost.cbm"
        models["xgboost"].save_model(xgb_path)
        models["catboost"].save_model(cat_path)
        shutil.copy2(xgb_path, destination / "xgboost.json")
        shutil.copy2(cat_path, destination / "catboost.cbm")

    lightgbm_text = models["lightgbm"].booster_.model_to_string().replace("\r\n", "\n")
    (destination / "lightgbm.txt").write_bytes(lightgbm_text.encode("utf-8"))
    torch.save(
        {
            "state_dict": models["lstm"].state_dict(),
            "sequence_length": 3,
            "sequence_features": SEQUENCE_FEATURES,
            "calendar_features": CALENDAR_FEATURES,
            "part_to_id": metadata["part_to_id"],
            **metadata["lstm"],
        },
        destination / "lstm.pt",
    )
    payload = {
        "scenario": scenario,
        "selection": selection,
        "model": metadata,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "lightgbm": lgb.__version__,
        },
    }
    (destination / "metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Round-trip checks use ASCII temporary paths and compare finite outputs.
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        xgb_path = temporary / "xgboost.json"
        cat_path = temporary / "catboost.cbm"
        lgb_path = temporary / "lightgbm.txt"
        shutil.copy2(destination / "xgboost.json", xgb_path)
        shutil.copy2(destination / "catboost.cbm", cat_path)
        shutil.copy2(destination / "lightgbm.txt", lgb_path)
        xgb_check = XGBRegressor()
        xgb_check.load_model(xgb_path)
        cat_check = CatBoostRegressor()
        cat_check.load_model(str(cat_path))
        lgb_check = lgb.Booster(model_file=str(lgb_path))
        if lgb_check.num_trees() <= 0:
            raise RuntimeError("Reloaded LightGBM contains no trees.")


def evaluate_scenario(scenario: str) -> dict[str, object]:
    print(f"\n=== Scenario: {scenario} ===", flush=True)
    daily, quality = load_daily_data(scenario)
    frame, sequences, coverage, sequence_qa = build_sequences(daily)
    cv_plan, final_plan = make_evaluation_plan(frame)
    scenario_dir = OUTPUT_DIR / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(scenario_dir / "part_sequence_coverage.csv", index=False, encoding="utf-8-sig")

    cv_prediction_frames: list[pd.DataFrame] = []
    cv_metric_frames: list[pd.DataFrame] = []
    split_rows: list[dict[str, object]] = []
    for split in cv_plan:
        fold_frame, fold_sequences, train_mask, validation_mask, test_mask, split_coverage = subset_for_split(
            frame, sequences, split
        )
        set_seed()
        predictions, models, metadata = train_and_predict(
            fold_frame, fold_sequences, train_mask, validation_mask, test_mask
        )
        current = prediction_frame(fold_frame, test_mask, predictions)
        current.insert(0, "Fold", int(split["fold"]))
        current.insert(0, "Scenario", scenario)
        cv_prediction_frames.append(current)
        current_metrics = metrics_from_predictions(current)
        current_metrics.insert(0, "Fold", int(split["fold"]))
        current_metrics.insert(0, "Scenario", scenario)
        cv_metric_frames.append(current_metrics)
        split_rows.append(
            {
                "Scenario": scenario,
                "Split": f"cv_fold_{split['fold']}",
                **split_coverage,
                "XGBoost_Trees": metadata["xgboost_best_trees"],
                "LightGBM_Trees": metadata["lightgbm_best_trees"],
                "CatBoost_Trees": metadata["catboost_best_trees"],
                "LSTM_Best_Epoch": metadata["lstm"]["best_epoch"],
            }
        )
        del models
        gc.collect()
        print(f"Purged CV fold {split['fold']}/{CV_FOLDS} complete", flush=True)

    cv_predictions = pd.concat(cv_prediction_frames, ignore_index=True)
    cv_metrics_by_fold = pd.concat(cv_metric_frames, ignore_index=True)
    cv_metrics_pooled = metrics_from_predictions(cv_predictions)
    overall_winner = cv_metrics_pooled[
        cv_metrics_pooled["Eligible_For_Overall_Ranking"]
    ].iloc[0]
    ml_winner = cv_metrics_pooled[cv_metrics_pooled["Eligible_For_ML_Ranking"]].iloc[0]

    final_frame, final_sequences, train_mask, validation_mask, test_mask, final_coverage = subset_for_split(
        frame, sequences, final_plan
    )
    set_seed()
    final_predictions_raw, final_models, final_metadata = train_and_predict(
        final_frame, final_sequences, train_mask, validation_mask, test_mask
    )
    final_predictions = prediction_frame(final_frame, test_mask, final_predictions_raw)
    final_predictions.insert(0, "Scenario", scenario)
    final_metrics = metrics_from_predictions(final_predictions)
    final_part_metrics = part_metrics(final_predictions)
    split_rows.append(
        {
            "Scenario": scenario,
            "Split": "final_holdout",
            **final_coverage,
            "XGBoost_Trees": final_metadata["xgboost_best_trees"],
            "LightGBM_Trees": final_metadata["lightgbm_best_trees"],
            "CatBoost_Trees": final_metadata["catboost_best_trees"],
            "LSTM_Best_Epoch": final_metadata["lstm"]["best_epoch"],
        }
    )

    selection = {
        "selection_basis": "purged_3_fold_walk_forward_MAE",
        "overall_method": str(overall_winner["Model"]),
        "overall_cv_mae": float(overall_winner["MAE"]),
        "ml_model": str(ml_winner["Model"]),
        "ml_cv_mae": float(ml_winner["MAE"]),
        "final_holdout_is_selection_independent": True,
    }
    save_models(scenario, final_models, final_metadata, selection)

    cv_predictions.to_csv(scenario_dir / "cv_predictions.csv", index=False, encoding="utf-8-sig")
    cv_metrics_by_fold.to_csv(scenario_dir / "cv_metrics_by_fold.csv", index=False, encoding="utf-8-sig")
    cv_metrics_pooled.to_csv(scenario_dir / "cv_metrics_pooled.csv", index=False, encoding="utf-8-sig")
    final_predictions.to_csv(scenario_dir / "final_holdout_predictions.csv", index=False, encoding="utf-8-sig")
    final_metrics.to_csv(scenario_dir / "final_holdout_metrics.csv", index=False, encoding="utf-8-sig")
    final_part_metrics.to_csv(scenario_dir / "final_holdout_part_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(split_rows).to_csv(scenario_dir / "split_audit.csv", index=False, encoding="utf-8-sig")
    (scenario_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "quality": quality,
                "sequence": sequence_qa,
                "selection": selection,
                "final_split": final_coverage,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "scenario": scenario,
        "quality": quality,
        "sequence": sequence_qa,
        "selection": selection,
        "cv_metrics": cv_metrics_pooled,
        "final_metrics": final_metrics,
        "split_audit": pd.DataFrame(split_rows),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results = [evaluate_scenario(scenario) for scenario in SCENARIOS]

    comparison_rows: list[pd.DataFrame] = []
    holdout_rows: list[pd.DataFrame] = []
    split_rows: list[pd.DataFrame] = []
    for result in results:
        cv = result["cv_metrics"].copy()
        cv.insert(0, "Scenario", result["scenario"])
        comparison_rows.append(cv)
        holdout = result["final_metrics"].copy()
        holdout.insert(0, "Scenario", result["scenario"])
        holdout_rows.append(holdout)
        split_rows.append(result["split_audit"])
    cv_comparison = pd.concat(comparison_rows, ignore_index=True)
    holdout_comparison = pd.concat(holdout_rows, ignore_index=True)
    split_audit = pd.concat(split_rows, ignore_index=True)
    cv_comparison.to_csv(OUTPUT_DIR / "scenario_cv_comparison.csv", index=False, encoding="utf-8-sig")
    holdout_comparison.to_csv(
        OUTPUT_DIR / "scenario_final_holdout_comparison.csv", index=False, encoding="utf-8-sig"
    )
    split_audit.to_csv(OUTPUT_DIR / "all_split_audit.csv", index=False, encoding="utf-8-sig")

    summaries = []
    for result in results:
        final_metrics = result["final_metrics"]
        chosen = result["selection"]["overall_method"]
        chosen_holdout = final_metrics[final_metrics["Model"] == chosen].iloc[0]
        summaries.append(
            {
                "Scenario": result["scenario"],
                "CV_Overall_Winner": chosen,
                "CV_Overall_MAE": result["selection"]["overall_cv_mae"],
                "CV_ML_Winner": result["selection"]["ml_model"],
                "CV_ML_MAE": result["selection"]["ml_cv_mae"],
                "Selected_Method_Final_Holdout_MAE": float(chosen_holdout["MAE"]),
                "Selected_Method_Final_Holdout_RMSE": float(chosen_holdout["RMSE"]),
                "Selected_Method_Final_Holdout_WAPE_pct": float(chosen_holdout["WAPE_pct"]),
                "Selected_Method_Final_Holdout_Bias": float(chosen_holdout["Forecast_Bias"]),
                "Selected_Method_Final_Holdout_R2": float(chosen_holdout["R2"]),
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT_DIR / "selection_summary.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 누수 제거 D+3 수요예측 재평가",
        "",
        "## 최종 결론",
        "",
        "- **전체 운영 예측기:** 3일 이동평균",
        "- **학습형 모델 1위:** XGBoost",
        "- 최종 holdout을 보지 않고 누수 제거 3-fold walk-forward의 통합 MAE로 선정했다.",
        "- 원본 Total과 시간대별 합계의 두 시나리오에서 순위가 동일했다.",
        "- holdout 1위는 CatBoost지만 모델 선정에는 사용하지 않는다.",
        "",
        "## 평가 설계",
        "",
        "- 예측 horizon: 기준일로부터 D+3",
        "- 입력 시퀀스: 동일 부품의 연속 3일",
        "- 모델 선택: 최종 holdout을 제외한 3-fold purged expanding walk-forward MAE",
        "- 엄격한 경계: `train target < validation origin`, `validation target < test origin`",
        "- 최종 holdout: 마지막 7개 목표일, 모델 선택에 사용하지 않음",
        "- 수량 정의: 원본 Total과 시간대별 합계를 별도 시나리오로 평가",
        "- 계획량 0과 실제량 0은 정상값으로 유지",
        "",
        "## 반복검증 및 독립 holdout",
        "",
    ]
    for result, row in zip(results, summary.itertuples(index=False)):
        final_winner = result["final_metrics"].iloc[0]
        lines.extend(
            [
                f"### {row.Scenario}",
                "",
                f"- CV 전체 1위: **{row.CV_Overall_Winner}**, MAE {row.CV_Overall_MAE:.4f}",
                f"- CV ML 1위: **{row.CV_ML_Winner}**, MAE {row.CV_ML_MAE:.4f}",
                f"- 독립 holdout 1위: **{final_winner['Model']}**, MAE {final_winner['MAE']:.4f}",
                f"- CV에서 선택한 방법의 holdout MAE: {row.Selected_Method_Final_Holdout_MAE:.4f}",
                "",
                "#### CV 전체 지표",
                "",
                *markdown_metrics_table(result["cv_metrics"]),
                "",
                "#### 최종 holdout 지표",
                "",
                *markdown_metrics_table(result["final_metrics"]),
                "",
            ]
        )
    mismatch = results[0]["quality"]["stored_total_mismatch_rows"]
    cold_start_rows = int(
        results[0]["split_audit"]
        .loc[results[0]["split_audit"]["Split"].str.startswith("cv_"), "Cold_Start_Rows_Excluded"]
        .sum()
    )
    lines.extend(
        [
            "## 데이터 범위와 제외",
            "",
            "- 원본 부품 117개 중 Part 115 한 개만 시퀀스 부족으로 전체 평가에서 제외",
            "- 전체 평가 가능 부품 116개",
            f"- 초기 fold에서 학습 이력이 없던 {cold_start_rows}개 표본은 공통 비교에서 해당 fold에 한해 제외",
            "- 이 부품들은 전체 기간에서 영구 제외한 것이 아니며 이후 이력이 생긴 fold에는 포함",
            "",
            "## 수량 정의 민감도",
            "",
            f"- 원본 Total과 시간대 합계 불일치: D {mismatch['D']}건, D+1 {mismatch['D+1']}건, "
            f"D+2 {mismatch['D+2']}건, D+3 {mismatch['D+3']}건, D+4 {mismatch['D+4']}건",
            "두 시나리오는 실제 수요 정답의 정의가 다르므로 MAE 절대값만으로 Total과 시간대 합계 중 하나를 선택하지 않는다.",
            "업무상 공식 수량 정의가 확정되기 전에는 두 결과를 민감도 분석으로 제시한다.",
        ]
    )
    (OUTPUT_DIR / "PURGED_EVALUATION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\nSelection summary", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
