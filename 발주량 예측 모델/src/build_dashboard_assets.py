from __future__ import annotations

import json
import math
import os
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "tmp" / "matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_evaluate_final_models import (
    DATA_PATH,
    MODEL_NAMES,
    REFERENCE_MODEL,
    calculate_metrics,
)
from train_evaluate_purged_models import load_daily_data


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs" / "purged_evaluation" / "source_total"
WALK_DIR = FINAL_DIR
OUTPUT_DIR = ROOT / "outputs" / "dashboard_data"
CSV_DIR = OUTPUT_DIR / "csv"
FIRESTORE_DIR = OUTPUT_DIR / "firestore"
FIGURE_DIR = FINAL_DIR / "figures"
MODEL_COLUMNS = MODEL_NAMES + [REFERENCE_MODEL]


def slug(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return text or "unknown"


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        return stamp.isoformat().replace("+00:00", "Z")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(item) for item in value]
    if pd.isna(value):
        return None
    return value


def write_jsonl(frame: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in frame.to_dict(orient="records"):
            stream.write(json.dumps(json_safe(record), ensure_ascii=False) + "\n")


def metrics_by_part(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for part_number, group in predictions.groupby("part_number", sort=True):
        actual = group["actual"].to_numpy(dtype=float)
        for model in MODEL_COLUMNS:
            prediction = group[model].to_numpy(dtype=float)
            metrics = calculate_metrics(actual, prediction)
            rows.append(
                {
                    "part_number": part_number,
                    "Model": model,
                    "N": len(group),
                    "Actual_Total": float(actual.sum()),
                    "Actual_Mean": float(actual.mean()),
                    "Zero_Actual_Count": int((actual == 0).sum()),
                    "Over_Prediction_Count": int((prediction > actual).sum()),
                    "Exact_Count": int(np.isclose(prediction, actual).sum()),
                    "Under_Prediction_Count": int((prediction < actual).sum()),
                    **metrics,
                }
            )
    detail = pd.DataFrame(rows)
    macro = (
        detail.groupby("Model", as_index=False)
        .agg(
            Parts=("part_number", "nunique"),
            Per_Part_MAE_Mean=("MAE", "mean"),
            Per_Part_MAE_Median=("MAE", "median"),
            Per_Part_RMSE_Mean=("RMSE", "mean"),
            Per_Part_WAPE_Mean=("WAPE_pct", "mean"),
            Per_Part_Bias_Mean=("Forecast_Bias", "mean"),
            Per_Part_R2_Mean=("R2", "mean"),
        )
        .sort_values("Per_Part_MAE_Mean")
        .reset_index(drop=True)
    )
    return detail, macro


def spike_analysis(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = predictions.groupby("part_number")["actual"].quantile(0.90)
    events = predictions.copy()
    events["Part_Actual_P90"] = events["part_number"].map(thresholds)
    events = events[(events["actual"] >= events["Part_Actual_P90"]) & (events["actual"] > 0)].copy()
    for model in MODEL_COLUMNS:
        events[f"{model}_Absolute_Error"] = (events[model] - events["actual"]).abs()

    rows: list[dict[str, object]] = []
    for model in MODEL_COLUMNS:
        rows.append(
            {
                "Model": model,
                "Spike_Rows": len(events),
                "Spike_Parts": events["part_number"].nunique(),
                **calculate_metrics(
                    events["actual"].to_numpy(dtype=float),
                    events[model].to_numpy(dtype=float),
                ),
            }
        )
    return events, pd.DataFrame(rows).sort_values("MAE")


def source_quality_profile(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(DATA_PATH)
    profile_rows: list[dict[str, object]] = []
    for position, column in enumerate(raw.columns):
        series = raw[column]
        numeric = pd.to_numeric(series, errors="coerce")
        profile_rows.append(
            {
                "column_position": position,
                "column_name": str(column),
                "pandas_dtype": str(series.dtype),
                "row_count": len(series),
                "non_null_count": int(series.notna().sum()),
                "missing_count": int(series.isna().sum()),
                "unique_count": int(series.nunique(dropna=True)),
                "constant_column": int(series.nunique(dropna=True)) <= 1,
                "numeric_parse_count": int(numeric.notna().sum()),
                "numeric_min": float(numeric.min()) if numeric.notna().any() else np.nan,
                "numeric_max": float(numeric.max()) if numeric.notna().any() else np.nan,
            }
        )
    profile = pd.DataFrame(profile_rows)

    timestamps = pd.to_datetime(
        raw.iloc[:, 83].astype(str), format="%Y%m%d%H%M", errors="coerce"
    )
    unique_timestamps = pd.Series(sorted(timestamps.dropna().unique()))
    intervals = unique_timestamps.diff().dropna().dt.total_seconds().div(3600)
    interval_summary = pd.DataFrame(
        [
            {
                "timestamp_count": int(len(unique_timestamps)),
                "start": unique_timestamps.min(),
                "end": unique_timestamps.max(),
                "mean_hours": float(intervals.mean()),
                "median_hours": float(intervals.median()),
                "min_hours": float(intervals.min()),
                "max_hours": float(intervals.max()),
                "p90_hours": float(intervals.quantile(0.90)),
            }
        ]
    )

    abrupt_rows: list[pd.DataFrame] = []
    for part_number, group in daily.groupby("part_number", sort=True):
        current = group.sort_values("date").copy()
        current["previous_actual"] = current["actual_d"].shift(1)
        current["absolute_change"] = (current["actual_d"] - current["previous_actual"]).abs()
        valid = current["absolute_change"].dropna()
        if valid.empty:
            continue
        q1, q3 = valid.quantile([0.25, 0.75])
        threshold = max(float(q3 + 3 * (q3 - q1)), 1.0)
        flagged = current[current["absolute_change"] > threshold].copy()
        flagged["part_change_threshold"] = threshold
        abrupt_rows.append(
            flagged[
                [
                    "part_number",
                    "date",
                    "previous_actual",
                    "actual_d",
                    "absolute_change",
                    "part_change_threshold",
                ]
            ]
        )
    abrupt = (
        pd.concat(abrupt_rows, ignore_index=True)
        if abrupt_rows
        else pd.DataFrame(
            columns=[
                "part_number",
                "date",
                "previous_actual",
                "actual_d",
                "absolute_change",
                "part_change_threshold",
            ]
        )
    )
    return profile, interval_summary, abrupt


def make_parts(
    coverage: pd.DataFrame, predictions: pd.DataFrame, part_metrics: pd.DataFrame
) -> pd.DataFrame:
    test_summary = predictions.groupby("part_number").agg(
        test_rows=("target_date", "size"),
        test_start=("target_date", "min"),
        test_end=("target_date", "max"),
        actual_total=("actual", "sum"),
        actual_mean=("actual", "mean"),
        actual_zero_count=("actual", lambda values: int((values == 0).sum())),
    )
    choices = part_metrics[part_metrics["Model"].isin(["XGBoost", "3-day Moving Average"])]
    choices = choices.sort_values(["part_number", "MAE", "RMSE"]).drop_duplicates("part_number")
    choices = choices.set_index("part_number")[["Model", "MAE"]].rename(
        columns={"Model": "recommended_model", "MAE": "recommended_model_test_mae"}
    )
    parts = coverage.set_index("part_number").join(test_summary).join(choices).reset_index()
    parts["document_id"] = parts["part_number"].map(slug)
    parts["has_test_data"] = parts["test_rows"].fillna(0).astype(int) > 0
    return parts


def make_daily_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    aggregation = {"actual": "sum", **{model: "sum" for model in MODEL_COLUMNS}}
    daily = predictions.groupby("target_date", as_index=False).agg(aggregation)
    daily["document_id"] = pd.to_datetime(daily["target_date"]).dt.strftime("%Y_%m_%d")
    daily["part_count"] = predictions.groupby("target_date")["part_number"].nunique().to_numpy()
    return daily


def latest_consecutive_three(
    daily: pd.DataFrame, preferred_part: str = "Part 94"
) -> pd.DataFrame:
    part_order = [preferred_part] + [
        part for part in sorted(daily["part_number"].unique()) if part != preferred_part
    ]
    for part_number in part_order:
        group = (
            daily[daily["part_number"] == part_number]
            .sort_values("date")
            .reset_index(drop=True)
        )
        for end in range(len(group) - 1, 1, -1):
            window = group.iloc[end - 2 : end + 1]
            if np.all(window["date"].diff().dropna().dt.days.to_numpy() == 1):
                return window[REQUIRED_INFERENCE_COLUMNS()].copy()
    raise ValueError("No part contains a valid consecutive three-day inference example.")


def make_alerts(predictions: pd.DataFrame, parts: pd.DataFrame) -> pd.DataFrame:
    recommended = parts.set_index("part_number")["recommended_model"].to_dict()
    latest = predictions.sort_values("target_date").groupby("part_number", as_index=False).tail(1).copy()
    latest["recommended_model"] = latest["part_number"].map(recommended).fillna("3-day Moving Average")
    latest["recommended_forecast"] = [
        row[model] for (_, row), model in zip(latest.iterrows(), latest["recommended_model"])
    ]
    latest["forecast_gap_vs_plan"] = latest["recommended_forecast"] - latest[REFERENCE_MODEL]
    latest["forecast_gap_pct"] = np.where(
        latest[REFERENCE_MODEL] > 0,
        latest["forecast_gap_vs_plan"] / latest[REFERENCE_MODEL] * 100,
        np.nan,
    )
    threshold = np.maximum(10.0, latest[REFERENCE_MODEL] * 0.20)
    latest["alert_type"] = np.select(
        [
            latest["forecast_gap_vs_plan"] >= threshold,
            latest["forecast_gap_vs_plan"] <= -threshold,
        ],
        ["forecast_above_plan", "forecast_below_plan"],
        default="within_plan_band",
    )
    latest["document_id"] = latest.apply(
        lambda row: f"{slug(row['part_number'])}__{pd.Timestamp(row['target_date']).strftime('%Y_%m_%d')}",
        axis=1,
    )
    return latest[
        [
            "document_id",
            "part_number",
            "origin_date",
            "target_date",
            "actual",
            REFERENCE_MODEL,
            "recommended_model",
            "recommended_forecast",
            "forecast_gap_vs_plan",
            "forecast_gap_pct",
            "alert_type",
        ]
    ]


def save_figures(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    part_metrics: pd.DataFrame,
    walk_metrics: pd.DataFrame | None,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})

    ranked = metrics[metrics["Eligible_For_Ranking"]].sort_values("MAE")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].barh(ranked["Model"], ranked["MAE"], color="#2563EB")
    axes[0].invert_yaxis()
    axes[0].set_title("Final holdout MAE")
    axes[0].set_xlabel("MAE")
    axes[1].barh(ranked["Model"], ranked["RMSE"], color="#0F766E")
    axes[1].invert_yaxis()
    axes[1].set_title("Final holdout RMSE")
    axes[1].set_xlabel("RMSE")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    daily = make_daily_summary(predictions)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(daily["target_date"], daily["actual"], marker="o", linewidth=2.5, label="Actual")
    ax.plot(daily["target_date"], daily["XGBoost"], marker="o", label="XGBoost")
    ax.plot(
        daily["target_date"],
        daily["3-day Moving Average"],
        marker="o",
        label="3-day Moving Average",
    )
    ax.set_title("Daily total actual and forecast")
    ax.set_ylabel("Quantity")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "actual_vs_predicted_daily.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    comparison = part_metrics[part_metrics["Model"].isin(["XGBoost", "3-day Moving Average"])]
    pivot = comparison.pivot(index="part_number", columns="Model", values="MAE")
    volume = predictions.groupby("part_number")["actual"].sum().sort_values(ascending=False)
    shown = pivot.reindex(volume.head(25).index)
    ax = shown.plot(kind="bar", figsize=(15, 6), color=["#0F766E", "#F59E0B"])
    ax.set_title("Per-part MAE for 25 highest-volume parts")
    ax.set_ylabel("MAE")
    ax.tick_params(axis="x", rotation=65)
    ax.grid(axis="y", alpha=0.2)
    ax.figure.tight_layout()
    ax.figure.savefig(FIGURE_DIR / "per_part_mae.png", dpi=160, bbox_inches="tight")
    plt.close(ax.figure)

    if walk_metrics is not None and not walk_metrics.empty:
        plot_frame = walk_metrics[walk_metrics["Model"].isin(MODEL_NAMES)]
        pivot = plot_frame.pivot(index="Fold", columns="Model", values="MAE")
        ax = pivot.plot(marker="o", figsize=(13, 5))
        ax.set_title("Expanding walk-forward MAE by fold")
        ax.set_ylabel("MAE")
        ax.grid(alpha=0.2)
        ax.figure.tight_layout()
        ax.figure.savefig(FIGURE_DIR / "walk_forward_mae.png", dpi=160, bbox_inches="tight")
        plt.close(ax.figure)


def build_firestore_files(
    datasets: dict[str, pd.DataFrame], config: dict[str, object]
) -> pd.DataFrame:
    FIRESTORE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    for collection, frame in datasets.items():
        path = FIRESTORE_DIR / f"{collection}.jsonl"
        write_jsonl(frame, path)
        manifest_rows.append(
            {
                "collection": collection,
                "file": path.name,
                "document_count": len(frame),
                "document_id_field": "document_id",
            }
        )
    config_frame = pd.DataFrame([{"document_id": "current", **config}])
    write_jsonl(config_frame, FIRESTORE_DIR / "dashboard_config.jsonl")
    manifest_rows.append(
        {
            "collection": "dashboard_config",
            "file": "dashboard_config.jsonl",
            "document_count": 1,
            "document_id_field": "document_id",
        }
    )
    manifest = pd.DataFrame(manifest_rows)
    (FIRESTORE_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "format": "UTF-8 JSON Lines; one Firestore document per line",
                "collections": manifest.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    FIRESTORE_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(FINAL_DIR / "final_holdout_predictions.csv", parse_dates=["origin_date", "target_date"])
    metrics = pd.read_csv(FINAL_DIR / "final_holdout_metrics.csv")
    metrics["Eligible_For_Ranking"] = metrics["Eligible_For_Overall_Ranking"]
    coverage = pd.read_csv(FINAL_DIR / "part_sequence_coverage.csv", parse_dates=["first_date", "last_date"])
    run_metadata = json.loads((FINAL_DIR / "run_metadata.json").read_text(encoding="utf-8"))
    quality_rows = []
    for key, value in {**run_metadata["quality"], **run_metadata["sequence"]}.items():
        if isinstance(value, dict):
            quality_rows.extend({"Metric": f"{key}.{nested}", "Value": item} for nested, item in value.items())
        else:
            quality_rows.append({"Metric": key, "Value": value})
    quality = pd.DataFrame(quality_rows)
    daily, _ = load_daily_data("source_total")

    part_metrics, macro_metrics = metrics_by_part(predictions)
    spike_events, spike_metrics = spike_analysis(predictions)
    parts = make_parts(coverage, predictions, part_metrics)
    daily_summary = make_daily_summary(predictions)
    alerts = make_alerts(predictions, parts)
    column_profile, interval_summary, abrupt_changes = source_quality_profile(daily)

    reference_mae = float(metrics.loc[metrics["Model"] == REFERENCE_MODEL, "MAE"].iloc[0])
    improvement = metrics.copy()
    improvement["MAE_Improvement_vs_D3_Plan_pct"] = (
        (reference_mae - improvement["MAE"]) / reference_mae * 100
    )

    walk_fold = None
    walk_pooled = None
    if (WALK_DIR / "cv_metrics_by_fold.csv").exists():
        walk_fold = pd.read_csv(WALK_DIR / "cv_metrics_by_fold.csv")
        walk_pooled = pd.read_csv(WALK_DIR / "cv_metrics_pooled.csv")

    flat_predictions = predictions.copy()
    flat_predictions.insert(
        0,
        "document_id",
        flat_predictions.apply(
            lambda row: f"{slug(row['part_number'])}__{pd.Timestamp(row['target_date']).strftime('%Y_%m_%d')}",
            axis=1,
        ),
    )
    daily_history = daily[
        ["part_number", "date", "actual_d", "plan_d1", "plan_d2", "plan_d3", "plan_d4", "plan_d5"]
    ].copy()
    daily_history.insert(
        0,
        "document_id",
        daily_history.apply(
            lambda row: f"{slug(row['part_number'])}__{pd.Timestamp(row['date']).strftime('%Y_%m_%d')}",
            axis=1,
        ),
    )

    part_metrics.insert(
        0,
        "document_id",
        part_metrics.apply(lambda row: f"{slug(row['part_number'])}__{slug(row['Model'])}", axis=1),
    )
    model_metrics = improvement.copy()
    model_metrics.insert(0, "document_id", model_metrics["Model"].map(slug))
    macro_metrics.insert(0, "document_id", macro_metrics["Model"].map(slug))
    quality_docs = quality.rename(columns={"Metric": "metric", "Value": "value"}).copy()
    quality_docs.insert(0, "document_id", quality_docs["metric"].map(slug))
    spike_events.insert(
        0,
        "document_id",
        spike_events.apply(
            lambda row: f"{slug(row['part_number'])}__{pd.Timestamp(row['target_date']).strftime('%Y_%m_%d')}",
            axis=1,
        ),
    )
    daily_summary.insert(0, "collection_version", "v1")

    csv_datasets: dict[str, pd.DataFrame] = {
        "model_metrics": model_metrics,
        "model_metrics_per_part": part_metrics,
        "model_metrics_per_part_macro": macro_metrics,
        "model_improvement_vs_baseline": improvement,
        "predictions_test": flat_predictions,
        "daily_summary": daily_summary,
        "parts": parts,
        "alerts": alerts,
        "daily_history": daily_history,
        "spike_events": spike_events,
        "spike_metrics": spike_metrics,
        "data_quality": quality,
        "source_column_profile": column_profile,
        "time_interval_summary": interval_summary,
        "abrupt_change_flags": abrupt_changes,
    }
    if walk_fold is not None and walk_pooled is not None:
        csv_datasets["walk_forward_metrics_by_fold"] = walk_fold
        csv_datasets["walk_forward_metrics_pooled"] = walk_pooled

    for name, frame in csv_datasets.items():
        frame.to_csv(CSV_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    walk_docs = pd.DataFrame()
    if walk_fold is not None:
        walk_docs = walk_fold.copy()
        walk_docs.insert(
            0,
            "document_id",
            walk_docs.apply(lambda row: f"fold_{int(row['Fold'])}__{slug(row['Model'])}", axis=1),
        )
    firestore_sets = {
        "parts": parts,
        "daily_history": daily_history,
        "forecasts": flat_predictions,
        "model_metrics": model_metrics,
        "part_metrics": part_metrics,
        "daily_summary": daily_summary,
        "alerts": alerts,
        "data_quality": quality_docs,
    }
    if not walk_docs.empty:
        firestore_sets["walk_forward_metrics"] = walk_docs

    config = {
        "primary_method": "3-day Moving Average",
        "primary_ml_model": "XGBoost",
        "fallback_model": "3-day Moving Average",
        "overall_holdout_winner": metrics.sort_values("MAE").iloc[0]["Model"],
        "forecast_horizon_days": 3,
        "sequence_length_days": 3,
        "test_start": predictions["target_date"].min(),
        "test_end": predictions["target_date"].max(),
        "model_version": "purged_v1_source_total",
        "selection_basis": "purged_3_fold_walk_forward_MAE",
        "data_limit": "Approximately 50 days; no annual seasonality conclusion",
    }
    manifest = build_firestore_files(firestore_sets, config)
    manifest.to_csv(OUTPUT_DIR / "firestore_manifest.csv", index=False, encoding="utf-8-sig")

    sample = latest_consecutive_three(daily)
    sample.to_csv(OUTPUT_DIR / "inference_input_template.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "inference_request_example.json").write_text(
        json.dumps({"records": json_safe(sample.to_dict(orient="records"))}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    data_dictionary_rows: list[dict[str, object]] = []
    for dataset_name, frame in csv_datasets.items():
        for column in frame.columns:
            data_dictionary_rows.append(
                {
                    "dataset": dataset_name,
                    "column": column,
                    "dtype": str(frame[column].dtype),
                    "nullable": bool(frame[column].isna().any()),
                    "row_count": len(frame),
                }
            )
    pd.DataFrame(data_dictionary_rows).to_csv(
        OUTPUT_DIR / "data_dictionary.csv", index=False, encoding="utf-8-sig"
    )

    save_figures(metrics, predictions, part_metrics, walk_fold)

    holdout_winner = metrics[metrics["Eligible_For_Ranking"]].sort_values("MAE").iloc[0]
    xgboost = walk_pooled[walk_pooled["Model"] == "XGBoost"].iloc[0]
    walk_text = "Walk-forward result was not available."
    if walk_pooled is not None:
        walk_winner = walk_pooled[walk_pooled["Model"].isin(MODEL_NAMES)].sort_values("MAE").iloc[0]
        walk_text = f"Pooled walk-forward MAE winner: **{walk_winner['Model']}** ({walk_winner['MAE']:.4f})."
    report = f"""# 발주량 예측 최종 분석 및 대시보드 데이터 준비 보고서

## 모델 결론

- 단일 최종 holdout 1위: **{holdout_winner['Model']}**, MAE {holdout_winner['MAE']:.4f}
- 누수 제거 반복검증 전체 1위: **3-day Moving Average**
- 누수 제거 반복검증 학습형 모델 1위: **XGBoost**, MAE {xgboost['MAE']:.4f}
- XGBoost의 독립 holdout D+3 계획값 대비 MAE 개선율: {float(improvement.loc[improvement['Model'] == 'XGBoost', 'MAE_Improvement_vs_D3_Plan_pct'].iloc[0]):.2f}%
- {walk_text}
- 운영안: 3일 이동평균을 기본 예측으로 사용하고 XGBoost를 학습형 보조 예측으로 함께 제공한다.

## 분석 범위

- 최종 공통 테스트: {len(predictions):,}건, {predictions['part_number'].nunique()}개 부품
- 부품별 성능, 과대·과소예측 횟수, 수요 급증 구간 성능을 별도 파일로 저장했다.
- 실제값·예측값, 모델 비교, 부품별 MAE, walk-forward 결과를 PNG로 저장했다.
- Firestore에는 아직 업로드하지 않았으며 JSONL 파일과 매니페스트만 생성했다.

## 제한사항

- 원본 수집 기간이 약 50일이므로 월·분기·연간 계절성은 검증할 수 없다.
- 예측값은 발주 수량 의사결정을 보조하며 재고, 단가, 리드타임 데이터가 없으므로 비용 최적화 결과는 아니다.
- 테스트 기간에 나타나지 않은 부품은 최종 holdout 지표에 포함되지 않는다.
"""
    (FINAL_DIR / "DASHBOARD_ANALYSIS_REPORT.md").write_text(report, encoding="utf-8")
    print(f"Dashboard assets written to {OUTPUT_DIR}")


def REQUIRED_INFERENCE_COLUMNS() -> list[str]:
    return ["part_number", "date", "actual_d", "plan_d3", "plan_d4", "plan_d5"]


if __name__ == "__main__":
    main()
