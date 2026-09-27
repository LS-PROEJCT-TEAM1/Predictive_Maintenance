from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED_DIR = ROOT / "firestore" / "seed"
DEFAULT_WORKSPACE_ID = "manufacturing-ai"
MAX_DISPLAY_POINTS = 300
MEASUREMENT_CHUNK_SIZE = 100
SCHEMA_VERSION = 3
DATA_VERSION = "2026-09-27.v3"
SOURCE_ARTIFACTS: dict[str, str] = {}


def source_csv(path: Path) -> pd.DataFrame:
    SOURCE_ARTIFACTS[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)


def clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def records(frame: pd.DataFrame, drop: Iterable[str] = ()) -> list[dict[str, Any]]:
    keep = frame.drop(columns=[c for c in drop if c in frame.columns])
    return [clean(row) for row in keep.to_dict("records")]


def slug(value: str) -> str:
    source = str(value).strip()
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", source).strip("_").lower() or "item"
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
    return f"{normalized}-{digest}"


def document(path: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"path": path, "data": clean(data)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    SOURCE_ARTIFACTS[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_demand(base: str) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    data_dir = ROOT / "발주량 예측 모델" / "outputs" / "dashboard_data" / "csv"
    metadata_path = ROOT / '발주량 예측 모델/outputs/audited_v2/metadata.json'
    SOURCE_ARTIFACTS[metadata_path.relative_to(ROOT).as_posix()] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    policy, auxiliary = metadata['selection']['overall_method'], metadata['selection']['ml_model']
    parts = source_csv(data_dir / "parts.csv")
    history = source_csv(data_dir / "daily_history.csv")
    forecasts = source_csv(data_dir / "predictions_test.csv")
    part_metrics = source_csv(data_dir / "model_metrics_per_part.csv")
    model_metrics = source_csv(data_dir / "model_metrics.csv")
    walk_forward = source_csv(data_dir / "walk_forward_metrics_pooled.csv")
    data_quality = source_csv(data_dir / "data_quality.csv")
    # One validation-selected policy for overview, part details and action queues.
    alerts = forecasts.sort_values("target_date").groupby("part_number", as_index=False).tail(1).copy()
    alerts["recommended_model"] = policy
    alerts["recommended_forecast"] = alerts[policy]
    alerts["forecast_gap_vs_plan"] = alerts["recommended_forecast"] - alerts["D+3 Plan Reference"]
    alerts["forecast_gap_pct"] = np.where(alerts["D+3 Plan Reference"] > 0,
        alerts["forecast_gap_vs_plan"] / alerts["D+3 Plan Reference"] * 100, np.nan)
    needs_review = alerts["forecast_gap_vs_plan"].abs() >= np.maximum(10, alerts["D+3 Plan Reference"] * .2)
    alerts["alert_type"] = np.where(needs_review,
        np.where(alerts["forecast_gap_vs_plan"] > 0, "upward_review", "downward_review"), "within_plan_band")

    docs: list[dict[str, Any]] = []
    alert_map = {str(row["part_number"]): clean(row) for row in alerts.to_dict("records")}
    for row in parts.to_dict("records"):
        part_number = str(row["part_number"])
        part_id = str(row.get("document_id") or slug(part_number))
        part_history = history[history["part_number"].eq(part_number)].sort_values("date")
        part_forecasts = forecasts[forecasts["part_number"].eq(part_number)].sort_values("target_date")
        metrics = part_metrics[part_metrics["part_number"].eq(part_number)].sort_values("MAE")
        docs.append(document(
            f"{base}/demandParts/{part_id}",
            {
                **clean({k: v for k, v in row.items() if k != "document_id"}),
                "recommended_model": policy if len(part_forecasts) else None,
                "recommended_model_test_mae": clean(metrics.loc[metrics["Model"].eq(policy), "MAE"].iloc[0]) if len(metrics.loc[metrics["Model"].eq(policy)]) else None,
                "selectionBasis": metadata['selection']['selection_basis'],
                "auxiliaryModel": auxiliary,
                "evaluationLabel": "retrospective_previously_observed_period",
                "quarantined": part_number in metadata['quarantined_parts'],
                "history": records(part_history, drop=("document_id", "part_number")),
                "forecasts": records(part_forecasts, drop=("document_id", "part_number")),
                "modelMetrics": records(metrics, drop=("document_id", "part_number")),
                "latestAlert": alert_map.get(part_number),
            },
        ))

    for row in model_metrics.to_dict("records"):
        model_id = str(row.get("document_id") or slug(row["Model"]))
        docs.append(document(f"{base}/demandModels/{model_id}", {k: v for k, v in row.items() if k != "document_id"}))

    latest_date = str(forecasts["target_date"].max())
    latest = forecasts[forecasts["target_date"].astype(str).eq(latest_date)].copy()
    latest_alerts = alerts[alerts["target_date"].astype(str).eq(latest_date)].copy()
    review = latest_alerts[~latest_alerts["alert_type"].eq("within_plan_band")].copy()
    recommended_total = float(latest[policy].sum())
    plan_total = float(latest["D+3 Plan Reference"].sum())
    summary = {
        "targetDate": latest_date,
        "recommendedForecast": round(recommended_total, 2),
        "planReference": round(plan_total, 2),
        "planGap": round(recommended_total - plan_total, 2),
        "planGapPct": round((recommended_total - plan_total) / plan_total * 100, 2) if plan_total else None,
        "reviewCount": int(len(review)),
        "upwardReviewCount": int((review["forecast_gap_vs_plan"] > 0).sum()),
        "downwardReviewCount": int((review["forecast_gap_vs_plan"] < 0).sum()),
        "bestWalkForwardModel": str(walk_forward.sort_values("MAE").iloc[0]["Model"]),
        "bestWalkForwardMae": float(walk_forward.sort_values("MAE").iloc[0]["MAE"]),
        "partCount": int(parts["part_number"].nunique()),
        "forecastPartCount": int(latest["part_number"].nunique()),
        "operatingModel": policy,
        "auxiliaryModel": auxiliary,
        "quarantinedParts": metadata['quarantined_parts'],
        "evaluationLabel": "retrospective_previously_observed_period",
        "modelVersion": metadata['version'],
    }
    docs.append(document(f"{base}/demandOverview/current", summary))
    docs.append(document(f"{base}/demandConfig/current", {
        "schemaVersion": SCHEMA_VERSION,
        "dataVersion": DATA_VERSION,
        "primaryModel": policy,
        "auxiliaryModel": auxiliary,
        "selectionBasis": metadata['selection']['selection_basis'],
        "modelVersion": metadata['version'],
        "evaluationLabel": "retrospective_previously_observed_period",
        "newUnseenValidationAvailable": False,
        "quarantinedParts": metadata['quarantined_parts'],
        "uncertainty": metadata['uncertainty'],
        "featureContract": metadata['feature_contract'],
        "modelMetrics": records(model_metrics, drop=("document_id",)),
        "walkForwardMetrics": records(walk_forward),
        "dataQuality": records(data_quality),
        "source": "KAMP 공급망 최적화 AI 데이터셋",
        "warning": "일별 최종 로그 확정 이후의 예측입니다. 마지막 기간은 이미 관찰한 자료의 회고 평가이며, 재고 최적화·서비스 수준 보장이 아닙니다.",
    }))

    actions: list[dict[str, Any]] = []
    if not review.empty:
        ranked = review.assign(abs_gap=review["forecast_gap_vs_plan"].abs()).sort_values("abs_gap", ascending=False).head(3)
        for priority, row in enumerate(ranked.to_dict("records"), 1):
            actions.append({
                "id": f"demand-{slug(row['part_number'])}",
                "priority": priority,
                "track": "demand",
                "trackLabel": "발주량",
                "target": row["part_number"],
                "title": f"계획 대비 {float(row['forecast_gap_vs_plan']):+,.0f}개 차이 검토",
                "severity": "warning",
                "status": "unconfirmed",
                "statusLabel": "미확인",
                "owner": None,
                "detailPath": "/demand",
            })
    return docs, summary, actions


def build_maintenance(base: str) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    seed_dir = ROOT / "배터리 예지보전 모델" / "firestore" / "seed"
    docs: list[dict[str, Any]] = []
    project = read_jsonl(seed_dir / "projects.jsonl")[0]["data"]
    docs.append(document(f"{base}/maintenanceConfig/current", project))

    for item in read_jsonl(seed_dir / "model_evaluations.jsonl"):
        doc_id = item["path"].split("/")[-1]
        docs.append(document(f"{base}/maintenanceModels/{doc_id}", item["data"]))
    for item in read_jsonl(seed_dir / "data_quality.jsonl"):
        doc_id = slug(item["path"].split("/")[-1])
        docs.append(document(f"{base}/maintenanceDataQuality/{doc_id}", item["data"]))

    run_items = read_jsonl(seed_dir / "welding_runs.jsonl")
    event_items = read_jsonl(seed_dir / "anomaly_events.jsonl")
    events_by_run: dict[str, list[dict[str, Any]]] = {}
    for item in event_items:
        source = str(item["data"]["sourceFile"])
        events_by_run.setdefault(source, []).append(item["data"])

    measurements_by_run: dict[str, list[dict[str, Any]]] = {}
    for item in read_jsonl(seed_dir / "measurements.jsonl"):
        source = str(item["data"]["sourceFile"])
        measurements_by_run.setdefault(source, []).append(item["data"])

    for item in run_items:
        run = item["data"]
        source = str(run["sourceFile"])
        events = events_by_run.get(source, [])
        docs.append(document(f"{base}/maintenanceRuns/{source}", {
            **run,
            "eventCount": len(events),
            "maxRiskRatio": max((float(e["max_risk_ratio"]) for e in events), default=0.0),
            "status": "danger" if events else "normal",
        }))
        for event in events:
            event_id = f"{int(event['event_id']):04d}"
            docs.append(document(f"{base}/maintenanceRuns/{source}/events/{event_id}", event))
        rows = sorted(measurements_by_run.get(source, []), key=lambda x: int(x["sourceRow"]))
        for chunk_start in range(0, len(rows), MEASUREMENT_CHUNK_SIZE):
            chunk = rows[chunk_start:chunk_start + MEASUREMENT_CHUNK_SIZE]
            docs.append(document(
                f"{base}/maintenanceRuns/{source}/measurementChunks/{chunk_start // MEASUREMENT_CHUNK_SIZE:04d}",
                {
                    "sourceFile": source,
                    "startRow": int(chunk[0]["sourceRow"]),
                    "endRow": int(chunk[-1]["sourceRow"]),
                    "points": chunk,
                },
            ))

    selected_source = "WeldingTest_04_NG" if "WeldingTest_04_NG" in measurements_by_run else sorted(measurements_by_run)[-1]
    selected_events = events_by_run.get(selected_source, [])
    selected_rows = measurements_by_run[selected_source]
    selected_models = project.get("defaultSupervisedModel"), project.get("defaultUnsupervisedModel")
    risks = []
    affected_pages: set[int] = set()
    for row in selected_rows:
        row_risks = []
        for model_name in selected_models:
            model = row["models"].get(model_name, {})
            threshold = float(model.get("threshold") or 0)
            score = float(model.get("score") or 0)
            if threshold > 0:
                row_risks.append(score / threshold)
            if int(model.get("prediction") or 0):
                affected_pages.add(int(row["pageNo"]))
        if row_risks:
            risks.append(max(row_risks))
    summary = {
        "sourceFile": selected_source,
        "status": "danger" if selected_events else "normal",
        "maxRiskRatio": round(max(risks, default=0.0), 2),
        "eventCount": len(selected_events),
        "affectedPages": sorted(affected_pages),
        "unconfirmedCount": len(selected_events),
        "mode": "evaluation-replay",
    }
    docs.append(document(f"{base}/maintenanceOverview/current", summary))

    actions: list[dict[str, Any]] = []
    if selected_events:
        event = sorted(selected_events, key=lambda e: float(e["max_risk_ratio"]), reverse=True)[0]
        actions.append({
            "id": f"maintenance-{selected_source}-{int(event['event_id']):04d}",
            "priority": 1,
            "track": "maintenance",
            "trackLabel": "예지보전",
            "target": selected_source,
            "title": f"{event['event_type']} 점검",
            "severity": "danger",
            "status": "unconfirmed",
            "statusLabel": "미확인",
            "owner": None,
            "detailPath": "/maintenance",
        })
    return docs, summary, actions


def infer_quality_metadata(file_name: str) -> dict[str, str]:
    return {
        "mode": "discharge" if "dchg" in file_name.lower() else "charge",
        "modeLabel": "방전" if "dchg" in file_name.lower() else "충전",
        "actualResult": "NG" if "_NG_" in file_name else "OK",
    }


def locate_quality_raw(file_name: str) -> Path:
    directory = ROOT / "배터리 품질보증 모델" / "data" / "raw_data" / "test"
    direct = directory / f"{file_name}.csv"
    if direct.exists():
        return direct
    matches = list(directory.glob(f"{file_name}*.csv"))
    if not matches:
        raise FileNotFoundError(f"품질 원본 파일을 찾을 수 없습니다: {file_name}")
    return matches[0]


def quality_test_payload(file_name: str, prediction_rows: pd.DataFrame, thresholds: dict[str, Any]) -> dict[str, Any]:
    raw = source_csv(locate_quality_raw(file_name))
    cv_cols = [c for c in raw.columns if re.fullmatch(r"M\d+CV\d+", c)]
    temp_cols = [c for c in raw.columns if re.fullmatch(r"M\d+T\d+", c)]
    data = raw[cv_cols + temp_cols].apply(pd.to_numeric, errors="coerce")
    # Match the current model's physical-range cleaning, without using labels.
    data[cv_cols] = data[cv_cols].where(data[cv_cols].ge(2) & data[cv_cols].le(5))
    data[temp_cols] = data[temp_cols].where(data[temp_cols].ge(-20) & data[temp_cols].le(100))
    data = data.interpolate(limit_direction="both").ffill().bfill()
    for columns in (cv_cols, temp_cols):
        valid = [c for c in columns if not data[c].isna().all()]
        for col in set(columns) - set(valid):
            data[col] = data[valid].median(axis=1)
    if len(data) != len(prediction_rows) or data.isna().any().any():
        raise ValueError(f"품질 시계열 정합성 오류: {file_name}")
    n = len(data)
    prediction_rows = prediction_rows.sort_values("시점").reset_index(drop=True)
    if not np.array_equal(prediction_rows["시점"].to_numpy(), np.arange(n)):
        raise ValueError(f"품질 시점 순서 오류: {file_name}")
    voltage = data[cv_cols].to_numpy(float)
    temperature = data[temp_cols].to_numpy(float)
    v_mean = voltage.mean(axis=1)
    v_std = voltage.std(axis=1)
    v_z = np.abs((voltage - v_mean[:, None]) / (v_std[:, None] + 1e-9))
    t_range = temperature.max(axis=1) - temperature.min(axis=1)
    max_z = v_z.max(axis=1)
    prediction = prediction_rows["예측"].to_numpy(dtype=bool)
    starts = np.flatnonzero(prediction & ~np.r_[False, prediction[:-1]])
    ends = np.flatnonzero(prediction & ~np.r_[prediction[1:], False])
    representative = int(np.argmax(np.maximum(
        prediction_rows["SPE"].to_numpy() / thresholds["spe"],
        prediction_rows["Hotelling_T2"].to_numpy() / thresholds["t2"])))
    indices = np.unique(np.linspace(0, max(n - 1, 0), min(MAX_DISPLAY_POINTS, n), dtype=int))
    series = []
    for idx in indices:
        pred = prediction_rows.iloc[int(idx)]
        series.append({
            "index": int(idx),
            "progressPct": round(int(idx) / max(n - 1, 1) * 100, 2),
            "actualLabel": int(pred["label"]),
            "pcaPrediction": int(pred["예측"]),
            "pcaQ": float(pred["SPE"]),
            "pcaT2": float(pred["Hotelling_T2"]),
            "voltageMean": round(float(v_mean[idx]), 5),
            "voltageRange": round(float(voltage[idx].max() - voltage[idx].min()), 5),
            "maxCellZ": round(float(max_z[idx]), 4),
            "temperatureMean": round(float(temperature[idx].mean()), 4),
            "temperatureRange": round(float(t_range[idx]), 4),
        })
    heatmap = []
    for column_index, column in enumerate(cv_cols):
        match = re.fullmatch(r"M(\d+)CV(\d+)", column)
        heatmap.append({
            "cellId": column,
            "module": int(match.group(1)) if match else None,
            "cell": int(match.group(2)) if match else None,
            "voltage": round(float(voltage[representative, column_index]), 5),
            "zScore": round(float(v_z[representative, column_index]), 4),
        })
    meta = infer_quality_metadata(file_name)
    location_scores = v_z[prediction].mean(axis=0) if prediction.any() else np.zeros(len(cv_cols))
    suspected = [{"순위": rank + 1, "셀": cv_cols[int(i)], "이상점수": round(float(location_scores[i]), 4)}
                 for rank, i in enumerate(np.argsort(-location_scores, kind="stable")[:10])] if prediction.any() else []
    return {
        "testId": file_name,
        **meta,
        "rows": n,
        "representativeIndex": representative,
        "maxTemperatureRange": round(float(t_range.max()), 4),
        "maxCellZ": round(float(max_z.max()), 4),
        "modelVersion": "PCA-T2-SPE-train10",
        "evaluationSet": "Test03,Test04,Test06,Test07,Test08",
        "pcaThresholds": thresholds,
        "aiStatus": "review_required" if prediction.any() else "normal",
        "decision": "pending",
        "decisionLabel": "작업자 판정 대기",
        "abnormalPointCount": int(prediction.sum()),
        "abnormalSegmentCount": len(starts),
        "anomalySegments": [{"startRow": int(a), "endRow": int(b), "rows": int(b-a+1)} for a, b in zip(starts, ends)],
        "displaySampling": {"method": "uniform", "fullRows": n, "displayRows": len(indices), "aggregatesUseFullRows": True},
        "locationScoreBasis": "mean absolute cell z-score over predicted anomaly rows; no ground-truth labels",
        "series": series,
        "cellHeatmap": heatmap,
        "temperatureHeatmap": [{"sensorId": c, "temperature": round(float(temperature[representative, i]), 4)} for i, c in enumerate(temp_cols)],
        "suspectedCells": suspected,
    }


def build_quality(base: str) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    output = ROOT / "배터리 품질보증 모델" / "output" / "models"
    predictions = source_csv(output / "model_C_pca_t2_spe_이상점수.csv")
    test_results = source_csv(output / "00_모델별_파일별결과.csv")
    metrics = source_csv(output / "00_모델별_성능비교.csv")
    cross_validation = source_csv(output / "00_모델별_교차검증.csv")
    data_quality = source_csv(output / "00_데이터_요약.csv")
    tuning = source_csv(output / "model_C_pca_t2_spe_관리한계탐색.csv")
    eligible = tuning[tuning["Precision"].ge(.9)]
    selected_threshold = eligible.sort_values("Recall", ascending=False, kind="stable").iloc[0] if len(eligible) else tuning.loc[tuning["F1-score"].idxmax()]
    thresholds = {"t2": float(selected_threshold["T2 한계"]), "spe": float(selected_threshold["SPE 한계"]),
                  "consecutivePoints": int(selected_threshold["연속 k"]), "normalQuantile": float(selected_threshold["분위수"]),
                  "source": "saved training-only threshold search; rounded CSV values; predictions preserved"}

    docs: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}
    for file_name, group in predictions.groupby("파일", sort=True):
        payload = quality_test_payload(str(file_name), group, thresholds)
        payloads[str(file_name)] = payload
        docs.append(document(f"{base}/qualityTests/{file_name}", payload))

    for row in metrics.to_dict("records"):
        model_id = slug(f"{row['모델']}-{row['구분']}")
        docs.append(document(f"{base}/qualityModels/{model_id}", {**row, "dataVersion": DATA_VERSION,
            "evaluationUnit": "timepoint", "evaluationSet": "locked_test" if row["구분"] == "테스트" else "development"}))
    mtad = source_csv(output.parent / "metrics_mtadgan_normal_locked_final.csv")
    totals = mtad[["tp", "tn", "fp", "fn"]].sum()
    tp, tn, fp, fn = (int(totals[c]) for c in ["tp", "tn", "fp", "fn"])
    docs.append(document(f"{base}/qualityModels/{slug('MTadGAN-테스트')}", {
        "모델": "MTadGAN", "구분": "테스트", "분류": "비지도(가이드북 재현)",
        "Accuracy": (tp+tn)/(tp+tn+fp+fn), "Precision": tp/max(tp+fp, 1),
        "Recall": tp/max(tp+fn, 1), "F1-score": 2*tp/max(2*tp+fp+fn, 1),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn, "dataVersion": DATA_VERSION,
        "evaluationUnit": "aligned window endpoint", "evaluationSet": "locked_test",
        "comparisonNote": "Same locked files; window alignment changes evaluated row count."}))
    docs.append(document(f"{base}/qualityConfig/current", {
        "schemaVersion": SCHEMA_VERSION,
        "dataVersion": DATA_VERSION,
        "primaryModel": "PCA (T²·SPE)",
        "secondaryModel": "Random Forest",
        "modelVersion": "PCA-T2-SPE-train10",
        "pcaThresholds": thresholds,
        "split": {"normalReferenceFiles": 10, "supervisedDevelopmentFiles": ["Test05_NG_chg", "Test09_NG_dchg"],
                  "lockedTestFiles": sorted(payloads)},
        "crossValidation": records(cross_validation),
        "testResults": records(test_results),
        "source": "KAMP 전자부품(배터리팩) 품질보증 AI 데이터셋",
        "warning": "AI 판정은 원인 확정이 아니라 검사 우선순위를 지원합니다.",
    }))
    for row in data_quality.to_dict("records"):
        docs.append(document(f"{base}/qualityDataQuality/{slug(row['파일'])}", row))

    selected_id = "Test07_NG_dchg" if "Test07_NG_dchg" in payloads else sorted(payloads)[-1]
    selected = payloads[selected_id]
    top_cell = selected["suspectedCells"][0]["셀"] if selected["suspectedCells"] else None
    summary = {
        "testId": selected_id,
        "status": selected["aiStatus"],
        "statusLabel": "AI 검토 필요" if selected["aiStatus"] == "review_required" else "AI 정상",
        "workerDecision": selected["decision"],
        "workerDecisionLabel": selected["decisionLabel"],
        "abnormalSegmentCount": selected["abnormalSegmentCount"],
        "abnormalPointCount": selected["abnormalPointCount"],
        "suspectedCell": top_cell,
        "maxCellZ": selected["maxCellZ"],
        "maxTemperatureRange": selected["maxTemperatureRange"],
        "mode": selected["mode"],
    }
    docs.append(document(f"{base}/qualityOverview/current", summary))
    actions = [{
        "id": f"quality-{slug(selected_id)}",
        "priority": 1,
        "track": "quality",
        "trackLabel": "품질보증",
        "target": selected_id,
        "title": f"{top_cell or '의심 셀'} 확인 및 재시험 검토",
        "severity": "danger",
        "status": "pending_action",
        "statusLabel": "조치 대기",
        "owner": None,
        "detailPath": "/quality",
    }]
    return docs, summary, actions


def validate_documents(documents: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for item in documents:
        path = item["path"]
        if len(path.split("/")) % 2:
            raise ValueError(f"Firestore 문서 경로가 아닙니다: {path}")
        if path in seen:
            raise ValueError(f"중복 문서 경로: {path}")
        size = len(json.dumps(item["data"], ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        if size >= 900_000:
            raise ValueError(f"문서 크기가 안전 한도를 넘습니다 ({size:,} bytes): {path}")
        seen.add(path)


def write_seed(seed_dir: Path, groups: dict[str, list[dict[str, Any]]], workspace_id: str) -> dict[str, Any]:
    previous_manifest = seed_dir / "manifest.json"
    if previous_manifest.exists():
        previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
        if previous.get("schemaVersion") == 1:
            archive = seed_dir.parent / "versions" / "seed-v1"
            archive.mkdir(parents=True, exist_ok=True)
            for filename in ["manifest.json", *(f["file"] for f in previous["files"])]:
                source, target = seed_dir / filename, archive / filename
                if target.exists() and target.read_bytes() != source.read_bytes():
                    raise ValueError(f"기존 v1 백업과 다릅니다: {target}")
                shutil.copy2(source, target)
        elif previous.get('dataVersion') != DATA_VERSION:
            archive = seed_dir.parent / 'versions' / ('seed-' + previous['dataVersion'])
            archive.mkdir(parents=True, exist_ok=True)
            for filename in ['manifest.json', *(f['file'] for f in previous['files'])]:
                source, target = seed_dir / filename, archive / filename
                if target.exists() and target.read_bytes() != source.read_bytes():
                    raise ValueError(f'Archive differs: {target}')
                shutil.copy2(source, target)
    seed_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for name, docs in groups.items():
        path = seed_dir / f"{name}.jsonl"
        content = "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in docs)
        path.write_text(content, encoding="utf-8")
        files.append({
            "file": path.name,
            "documents": len(docs),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    manifest = {
        "format": "firestore-document-jsonl-v1",
        "schemaVersion": SCHEMA_VERSION,
        "dataVersion": DATA_VERSION,
        "supersedes": "2026-09-26.v2",
        "deploymentStatus": "local_only_pending_upload",
        "sourceArtifacts": [{"path": path, "sha256": digest} for path, digest in sorted(SOURCE_ARTIFACTS.items())],
        "workspaceId": workspace_id,
        "rootDocument": f"manufacturingAi/{workspace_id}",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "totalDocuments": sum(item["documents"] for item in files),
    }
    (seed_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="세 모델과 통합 현황용 Firestore 압축 시드 생성")
    parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID)
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    args = parser.parse_args()
    base = f"manufacturingAi/{args.workspace_id}"

    demand_docs, demand_summary, demand_actions = build_demand(base)
    maintenance_docs, maintenance_summary, maintenance_actions = build_maintenance(base)
    quality_docs, quality_summary, quality_actions = build_quality(base)
    all_actions = maintenance_actions + quality_actions + demand_actions
    for priority, item in enumerate(all_actions, 1):
        item["priority"] = priority

    core_docs = [
        document(base, {
            "name": "BatteryFlow AI 운영센터",
            "schemaVersion": SCHEMA_VERSION,
            "dataVersion": DATA_VERSION,
            "tracks": ["demand", "maintenance", "quality"],
            "rawDataMerged": False,
            "updatedAt": datetime.now(timezone.utc),
        }),
        document(f"{base}/overview/current", {
            "updatedAt": datetime.now(timezone.utc),
            "connectionStatus": "local_seed",
            "dataVersion": DATA_VERSION,
            "connectedTrackCount": 3,
            "rawDataMerged": False,
            "demand": demand_summary,
            "maintenance": maintenance_summary,
            "quality": quality_summary,
            "actionCounts": {
                "critical": sum(a["severity"] == "danger" for a in all_actions),
                "warning": sum(a["severity"] == "warning" for a in all_actions),
                "unconfirmed": sum(a["status"] == "unconfirmed" for a in all_actions),
                "pending": sum(a["status"] == "pending_action" for a in all_actions),
            },
        }),
    ]
    core_docs.extend(document(f"{base}/actions/{item['id']}", item) for item in all_actions)
    groups = {
        "core": core_docs,
        "demand": demand_docs,
        "maintenance": maintenance_docs,
        "quality": quality_docs,
    }
    for documents in groups.values():
        for doc in documents:
            doc["data"]["dataVersion"] = DATA_VERSION
            doc["data"]["schemaVersion"] = SCHEMA_VERSION
    validate_documents([doc for docs in groups.values() for doc in docs])
    manifest = write_seed(args.seed_dir, groups, args.workspace_id)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
