from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app as dashboard  # noqa: E402


SEED_DIR = ROOT / "firestore" / "seed"
PROJECT_ID = "track-b-final-v2"


def native(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: native(value) for key, value in record.items()}


def write_jsonl(name: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
    path = SEED_DIR / name
    payload = "".join(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n"
        for document in documents
    )
    path.write_text(payload, encoding="utf-8")
    return {
        "file": name,
        "documents": len(documents),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def project_document(generated_at: str) -> dict[str, Any]:
    manifest = dashboard.manifest
    return {
        "path": f"projects/{PROJECT_ID}",
        "data": {
            "name": "배터리팩 레이저 용접 예지보전",
            "pipelineVersion": manifest["pipeline_version"],
            "evaluationProtocol": manifest["evaluation_protocol"],
            "selectedModel": manifest["winner_selected_on_validation"],
            "defaultSupervisedModel": dashboard.DEFAULT_SUPERVISED,
            "defaultUnsupervisedModel": dashboard.DEFAULT_UNSUPERVISED,
            "generatedAt": generated_at,
            "mode": "evaluation-replay",
        },
    }


def model_documents() -> list[dict[str, Any]]:
    documents = []
    for row in dashboard.metrics.to_dict("records"):
        model = str(row["model"])
        split = str(row["split"])
        documents.append(
            {
                "path": f"projects/{PROJECT_ID}/modelEvaluations/{model}_{split}",
                "data": clean_record(row),
            }
        )
    return documents


def quality_documents() -> list[dict[str, Any]]:
    return [
        {
            "path": f"projects/{PROJECT_ID}/dataQuality/{row['file']}",
            "data": clean_record(row),
        }
        for row in dashboard.data_quality.to_dict("records")
    ]


def run_and_measurement_documents() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs: list[dict[str, Any]] = []
    measurements: list[dict[str, Any]] = []
    model_names = dashboard.supervised_models + dashboard.unsupervised_models

    for source in dashboard.source_files:
        base = dashboard.model_rows(source, model_names[0])
        predictions = {
            model: dashboard.model_rows(source, model).set_index("source_row")
            for model in model_names
        }
        runs.append(
            {
                "path": f"projects/{PROJECT_ID}/weldingRuns/{source}",
                "data": {
                    "sourceFile": source,
                    "result": "NG" if source.endswith("NG") else "OK",
                    "rows": int(len(base)),
                    "cycles": int(base["cycle_local"].nunique()),
                    "positiveRows": int(base["label"].sum()),
                    "firstWorkingTime": str(base["WorkingTime"].iloc[0]),
                    "lastWorkingTime": str(base["WorkingTime"].iloc[-1]),
                },
            }
        )

        for row in base.itertuples(index=False):
            model_scores = {}
            for model in model_names:
                predicted = predictions[model].loc[row.source_row]
                model_scores[model] = {
                    "family": str(predicted["family"]),
                    "score": native(predicted["score"]),
                    "threshold": native(predicted["threshold"]),
                    "prediction": int(predicted["prediction"]),
                }
            measurements.append(
                {
                    "path": (
                        f"projects/{PROJECT_ID}/weldingRuns/{source}/measurements/"
                        f"{int(row.source_row):06d}"
                    ),
                    "data": {
                        "sourceFile": source,
                        "sourceRow": int(row.source_row),
                        "cycle": int(row.cycle_local),
                        "groupId": str(row.group_id),
                        "pageNo": int(row.PageNo),
                        "workingTime": str(row.WorkingTime),
                        "signals": {
                            "realPower": native(row.RealPower),
                            "setPower": native(row.SetPower),
                            "gateOnTime": native(row.GateOnTime),
                            "speed": native(row.Speed),
                            "length": native(row.Length),
                        },
                        "normalReference": {
                            "expectedPower": native(row.expected_power),
                            "scale": native(row.normal_scale),
                        },
                        "actualLabel": int(row.label),
                        "models": model_scores,
                    },
                }
            )
    return runs, measurements


def event_documents() -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for source in dashboard.source_files:
        frame = dashboard.combined_rows(
            source,
            dashboard.DEFAULT_SUPERVISED,
            dashboard.DEFAULT_UNSUPERVISED,
        )
        events = dashboard.events_for(frame)
        for row in events.to_dict("records"):
            _, actions = dashboard.recommendations(pd.Series(row))
            event_id = int(row["event_id"])
            data = clean_record(row)
            data.update(
                {
                    "sourceFile": source,
                    "supervisedModel": dashboard.DEFAULT_SUPERVISED,
                    "unsupervisedModel": dashboard.DEFAULT_UNSUPERVISED,
                    "recommendedChecks": actions,
                }
            )
            documents.append(
                {
                    "path": (
                        f"projects/{PROJECT_ID}/weldingRuns/{source}/events/"
                        f"{event_id:04d}"
                    ),
                    "data": data,
                }
            )
    return documents


def main() -> None:
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    runs, measurements = run_and_measurement_documents()
    groups = [
        ("projects.jsonl", [project_document(generated_at)]),
        ("model_evaluations.jsonl", model_documents()),
        ("data_quality.jsonl", quality_documents()),
        ("welding_runs.jsonl", runs),
        ("measurements.jsonl", measurements),
        ("anomaly_events.jsonl", event_documents()),
    ]
    files = [write_jsonl(name, documents) for name, documents in groups]
    manifest = {
        "format": "firestore-document-jsonl-v1",
        "projectDocument": f"projects/{PROJECT_ID}",
        "generatedAt": generated_at,
        "defaultModels": {
            "supervised": dashboard.DEFAULT_SUPERVISED,
            "unsupervised": dashboard.DEFAULT_UNSUPERVISED,
        },
        "files": files,
        "totalDocuments": sum(item["documents"] for item in files),
    }
    (SEED_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
