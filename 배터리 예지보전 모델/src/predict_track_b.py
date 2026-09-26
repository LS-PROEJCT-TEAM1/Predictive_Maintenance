from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from track_b_final_v2 import Detector, add_identity, read_signal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "outputs" / "track_b_final_v2" / "models"
MANIFEST_PATH = PROJECT_ROOT / "outputs" / "track_b_final_v2" / "run_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a battery welding CSV with a saved Track B model")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--model", help="Saved model name; defaults to the validation-selected winner")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    model_name = args.model or manifest["winner_selected_on_validation"]
    payload = joblib.load(MODEL_DIR / f"{model_name}.joblib")
    scored_model = Detector(
        name=payload["name"],
        family=payload["family"],
        model=payload["model"],
        threshold=float(payload["threshold"]),
        feature_names=list(payload["feature_names"]),
        reference=payload["reference"],
        threshold_source=payload["threshold_source"],
        training_seconds=0.0,
    )
    frame = add_identity(read_signal(args.input_csv), args.input_csv.stem)
    frame["score"] = scored_model.score(frame)
    frame["threshold"] = scored_model.threshold
    frame["prediction"] = (frame["score"] >= scored_model.threshold).astype(int)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"model={model_name} rows={len(frame)} anomalies={int(frame['prediction'].sum())}")
    print(args.output_csv.resolve())


if __name__ == "__main__":
    main()
