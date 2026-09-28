# -*- coding: utf-8 -*-
"""개발 파일(Test05/Test09)만으로 MTadGAN 운영 임계값을 선택한다."""

import itertools
import json
import os

import numpy as np
import pandas as pd

from battery_quality_mtadgan import Anomaly, PATHS, evaluate


DEV = {
    "chg": ("Test05_NG_chg", "Test05_NG_chg_Label.csv"),
    "dchg": ("Test09_NG_dchg", "Test09_NG_dchg_Label.csv"),
}
PADDINGS = (10, 25, 50, 100)


def load_dev(mode):
    tag, label_name = DEV[mode]
    score_df = pd.read_csv(os.path.join(PATHS["output"], "scores_%s.csv" % tag))
    labels = pd.read_csv(os.path.join(PATHS["pre_test"], label_name))["label"].values
    return tag, score_df["anomaly_score"].values, score_df["index"].values, labels


def metrics_for(scores, index, labels, threshold, padding):
    anomalies = Anomaly().find_anomalies(
        scores, index, anomaly_padding=padding, min_percent=0.1,
        fixed_threshold=True, threshold_value=threshold)
    return evaluate(scores, anomalies, labels, "selection", np.empty((0, 10, 3)), plot=False)


def main():
    calibration_path = os.path.join(PATHS["output"], "mtadgan_detection_config.json")
    with open(calibration_path, "r", encoding="utf-8") as f:
        calibration = json.load(f)

    grids = calibration["threshold_grid_by_mode"]
    dev = {mode: load_dev(mode) for mode in DEV}
    rows = []
    for (q_chg, threshold_chg), (q_dchg, threshold_dchg), padding in itertools.product(
            grids["chg"].items(), grids["dchg"].items(), PADDINGS):
        results = {}
        for mode, threshold in (("chg", threshold_chg), ("dchg", threshold_dchg)):
            _, scores, index, labels = dev[mode]
            results[mode] = metrics_for(scores, index, labels, threshold, padding)

        tp = sum(r["tp"] for r in results.values())
        tn = sum(r["tn"] for r in results.values())
        fp = sum(r["fp"] for r in results.values())
        fn = sum(r["fn"] for r in results.values())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "quantile_chg": float(q_chg),
            "threshold_chg": float(threshold_chg),
            "quantile_dchg": float(q_dchg),
            "threshold_dchg": float(threshold_dchg),
            "anomaly_padding": int(padding),
            "accuracy": (tp + tn) / (tp + tn + fp + fn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        })

    table = pd.DataFrame(rows).sort_values(
        ["f1", "precision", "recall", "quantile_chg", "quantile_dchg"],
        ascending=[False, False, False, False, False]).reset_index(drop=True)
    best = table.iloc[0]
    config = {
        "selection_data": [DEV["chg"][0], DEV["dchg"][0]],
        "selection_metric": "pooled_f1",
        "threshold_source": "normal_training_score_quantile_grid",
        "threshold_quantile_by_mode": {
            "chg": float(best["quantile_chg"]),
            "dchg": float(best["quantile_dchg"]),
        },
        "threshold_by_mode": {
            "chg": float(best["threshold_chg"]),
            "dchg": float(best["threshold_dchg"]),
        },
        "anomaly_padding": int(best["anomaly_padding"]),
        "development_metrics": {
            key: (int(value) if key in ("tp", "tn", "fp", "fn") else float(value))
            for key, value in best[["accuracy", "precision", "recall", "f1",
                                    "tp", "tn", "fp", "fn"]].items()
        },
    }
    table.to_csv(os.path.join(PATHS["output"], "mtadgan_operating_point_search.csv"),
                 index=False)
    out = os.path.join(PATHS["output"], "mtadgan_operating_point.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(json.dumps(config, ensure_ascii=False, indent=2))
    print("저장:", out)


if __name__ == "__main__":
    main()
