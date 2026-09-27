"""이상 탐지 탭 계산 검증 — 화면 요구사항 [검증] 값과 비교한다.

실행:  dashboard 폴더에서  python -m pytest tests  (또는 python tests/test_anomaly.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tabs.anomaly import data  # noqa: E402


def test_overall_metrics():
    metrics, _ = data.get_metrics(data.get_predictions(), data.ALL_FILES,
                                  data.get_default_quantile(), row_filtered=False)
    assert round(metrics["accuracy"], 4) == 0.9975
    assert round(metrics["precision"], 4) == 0.9574
    assert metrics["recall"] == 1.0
    assert round(metrics["f1"], 4) == 0.9782
    assert round(metrics["fpr"], 4) == 0.0026
    assert (metrics["event_detected"], metrics["event_total"]) == (20, 20)
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (4921, 13, 0, 292)


def test_recomputed_metrics_match_metrics_by_file():
    """predictions 에서 다시 계산한 값 == metrics_by_file (필터가 없을 때)."""
    pred = data.get_predictions()
    by_file = data.get_metrics_by_file()
    for _, row in by_file.iterrows():
        subset = pred if row["file"] == data.ALL_FILES else pred[pred["file"] == row["file"]]
        computed = data.compute_metrics(subset)
        for key in ["tn", "fp", "fn", "tp", "event_detected", "event_total"]:
            assert computed[key] == row[key], (row["file"], key)
        assert round(computed["accuracy"], 4) == round(row["accuracy"], 4)


def test_modules():
    modules = data.get_modules()
    counts = data.judge_counts(modules)
    assert len(modules) == 134
    assert (counts[data.GOOD], counts[data.RECHECK], counts[data.SUSPECT]) == (106, 20, 8)

    first = modules[(modules["file"] == "WeldingTest_04_NG") & (modules["cycle_id"] == 1)].iloc[0]
    assert data.parse_pages(first["anomaly_page_nos"]) == {2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 18}
    assert first["anomaly_points"] == 11


def test_condition_ratio():
    ratio = data.condition_ratio(data.get_predictions()).set_index(["file", "SetPower"])
    assert "WeldingTest_01_OK" not in ratio.index.get_level_values("file")
    assert round(ratio.loc[("WeldingTest_03_NG", 35), "ratio"] * 100, 1) == 90.5
    ng04 = ratio.loc["WeldingTest_04_NG", "ratio"] * 100
    assert round(ng04.min(), 1) == 77.8 and round(ng04.max(), 1) == 82.4


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print("OK", name)
