# -*- coding: utf-8 -*-
"""
설비 예지보전(LightGBM) 데이터 → Firestore 구조 추출

담당: 유현경 / DB화: 최우찬
입력: data/team/lightgbm/  (유현경님이 develop 브랜치에 올린 산출물 13종)
출력: output/firebase/pdm/

구조 설계 근거
    산출물 README의 "쓰이는 화면" 표를 그대로 따랐다.
    파일(WeldingTest_01_OK 등) 4개가 조회 단위이므로 파일을 문서 하나로 잡고,
    행 단위 데이터는 배열로 묶어 하위 문서에 담는다.
    시점마다 문서를 만들면 5,226개가 되지만, 배열로 묶으면 4개다.

컬렉션
    pdm_meta/config             임계값, 분할 정보, 설비 정보, 변수 중요도
    pdm_files/{fileId}          파일 마스터 → 드롭다운, KPI 카드
      └ series/predictions      행 단위 배열 → 실제vs예측 / 이상점수 그래프
      └ modules/judgement       모듈(사이클) 판정 배열 → 모듈 판정 표
    pdm_segments/{fileId}       이상 구간 → 점검 목록, 우선순위
    pdm_models/lightgbm         성능 지표 → 혼동행렬, 이벤트 지표
    pdm_eda/{docId}             조건별 분포 / 월별 추이 / 상관 / 품질검증 /
                                임계값 스윕 / PageNo 기준값

실행
    python src/models/extract_pdm.py
    python src/models/extract_pdm.py --max-points 2000   # 시계열 상한
"""

import os
import sys
import json
import math
import argparse

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "data", "team", "lightgbm")
OUT = os.path.join(ROOT, "output", "firebase", "pdm")

MAX_POINTS = 3000          # 파일당 시계열 상한 (Firestore 문서 1 MiB 한도 대비)


# ── 공통 ────────────────────────────────────────────────────────────
def clean(v):
    """NaN/Inf를 None으로. Firestore는 NaN을 못 받는다."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if (math.isnan(v) or math.isinf(v)) else round(float(v), 6)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return str(v) if not isinstance(v, (int, str)) else v


def records(df):
    return [{k: clean(v) for k, v in row.items()} for row in df.to_dict("records")]


def read(name, **kw):
    p = os.path.join(SRC, name)
    if not os.path.exists(p):
        sys.exit(f"[오류] 없는 파일: {p}\n       data/team/lightgbm/ 에 산출물을 넣었는지 확인하세요.")
    return pd.read_csv(p, **kw)


def write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, ensure_ascii=False, separators=(",", ":"))
    return os.path.getsize(path)


def downsample(df, n):
    """행이 너무 많으면 고르게 솎아낸다. 단 이상 판정 행은 반드시 남긴다."""
    if len(df) <= n:
        return df, len(df)
    keep = df["anomaly_pred"] == 1 if "anomaly_pred" in df.columns else pd.Series(False, index=df.index)
    rest = df[~keep]
    room = max(n - int(keep.sum()), 0)
    if room and len(rest) > room:
        idx = np.linspace(0, len(rest) - 1, room).astype(int)
        rest = rest.iloc[idx]
    out = pd.concat([df[keep], rest]).sort_index()
    return out, len(df)


# ── 추출 ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-points", type=int, default=MAX_POINTS)
    a = ap.parse_args()

    pred = read("predictions.csv")
    modj = read("module_judgement.csv")
    segs = read("anomaly_segments.csv")
    mfile = read("metrics_by_file.csv")
    rmet = read("regression_metrics.csv")
    equip = read("equipment_info.csv")
    fimp = read("lightgbm_feature_importance.csv")
    results = json.load(open(os.path.join(SRC, "lightgbm_results.json"), encoding="utf-8"))

    files = sorted(pred["file"].unique())
    docs, total = 0, 0

    # ── 파일 마스터 + 하위 문서 ────────────────────────────────────
    masters = []
    for f in files:
        p = pred[pred["file"] == f]
        m = modj[modj["file"] == f]
        s = segs[segs["file"] == f]
        mf = mfile[mfile["file"] == f]
        rm = rmet[(rmet["file"] == f) & (rmet["subset"] == "전체 행")]

        n_anom = int((p["anomaly_pred"] == 1).sum())
        judge = m["judgement"].value_counts().to_dict() if len(m) else {}

        master = {
            "file_id": f,
            "grade": "NG" if f.endswith("_NG") else "OK",
            "n_rows": len(p),
            "n_cycles": int(p["cycle_id"].nunique()),
            "anomaly_rows": n_anom,
            "anomaly_ratio": round(n_anom / max(len(p), 1), 4),
            "max_anomaly_score": clean(p["anomaly_score"].max()),
            "threshold": clean(p["threshold"].iloc[0]),
            "start_time": str(p["WorkingTime"].iloc[0]),
            "end_time": str(p["WorkingTime"].iloc[-1]),
            "n_segments": len(s),
            "module_judgement": {k: int(v) for k, v in judge.items()},
            "power_mean": clean(p["RealPower"].mean()),
            "power_min": clean(p["RealPower"].min()),
            "power_max": clean(p["RealPower"].max()),
            "set_powers": sorted(int(x) for x in p["SetPower"].dropna().unique()),
            "metrics": records(mf)[0] if len(mf) else None,
            "regression": records(rm)[0] if len(rm) else None,
        }
        masters.append(master)

        # 시계열 — 열 단위 배열로 담는다 (행 단위 dict 5천 개보다 훨씬 작다)
        ds, n_total = downsample(p, a.max_points)
        cols = ["row_index", "WorkingTime", "cycle_id", "PageNo", "Speed", "Length",
                "SetPower", "GateOnTime", "RealPower", "predicted_power",
                "page_median_power", "abs_residual", "anomaly_score", "risk_ratio",
                "risk_level", "anomaly_pred", "label", "inspection_reason"]
        series = {"file_id": f, "n_total": n_total, "n_points": len(ds),
                  "threshold": clean(p["threshold"].iloc[0])}
        for c in cols:
            if c in ds.columns:
                series[c] = [clean(v) for v in ds[c].tolist()]
        total += write(os.path.join(OUT, "series", f + ".json"), series); docs += 1

        # 모듈 판정
        mod = {"file_id": f, "n": len(m), "items": records(m.drop(columns=["file"]))}
        total += write(os.path.join(OUT, "modules", f + ".json"), mod); docs += 1

        # 이상 구간
        if len(s):
            seg = {"file_id": f, "n": len(s), "items": records(s.drop(columns=["file"]))}
            total += write(os.path.join(OUT, "segments", f + ".json"), seg); docs += 1

    total += write(os.path.join(OUT, "files.json"), masters); docs += len(masters)

    # ── 메타 ──────────────────────────────────────────────────────
    meta = {
        "generated_at": pd.Timestamp.now('UTC').isoformat(),
        "module": "설비 예지보전 (용접)",
        "owner": "유현경",
        "model": "LightGBM (정상 출력 회귀 + 잔차 기반 이상탐지)",
        "threshold": results.get("threshold"),
        "threshold_quantile": results.get("protocol", {}).get("threshold_quantile"),
        "split": results.get("split"),
        "best_iteration": results.get("best_iteration"),
        "files": files,
        "page_range": [1, 39],
        "feature_importance": records(fimp),
        "equipment": records(equip),
    }
    total += write(os.path.join(OUT, "meta.json"), meta); docs += 1

    # ── 모델 성능 ─────────────────────────────────────────────────
    overall = next((x for x in results.get("summary", []) if x.get("file") == "전체 테스트"), {})
    model = {"model": "LightGBM", "overall": {k: clean(v) for k, v in overall.items()},
             "by_file": records(mfile), "regression": records(rmet)}
    total += write(os.path.join(OUT, "models.json"), model); docs += 1

    # ── EDA / 참고 표 ─────────────────────────────────────────────
    extras = {
        "condition_summary": read("eda_condition_summary.csv"),
        "monthly_trend": read("eda_monthly_trend.csv"),
        "correlation": read("eda_correlation.csv"),
        "data_quality": read("data_quality_summary.csv"),
        "threshold_sweep": read("threshold_sweep.csv"),
        "baseline_by_pageno": read("model_baseline_by_pageno.csv"),
    }
    eda = {}
    for k, df in extras.items():
        eda[k] = {"n": len(df), "items": records(df)}
    total += write(os.path.join(OUT, "eda.json"), eda); docs += len(eda)

    # ── 요약 ──────────────────────────────────────────────────────
    big = []
    for dp, _, fs in os.walk(OUT):
        for fn in fs:
            sz = os.path.getsize(os.path.join(dp, fn))
            if sz > 900_000:
                big.append((os.path.relpath(os.path.join(dp, fn), OUT), sz))

    print("=" * 62)
    print("  설비 예지보전 → Firestore 구조 추출 완료")
    print("=" * 62)
    for m in masters:
        print(f"  {m['file_id']:20s} {m['n_rows']:5,d}행 / 사이클 {m['n_cycles']:3d} / "
              f"이상 {m['anomaly_ratio']*100:5.1f}% / 구간 {m['n_segments']}")
    print(f"  {'-'*58}")
    print(f"  총 문서       {docs:5d}개   (무료 쓰기 한도 20,000/일)")
    print(f"  총 용량       {total/1024/1024:5.2f} MB (무료 저장 한도 1 GiB)")
    print(f"  화면 1회 조회  약 4 reads")
    print(f"  저장 위치     {OUT}")
    if big:
        print("\n  [경고] 1 MiB에 근접한 문서 — --max-points 를 줄이세요:")
        for n, s in big:
            print(f"    {n} {s/1024:.0f} KB")
    print("=" * 62)


if __name__ == "__main__":
    main()
