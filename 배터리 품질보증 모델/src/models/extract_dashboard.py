# -*- coding: utf-8 -*-
"""
대시보드용 확장 추출 — 셀 전압 + 모듈 온도 채널, 4대 불량 유형 판정

extract_firebase.py 결과(output/firebase)에 다음을 덧붙인다.
    · 모듈 온도 채널 스냅샷 (16모듈 × 2센서)
    · 가이드북 4대 불량 유형 지표 4종 + 정상 팩 3σ 임계값

4대 불량 유형 지표
    용량불량       한 시점에서 다음 시점 사이 셀 전압이 가장 크게 튄 폭 (mV)
    용접불량       팩 평균보다 가장 많이 내려앉은 셀의 낙폭 (mV)
    센싱와이어불량 한 모듈 안에서 이웃한 셀끼리 벌어진 최대 전압차 (mV)
    센서불량       모듈 온도 센서가 중앙값에서 가장 멀어진 폭 (℃)

임계값은 학습용 정상 팩의 평균 + 3σ (교안 Ch46 3σ 규칙).

실행
    python src/models/extract_dashboard.py
출력
    output/firebase/dashboard.json
"""

import os
import re
import sys
import glob
import json

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRE = os.path.join(ROOT, "data", "preprocessed")
FB = os.path.join(ROOT, "output", "firebase")
RAW_TEST = os.path.join(ROOT, "data", "raw_data", "test")

CV_RE = re.compile(r"M(\d+)CV(\d+)$")
TP_RE = re.compile(r"M(\d+)T(\d+)$")


def snapshot(vals):
    """한 시점의 채널 값 → 값 / 중앙값 대비 z / 범위."""
    v = np.asarray(vals, dtype=float)
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return None
    mu, sd = float(np.nanmean(v)), float(np.nanstd(v))
    sd = sd if sd > 1e-9 else 1e-9
    z = np.abs(v - mu) / sd
    return {
        "v": [round(float(x), 4) if np.isfinite(x) else None for x in v],
        "z": [round(float(x), 2) if np.isfinite(x) else 0.0 for x in z],
        "v_min": round(float(np.nanmin(v)), 4),
        "v_max": round(float(np.nanmax(v)), 4),
    }


def defect_metrics(cv, tp, mods, peak):
    """4대 불량 유형 지표. cv/tp는 (행, 채널) 배열."""
    out = {}

    # 용량불량 — 시점 간 셀 전압 급변 (mV)
    if cv.shape[0] > 1:
        d = np.abs(np.diff(cv, axis=0))
        out["capacity"] = round(float(np.nanmax(d)) * 1000, 2) if np.isfinite(d).any() else 0.0
    else:
        out["capacity"] = 0.0

    row = cv[peak]
    mu = float(np.nanmean(row))
    # 용접불량 — 팩 평균 대비 최대 낙폭 (mV)
    out["weld"] = round(max(0.0, mu - float(np.nanmin(row))) * 1000, 2)

    # 센싱와이어불량 — 모듈 내 이웃 셀 간 최대 전압차 (mV)
    worst = 0.0
    for idx in mods.values():
        seg = row[idx]
        if len(seg) > 1:
            g = np.nanmax(np.abs(np.diff(seg)))
            if np.isfinite(g):
                worst = max(worst, float(g))
    out["wire"] = round(worst * 1000, 2)

    # 센서불량 — 모듈 온도 센서의 중앙값 대비 최대 이탈 (℃)
    trow = tp[peak]
    med = float(np.nanmedian(trow))
    out["sensor"] = round(float(np.nanmax(np.abs(trow - med))), 2)
    return out


def main():
    packs = json.load(open(os.path.join(FB, "packs.json"), encoding="utf-8"))
    heat = {}
    for f in glob.glob(os.path.join(FB, "heatmap", "*.json")):
        d = json.load(open(f, encoding="utf-8"))
        heat[d["pack_id"]] = d

    result, temp_ids, cell_ids = {}, None, None
    for i, p in enumerate(packs, 1):
        pid = p["pack_id"]
        sub = "test" if p["source"] == "test" else "train"
        # 시험 팩은 전처리본이 라벨만 담고 있어 원본에서 읽는다 (파일명에 공백이 섞인 것도 있음)
        if sub == "test":
            cand = glob.glob(os.path.join(RAW_TEST, pid + "*.csv"))
            src = cand[0] if cand else ""
        else:
            src = os.path.join(PRE, sub, pid + ".csv")
        if not src or not os.path.exists(src):
            print(f"  [건너뜀] {pid} 파일 없음")
            continue

        d = pd.read_csv(src)
        cvc = [c for c in d.columns if CV_RE.match(c)]
        tpc = [c for c in d.columns if TP_RE.match(c)]
        if not cvc or not tpc:
            print(f"  [건너뜀] {pid} 채널 불일치")
            continue
        cell_ids = cell_ids or cvc
        temp_ids = temp_ids or tpc

        cv = d[cvc].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        tp = d[tpc].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

        mods = {}
        for j, c in enumerate(cvc):
            mods.setdefault(CV_RE.match(c).group(1), []).append(j)

        n = cv.shape[0]
        hm = heat.get(pid, {}).get("snapshots", {})
        peak = min(int(hm.get("peak", {}).get("t", n - 1)), n - 1)
        last = n - 1

        result[pid] = {
            "temp": {"peak": snapshot(tp[peak]), "last": snapshot(tp[last])},
            "metrics": defect_metrics(cv, tp, mods, peak),
            "t_peak": peak, "t_last": last,
        }
        if i % 20 == 0:
            print(f"  {i}/{len(packs)}")

    # 임계값 = 학습용 정상 팩 평균 + 3σ
    norm = [p["pack_id"] for p in packs
            if p["source"] == "train" and p["pack_id"] in result]
    thr = {}
    for k in ("capacity", "weld", "wire", "sensor"):
        a = np.array([result[q]["metrics"][k] for q in norm], dtype=float)
        thr[k] = round(float(a.mean() + 3 * a.std()), 2)

    for pid, r in result.items():
        r["flags"] = {k: bool(r["metrics"][k] > thr[k]) for k in thr}

    out = {"packs": result, "thresholds": thr,
           "temp_ids": temp_ids, "cell_ids": cell_ids,
           "n_normal_ref": len(norm)}
    dst = os.path.join(FB, "dashboard.json")
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))

    print(f"\n완료: {len(result)}개 팩 → {dst}")
    print(f"정상 팩 {len(norm)}개 기준 임계값 (평균+3σ)")
    for k, v in thr.items():
        hit = sum(r["flags"][k] for r in result.values())
        print(f"  {k:9s} {v:8.2f}   초과 팩 {hit}개")


if __name__ == "__main__":
    main()
