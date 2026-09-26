# -*- coding: utf-8 -*-
"""
Firestore 업로드용 데이터 추출

대시보드 드롭다운에서 선택할 수 있는 모든 배터리(정상 102 + 시험 9 = 111개)를
Firestore 문서 구조로 변환한다.

설계 원칙
    Firestore는 '문서' 단위로 읽기 과금된다. 시점마다 문서를 만들면
    (약 50만 문서) 그래프 한 번 그리는 데 수천 건을 읽어 무료 한도를 넘긴다.
    시계열은 배열로 묶어 파일당 문서 1개로 만든다.

출력 구조 (output/firebase/)
    meta.json                셀 ID 목록, 관리한계, 모델 정보
    packs.json               팩 마스터 111개  → 드롭다운 + KPI 카드
    series/{packId}.json     시계열 배열      → 이상 점수 그래프
    heatmap/{packId}.json    셀 스냅샷 2종    → 셀 전압 Heatmap
    segments.json            이상 구간        → 그래프 하이라이트
    models.json              모델 5종 성능지표

실행
    python src/models/extract_firebase.py
"""

import os
import re
import sys
import json
import glob
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import common_data as C

OUT = os.path.join(C.ROOT, "output", "firebase")
os.makedirs(os.path.join(OUT, "series"), exist_ok=True)
os.makedirs(os.path.join(OUT, "heatmap"), exist_ok=True)

MAX_POINTS = 1000       # 시계열 다운샘플 목표 점수 (화면 폭 이상은 무의미)
K_CONSEC = 10           # 연속 초과 시점 수
UCL_Q = 0.95            # 관리한계 분위수
NG_RATIO = 0.05         # 팩 단위 NG 판정 기준 (이상 시점 비율)

# 정상 기준 = 전체 정상 팩(preprocessed/train). None 이면 전부 사용.
# 기준을 10개 팩으로 좁히면 정상 팩 오탐율이 20%까지 오른다(기준 팩의
# 셀 전압 편차 분포가 좁아 Test01/Test02 같은 정상 팩을 이상으로 본다).
REF_PACKS = None


# =============================================================================
# 파일 로딩
# =============================================================================

def list_all_files():
    """정상 팩(preprocessed/train) + 시험 파일(raw_data/test) 목록."""
    items = []
    for p in sorted(glob.glob(os.path.join(C.PRE_TRAIN, "*.csv"))):
        name = os.path.basename(p).replace(".csv", "")
        mode = "방전" if name.endswith("dchg") else "충전"
        items.append({"pack_id": name, "path": p, "source": "train",
                      "pack_no": name.split("_")[0], "mode": mode,
                      "grade": "OK", "label_path": None})
    extra = [("Test01_OK_chg", "충전", "OK", "test"),
             ("Test02_OK_dchg", "방전", "OK", "test")]
    for prefix, mode, grade, _ in list(C.TEST_FILES) + extra:
        lp = os.path.join(C.PRE_TEST, prefix + "_Label.csv")
        items.append({"pack_id": prefix, "path": C._find_raw(prefix), "source": "test",
                      "pack_no": prefix.split("_")[0], "mode": mode,
                      "grade": grade, "label_path": lp if os.path.exists(lp) else None})
    return items


def load_one(item):
    """1개 파일 → (정제된 208변수, Feature, 셀 z-score, 라벨).

    빈 파일이나 손상된 파일은 None을 돌려주고 건너뛴다.
    (전처리 단계에서 일부 파일이 0바이트로 저장된 경우가 있다)
    """
    try:
        df = pd.read_csv(item["path"])
    except Exception:
        return None
    if df.empty:
        return None
    cv, tp = C._split_columns(df)
    if not cv or not tp:
        return None
    data, _, _ = C._clean(df[cv + tp].apply(pd.to_numeric, errors="coerce"), cv, tp)
    feat, z = C.make_features(data, cv, tp, item["mode"])

    label = None
    if item["label_path"]:
        label = pd.read_csv(item["label_path"])["label"].values
        n = min(len(feat), len(label))
        feat, z, data, label = feat.iloc[:n], z[:n], data.iloc[:n], label[:n]
    return data, feat, z, label, cv, tp


# =============================================================================
# 이상 구간 추출
# =============================================================================

def to_segments(binary, min_len=1):
    segs, begin, inside = [], 0, False
    for k, v in enumerate(binary):
        if v == 1 and not inside:
            begin, inside = k, True
        elif v == 0 and inside:
            if k - begin >= min_len:
                segs.append((begin, k - 1))
            inside = False
    if inside and len(binary) - begin >= min_len:
        segs.append((begin, len(binary) - 1))
    return segs


def flag_consecutive(mask, k):
    if k <= 1:
        return mask.astype(int)
    keep = pd.Series(mask.astype(int)).rolling(k, min_periods=k).min().fillna(0).values
    out = np.zeros(len(mask), dtype=int)
    for i in np.where(keep == 1)[0]:
        out[i - k + 1:i + 1] = 1
    return out


def downsample(n, target):
    """시계열 인덱스 다운샘플. 마지막 시점은 항상 포함한다."""
    if n <= target:
        return np.arange(n)
    idx = np.linspace(0, n - 1, target).astype(int)
    return np.unique(np.append(idx, n - 1))


# =============================================================================
# 메인
# =============================================================================

def main():
    items = list_all_files()
    print(f"대상 파일 {len(items)}개 (정상 {sum(i['source']=='train' for i in items)} / "
          f"시험 {sum(i['source']=='test' for i in items)})\n")

    feats = [c for c in C.REL_FEATURES if c != "방전여부"]

    # ---- 1단계: 정상 기준 팩으로 PCA 적합 ----
    ref_ids = (REF_PACKS if REF_PACKS is not None
               else [i["pack_id"] for i in items if i["source"] == "train"])
    print(f"[1/3] 정상 기준 팩 {len(ref_ids)}개로 PCA 적합 중...")
    ref_frames, cell_cols = [], None
    for it in items:
        if it["pack_id"] in ref_ids:
            r = load_one(it)
            if r is None:
                continue
            _, f, _, _, cv, _ = r
            cell_cols = cv
            ref_frames.append(f)
    ref = pd.concat(ref_frames, ignore_index=True)

    scaler = StandardScaler().fit(ref[feats])
    pca = PCA(n_components=0.95, random_state=C.RANDOM_STATE).fit(scaler.transform(ref[feats]))

    def scores(frame):
        Z = scaler.transform(frame[feats])
        P = pca.transform(Z)
        T2 = np.sum(P ** 2 / pca.explained_variance_, axis=1)
        SPE = np.sum((Z - pca.inverse_transform(P)) ** 2, axis=1)
        return T2, SPE

    T2_ok, SPE_ok = scores(ref)
    ucl_t2 = float(np.quantile(T2_ok, UCL_Q))
    ucl_spe = float(np.quantile(SPE_ok, UCL_Q))

    # 규칙 기반 판정용 3σ 임계값 (충방전 분리)
    rule_thr = {}
    for mode in ("충전", "방전"):
        sub = ref[ref["방전여부"] == (1 if mode == "방전" else 0)] if "방전여부" in ref else ref
        sub = sub if len(sub) else ref
        rule_thr[mode] = {
            "dv": float(sub["셀전압_편차"].mean() + 3 * sub["셀전압_편차"].std()),
            "temp": float(sub["모듈온도_편차"].mean() + 3 * sub["모듈온도_편차"].std()),
        }

    print(f"      주성분 {pca.n_components_}개 / 설명분산 {pca.explained_variance_ratio_.sum():.4f}")
    print(f"      관리한계  T² > {ucl_t2:.3f}   SPE > {ucl_spe:.3f}\n")

    # ---- 2단계: 전체 파일 처리 ----
    print("[2/3] 전체 파일 처리 중...")
    packs, all_segments, skipped = [], [], []

    for n, it in enumerate(items, 1):
        r = load_one(it)
        if r is None or len(r[1]) < 50:
            skipped.append({"pack_id": it["pack_id"],
                            "reason": "행 수 부족 또는 컬럼 불일치"})
            continue
        data, feat, z, label, cv, tp = r
        T2, SPE = scores(feat)
        phi = T2 / ucl_t2 + SPE / ucl_spe
        pred = flag_consecutive((T2 > ucl_t2) | (SPE > ucl_spe), K_CONSEC)

        # --- 규칙 기반 판정 (KPI 카드) ---
        thr = rule_thr[it["mode"]]
        dv_mv = float(feat["셀전압_편차"].iloc[-1] * 1000)
        temp_dev = float(feat["모듈온도_편차"].iloc[-1])
        verdict = "NG" if (dv_mv / 1000 > thr["dv"] or temp_dev > thr["temp"]) else "OK"

        # --- 시계열 (다운샘플) ---
        ds = downsample(len(feat), MAX_POINTS)
        series = {
            "pack_id": it["pack_id"],
            "n_total": int(len(feat)),
            "n_points": int(len(ds)),
            "t": ds.tolist(),
            "t2": np.round(T2[ds], 3).tolist(),
            "spe": np.round(SPE[ds], 4).tolist(),
            "phi": np.round(phi[ds], 4).tolist(),
            "pred": pred[ds].tolist(),
            "ucl_t2": round(ucl_t2, 3),
            "ucl_spe": round(ucl_spe, 4),
        }
        if label is not None:
            series["label"] = label[ds].tolist()
        with open(os.path.join(OUT, "series", f"{it['pack_id']}.json"), "w",
                  encoding="utf-8") as fp:
            json.dump(series, fp, ensure_ascii=False, separators=(",", ":"))

        # --- Heatmap 스냅샷 2종: 마지막 시점 / 이상 점수 최대 시점 ---
        V = data[cv].values
        peak = int(np.argmax(phi))
        snaps = {"last": len(V) - 1, "peak": peak}
        hm = {"pack_id": it["pack_id"], "n_cells": len(cv), "snapshots": {}}
        for key, idx in snaps.items():
            hm["snapshots"][key] = {
                "t": int(idx),
                "v": np.round(V[idx], 4).tolist(),
                "z": np.round(z[idx], 3).tolist(),
                "v_min": float(np.round(V[idx].min(), 4)),
                "v_max": float(np.round(V[idx].max(), 4)),
            }
        with open(os.path.join(OUT, "heatmap", f"{it['pack_id']}.json"), "w",
                  encoding="utf-8") as fp:
            json.dump(hm, fp, ensure_ascii=False, separators=(",", ":"))

        # --- 이상 구간 ---
        for s, e in to_segments(pred, min_len=K_CONSEC):
            all_segments.append({
                "pack_id": it["pack_id"], "start": int(s), "end": int(e),
                "duration": int(e - s + 1),
                "max_phi": float(np.round(phi[s:e + 1].max(), 3)),
                "max_spe": float(np.round(SPE[s:e + 1].max(), 3)),
            })

        # --- 셀 단위 이상 순위 (상위 5개) ---
        cell_mean = z[pred == 1].mean(axis=0) if pred.sum() else z.mean(axis=0)
        top = np.argsort(cell_mean)[::-1][:5]

        packs.append({
            "pack_id": it["pack_id"],
            "pack_no": it["pack_no"],
            "mode": it["mode"],
            "source": it["source"],
            "grade": it["grade"],
            "has_label": label is not None,
            "n_rows": int(len(feat)),
            "dv_mv": round(dv_mv, 1),
            "temp_dev": round(temp_dev, 2),
            "thr_dv_mv": round(thr["dv"] * 1000, 1),
            "thr_temp": round(thr["temp"], 2),
            "rule_verdict": verdict,
            "cell_v_min": float(np.round(V[-1].min(), 4)),
            "cell_v_max": float(np.round(V[-1].max(), 4)),
            "anomaly_ratio": round(float(pred.mean()), 4),
            "max_phi": float(np.round(phi.max(), 3)),
            "ai_verdict": "NG" if pred.mean() >= NG_RATIO else "OK",
            "label_anomaly_ratio": (round(float(label.mean()), 4)
                                    if label is not None else None),
            "top_cells": [{"cell": cv[i], "score": float(np.round(cell_mean[i], 3))}
                          for i in top],
        })
        if n % 20 == 0:
            print(f"      {n}/{len(items)} 완료")

    # ---- 3단계: 마스터 파일 저장 ----
    print(f"\n[3/3] 저장 중... (처리 {len(packs)}개 / 제외 {len(skipped)}개)")

    with open(os.path.join(OUT, "packs.json"), "w", encoding="utf-8") as fp:
        json.dump(packs, fp, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "segments.json"), "w", encoding="utf-8") as fp:
        json.dump(all_segments, fp, ensure_ascii=False, indent=1)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_packs": len(packs),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "cell_ids": cell_cols,
        "model": {
            "name": "PCA (Hotelling T2 / SPE)",
            "n_components": int(pca.n_components_),
            "explained_variance": round(float(pca.explained_variance_ratio_.sum()), 4),
            "features": feats,
            "reference_packs": ref_ids,
            "reference_rows": int(len(ref)),
        },
        "thresholds": {
            "ucl_t2": round(ucl_t2, 3), "ucl_spe": round(ucl_spe, 4),
            "quantile": UCL_Q, "k_consecutive": K_CONSEC,
            "ng_ratio": NG_RATIO,
            "rule_3sigma": rule_thr,
        },
        "downsample": {"max_points": MAX_POINTS},
    }
    with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=1)

    # 모델 성능지표
    perf = os.path.join(C.OUT, "00_모델별_성능비교.csv")
    cvf = os.path.join(C.OUT, "00_모델별_교차검증.csv")
    models = []
    if os.path.exists(perf):
        d = pd.read_csv(perf)
        cvd = pd.read_csv(cvf) if os.path.exists(cvf) else None
        for m in d["모델"].unique():
            row = {"model": m}
            for _, r in d[d["모델"] == m].iterrows():
                key = "train" if r["구분"] == "학습" else "test"
                row[key] = {k.lower(): float(r[k]) for k in
                            ["Accuracy", "Precision", "Recall", "F1-score"]}
            if cvd is not None and m in set(cvd["모델"]):
                c = cvd[cvd["모델"] == m].iloc[0]
                row["cv"] = {k.lower(): float(c[k]) for k in
                             ["Accuracy", "Precision", "Recall", "F1-score"]}
            models.append(row)
    with open(os.path.join(OUT, "models.json"), "w", encoding="utf-8") as fp:
        json.dump(models, fp, ensure_ascii=False, indent=1)

    # ---- 요약 ----
    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(OUT) for f in fs)
    n_docs = len(packs) * 3 + 3
    print(f"\n{'='*62}")
    print(f"  문서 수      : 약 {n_docs}개  (무료 쓰기 한도 20,000/일)")
    print(f"  총 용량      : {size/1024/1024:.1f} MB  (무료 저장 한도 1 GiB)")
    print(f"  화면 1회 조회: 약 4 reads  (무료 읽기 한도 50,000/일)")
    print(f"  저장 위치    : {OUT}")
    print("=" * 62)
    if skipped:
        print("\n제외된 파일:")
        for s in skipped:
            print(f"  {s['pack_id']:16s} {s['reason']}")
    print("\n다음 단계:")
    print("  python src/models/extract_dashboard.py   # 온도 채널 + 4대 불량 유형")
    print("  python src/models/firebase_upload.py     # Firestore 업로드")
    return packs


if __name__ == "__main__":
    main()
