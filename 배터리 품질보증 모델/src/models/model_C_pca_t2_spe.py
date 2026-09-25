# -*- coding: utf-8 -*-
"""
모델 C : PCA 기반 이상탐지 (Hotelling T² · SPE)  - 다변량 공정관리(MSPC) 방식

정상 제품의 '변수 간 상관 구조'를 주성분으로 학습한다.
정상 팩에서는 셀 전압끼리, 모듈 온도끼리 일정한 관계를 유지하는데,
불량이 생기면 값 자체보다 이 관계가 먼저 깨진다.

    Hotelling T² : 정상 상관 구조 '안에서' 얼마나 멀리 갔는가
                   (주성분 공간 내부의 거리 - 정상 패턴을 따르되 정도가 심한 경우)
    SPE (Q 통계량) : 정상 상관 구조로 '설명되지 않는' 잔차의 크기
                   (주성분 공간 밖의 거리 - 관계 자체가 깨진 경우)

    두 지표는 서로 다른 이상을 잡으므로 OR로 결합한다.

관리한계(Control Limit)
    T²  : 정상 데이터 분위수 (F분포 근사 대신 경험적 분위수 사용)
    SPE : 정상 데이터 분위수

데이터
    학습 : data/preprocessed/train 10개 파일 (정상 56,805행)만 사용 - 라벨 미사용
    평가 : Test03, Test04, Test06, Test07, Test08

입력 변수
    셀 전압 176개와 모듈 온도 32개를 그대로 넣지 않고,
    셀 간 전압 불균형 / 모듈 간 온도 불균형 / 시간 변화로 요약한 9개 지표를 쓴다.
    원 208변수는 충전 진행도에 따라 값이 함께 움직여 상관이 거의 1에 가깝고,
    시험 환경(계절·설비)에 따라 절대 수준이 달라져 새 팩에 그대로 적용하기 어렵다.

실행
    python src/models/model_C_pca_t2_spe.py
"""

import os
import sys

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import common_data as C

MODEL_NAME = "PCA (T²·SPE)"
PREFIX = "C_pca_t2_spe"

VAR_RATIO = 0.95
QUANTILES = (0.99, 0.999, 0.9999)
KS = (1, 10, 30, 60)


def flag_with_rule(T2, SPE, t_thr, s_thr, k):
    raw = ((T2 > t_thr) | (SPE > s_thr)).astype(int)
    if k <= 1:
        return raw
    keep = pd.Series(raw).rolling(k, min_periods=k).min().fillna(0).values
    out = np.zeros_like(raw)
    for i in np.where(keep == 1)[0]:
        out[i - k + 1:i + 1] = 1
    return out


def main(df_all=None, cell_z=None):
    if df_all is None:
        df_all, _, cell_z = C.load_dataset()
    normal_df, train_df, test_df = C.get_splits(df_all)

    feats = [c for c in C.REL_FEATURES if c != "방전여부"]
    tag = "요약 Feature"
    print(f"\n[PCA] 입력: {tag} {len(feats)}개")

    # ---- 정상 데이터로 상관 구조 학습 ----
    scaler = StandardScaler().fit(normal_df[feats])
    pca = PCA(n_components=VAR_RATIO, random_state=C.RANDOM_STATE)
    pca.fit(scaler.transform(normal_df[feats]))
    print(f"[PCA] 정상 {len(normal_df):,}행으로 적합 → 주성분 {pca.n_components_}개 "
          f"/ 누적 설명분산 {pca.explained_variance_ratio_.sum():.4f}")
    print("      주성분별 설명분산:",
          np.round(pca.explained_variance_ratio_, 4).tolist())

    def scores(frame):
        Z = scaler.transform(frame[feats])
        P = pca.transform(Z)
        T2 = np.sum(P ** 2 / pca.explained_variance_, axis=1)   # 주성분 공간 내부
        SPE = np.sum((Z - pca.inverse_transform(P)) ** 2, axis=1)  # 주성분 공간 밖
        return T2, SPE

    T2_ok, SPE_ok = scores(normal_df)

    # ---- 관리한계 탐색 : 학습 데이터에서만 ----
    y_tr = train_df["label"].values
    tune, best = [], None
    for q in QUANTILES:
        t_thr, s_thr = np.quantile(T2_ok, q), np.quantile(SPE_ok, q)
        for k in KS:
            pred = np.zeros(len(train_df), dtype=int)
            files = train_df["파일"].values
            for f_name in pd.unique(files):
                m = files == f_name
                T2f, SPEf = scores(train_df.loc[m])
                pred[m] = flag_with_rule(T2f, SPEf, t_thr, s_thr, k)
            pr = precision_score(y_tr, pred, zero_division=0)
            rc = recall_score(y_tr, pred, zero_division=0)
            tune.append({"분위수": q, "연속 k": k, "T2 한계": t_thr, "SPE 한계": s_thr,
                         "Precision": pr, "Recall": rc,
                         "F1-score": f1_score(y_tr, pred, zero_division=0)})
            if pr >= 0.9 and (best is None or rc > best[0]):
                best = (rc, q, k, t_thr, s_thr)
    tune_df = pd.DataFrame(tune).round(4)
    C.save(tune_df, f"model_{PREFIX}_관리한계탐색.csv")
    if best is None:
        r = tune_df.loc[tune_df["F1-score"].idxmax()]
        best = (r["Recall"], r["분위수"], int(r["연속 k"]), r["T2 한계"], r["SPE 한계"])
    _, best_q, best_k, t_thr, s_thr = best
    print(f"[PCA] 관리한계: 정상 {best_q} 분위수, 연속 {best_k}시점 "
          f"→ T² > {t_thr:.3f} 또는 SPE > {s_thr:.3f}\n")

    def predict(frame):
        out = np.zeros(len(frame), dtype=int)
        T2s, SPEs = np.zeros(len(frame)), np.zeros(len(frame))
        files = frame["파일"].values
        for f_name in pd.unique(files):
            m = files == f_name
            T2f, SPEf = scores(frame.loc[m])
            out[m] = flag_with_rule(T2f, SPEf, t_thr, s_thr, best_k)
            T2s[m], SPEs[m] = T2f, SPEf
        return out, T2s, SPEs

    tr_pred, _, _ = predict(train_df)
    te_pred, T2_te, SPE_te = predict(test_df)

    results = [C.evaluate(MODEL_NAME, train_df["label"], tr_pred, "학습"),
               C.evaluate(MODEL_NAME, test_df["label"], te_pred, "테스트")]

    # T²만 / SPE만 썼을 때의 성능도 비교 (어느 쪽이 어떤 이상을 잡는지)
    y_te = test_df["label"]
    only = []
    for name, flag in [("T² 단독", (T2_te > t_thr).astype(int)),
                       ("SPE 단독", (SPE_te > s_thr).astype(int))]:
        only.append({"지표": name,
                     "Precision": precision_score(y_te, flag, zero_division=0),
                     "Recall": recall_score(y_te, flag, zero_division=0),
                     "F1-score": f1_score(y_te, flag, zero_division=0)})
    only_df = pd.DataFrame(only).round(4)
    print("\nT² / SPE 단독 성능 (연속 규칙 미적용, 테스트)")
    print(only_df.to_string(index=False))
    C.save(only_df, f"model_{PREFIX}_T2_SPE_단독.csv")

    C.save(pd.DataFrame(results).round(4), f"model_{PREFIX}_성능.csv")
    C.save(C.evaluate_per_file(MODEL_NAME, test_df, te_pred), f"model_{PREFIX}_파일별.csv")

    # ---- 파일 단위 교차검증 (Out-of-Fold 통합, 단일 값) ----
    # 비지도 모델이라 PCA 자체는 정상 데이터로만 적합되고, 라벨이 쓰이는 곳은
    # 관리한계 선택 단계다. fold마다 남은 시험에서 관리한계를 다시 정하고
    # 빠진 시험에 적용해, 그 절차가 새 시험에 통하는지 검증한다.
    lab = df_all[df_all["파일"].str.startswith("Test")].reset_index(drop=True)
    lab_T2, lab_SPE = np.zeros(len(lab)), np.zeros(len(lab))
    for f_name in lab["파일"].unique():
        m = (lab["파일"] == f_name).values
        lab_T2[m], lab_SPE[m] = scores(lab.loc[m])

    def fit_predict(tr_part, te_part):
        tr_idx, te_idx = tr_part.index.values, te_part.index.values
        best_local = None
        for q in QUANTILES:
            t_t, s_s = np.quantile(T2_ok, q), np.quantile(SPE_ok, q)
            for kk in KS:
                pr_pred = np.zeros(len(tr_part), dtype=int)
                for f_name in tr_part["파일"].unique():
                    mm = (tr_part["파일"] == f_name).values
                    pr_pred[mm] = flag_with_rule(lab_T2[tr_idx][mm], lab_SPE[tr_idx][mm],
                                                 t_t, s_s, kk)
                pr = precision_score(tr_part["label"], pr_pred, zero_division=0)
                rc = recall_score(tr_part["label"], pr_pred, zero_division=0)
                f1 = f1_score(tr_part["label"], pr_pred, zero_division=0)
                key = (rc if pr >= 0.9 else -1, f1)
                if best_local is None or key > best_local[0]:
                    best_local = (key, t_t, s_s, kk)
        _, t_t, s_s, kk = best_local
        pred = np.zeros(len(te_part), dtype=int)
        for f_name in te_part["파일"].unique():
            mm = (te_part["파일"] == f_name).values
            pred[mm] = flag_with_rule(lab_T2[te_idx][mm], lab_SPE[te_idx][mm],
                                      t_t, s_s, kk)
        return pred

    cv, fold_df, _ = C.oof_cross_validation(lab, fit_predict, model_name=MODEL_NAME)
    C.save(cv, f"model_{PREFIX}_교차검증.csv")
    C.save(fold_df, f"model_{PREFIX}_교차검증_fold별.csv")

    out = test_df[["파일", "충방전", "판정", "시점", "label"]].copy()
    out["예측"] = te_pred
    out["Hotelling_T2"] = np.round(T2_te, 4)
    out["SPE"] = np.round(SPE_te, 4)
    C.save(out, f"model_{PREFIX}_이상점수.csv")

    # ---- 이상 셀 순위 (셀 전압 Heatmap 입력) ----
    if cell_z:
        ranks = []
        for prefix, (z, label, cv_cols) in cell_z.items():
            if label.sum() == 0:
                continue
            sc = z[label == 1].mean(axis=0)
            for rank, idx in enumerate(np.argsort(sc)[::-1][:5], 1):
                ranks.append({"파일": prefix, "순위": rank, "셀": cv_cols[idx],
                              "이상점수": round(float(sc[idx]), 3)})
        C.save(pd.DataFrame(ranks), f"model_{PREFIX}_이상셀순위.csv")

    return results


if __name__ == "__main__":
    main()
