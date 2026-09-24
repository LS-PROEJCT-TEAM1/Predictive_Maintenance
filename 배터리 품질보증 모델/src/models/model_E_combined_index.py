# -*- coding: utf-8 -*-
"""
모델 E : Hotelling T² + SPE(Q) 결합 지표  (MSPC Combined Index)

모델 C는 T²와 SPE에 각각 관리한계를 두고 "둘 중 하나라도 넘으면 이상"으로
판정한다(OR 결합). 이 방식은 두 지표가 각각 한계를 살짝 밑돌 때
— 즉 어느 쪽도 단독으로는 이상이 아니지만 둘 다 평소보다 높을 때 —
이상을 놓친다.

결합 지표는 두 통계량을 각자의 관리한계로 나눠 하나의 값으로 합친다.

    φ = T² / UCL(T²)  +  SPE / UCL(SPE)

    각 항이 1이면 딱 관리한계 수준이라는 뜻이므로, 서로 다른 단위를 가진
    두 통계량을 같은 척도에서 더할 수 있다. 둘 다 0.8 수준이면
    OR 방식은 정상으로 보지만 결합 지표는 1.6이 되어 이상으로 잡는다.

    (Raich & Cinar 1996 의 combined index 와 같은 형태)

데이터
    학습 : data/preprocessed/train 10개 파일 (정상 56,805행)만 사용 - 라벨 미사용
    평가 : Test03, Test04, Test06, Test07, Test08

비교
    같은 PCA 모델에서 나온 T²·SPE를 쓰므로, 모델 C와의 차이는
    '두 지표를 어떻게 합치는가' 하나뿐이다. 결과를 직접 비교할 수 있다.

실행
    python src/models/model_E_combined_index.py
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

MODEL_NAME = "T²+SPE 결합지표"
PREFIX = "E_combined_index"

VAR_RATIO = 0.95
UCL_Q = 0.99                      # 각 통계량의 관리한계 분위수
QUANTILES = (0.99, 0.999, 0.9999)  # 결합 지표 자체의 임계값 분위수
KS = (1, 10, 30, 60)


def flag_with_rule(score, thr, k):
    """임계값 초과가 연속 k시점 이상 지속될 때만 이상으로 확정."""
    raw = (score > thr).astype(int)
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

    # ---- 정상 데이터로 상관 구조 학습 (모델 C와 동일) ----
    scaler = StandardScaler().fit(normal_df[feats])
    pca = PCA(n_components=VAR_RATIO, random_state=C.RANDOM_STATE)
    pca.fit(scaler.transform(normal_df[feats]))
    print(f"\n[결합지표] 정상 {len(normal_df):,}행으로 적합 → "
          f"주성분 {pca.n_components_}개 / "
          f"누적 설명분산 {pca.explained_variance_ratio_.sum():.4f}")

    def t2_spe(frame):
        Z = scaler.transform(frame[feats])
        P = pca.transform(Z)
        T2 = np.sum(P ** 2 / pca.explained_variance_, axis=1)
        SPE = np.sum((Z - pca.inverse_transform(P)) ** 2, axis=1)
        return T2, SPE

    T2_ok, SPE_ok = t2_spe(normal_df)

    # ---- 각 통계량의 관리한계 (정규화 기준) ----
    ucl_t2 = np.quantile(T2_ok, UCL_Q)
    ucl_spe = np.quantile(SPE_ok, UCL_Q)
    print(f"[결합지표] 관리한계  UCL(T²) = {ucl_t2:.4f}  "
          f"UCL(SPE) = {ucl_spe:.4f}  (정상 {UCL_Q} 분위수)")

    def combined(frame):
        """φ = T²/UCL(T²) + SPE/UCL(SPE). 두 항 모두 1이면 관리한계 수준."""
        T2, SPE = t2_spe(frame)
        return T2 / ucl_t2 + SPE / ucl_spe, T2, SPE

    phi_ok, _, _ = combined(normal_df)
    print(f"[결합지표] 정상 데이터 φ 평균 {phi_ok.mean():.3f} / "
          f"99% {np.quantile(phi_ok, 0.99):.3f} / 최대 {phi_ok.max():.3f}\n")

    # ---- 임계값·연속길이 탐색 : 학습 데이터에서만 ----
    y_tr = train_df["label"].values
    phi_tr, _, _ = combined(train_df)
    tune, best = [], None
    for q in QUANTILES:
        thr = np.quantile(phi_ok, q)
        for k in KS:
            pred = np.zeros(len(train_df), dtype=int)
            files = train_df["파일"].values
            for f_name in pd.unique(files):
                m = files == f_name
                pred[m] = flag_with_rule(phi_tr[m], thr, k)
            pr = precision_score(y_tr, pred, zero_division=0)
            rc = recall_score(y_tr, pred, zero_division=0)
            tune.append({"분위수": q, "연속 k": k, "φ 임계값": thr,
                         "Precision": pr, "Recall": rc,
                         "F1-score": f1_score(y_tr, pred, zero_division=0)})
            if pr >= 0.9 and (best is None or rc > best[0]):
                best = (rc, q, k, thr)
    tune_df = pd.DataFrame(tune).round(4)
    C.save(tune_df, f"model_{PREFIX}_임계값탐색.csv")
    if best is None:
        r = tune_df.loc[tune_df["F1-score"].idxmax()]
        best = (r["Recall"], r["분위수"], int(r["연속 k"]), r["φ 임계값"])
    _, best_q, best_k, thr = best
    print(f"[결합지표] 선택: 정상 {best_q} 분위수, 연속 {best_k}시점 → φ > {thr:.4f}\n")

    def predict(frame):
        phi, T2, SPE = combined(frame)
        out = np.zeros(len(frame), dtype=int)
        files = frame["파일"].values
        for f_name in pd.unique(files):
            m = files == f_name
            out[m] = flag_with_rule(phi[m], thr, best_k)
        return out, phi, T2, SPE

    tr_pred, _, _, _ = predict(train_df)
    te_pred, phi_te, T2_te, SPE_te = predict(test_df)

    results = [C.evaluate(MODEL_NAME, train_df["label"], tr_pred, "학습"),
               C.evaluate(MODEL_NAME, test_df["label"], te_pred, "테스트")]

    C.save(pd.DataFrame(results).round(4), f"model_{PREFIX}_성능.csv")
    C.save(C.evaluate_per_file(MODEL_NAME, test_df, te_pred), f"model_{PREFIX}_파일별.csv")

    # ---- OR 결합(모델 C)과 직접 비교 ----
    # 같은 PCA·같은 관리한계를 쓰되, 합치는 방식만 바꿔 성능 차이를 본다.
    y_te = test_df["label"]
    or_pred = np.zeros(len(test_df), dtype=int)
    files = test_df["파일"].values
    or_raw = ((T2_te > ucl_t2) | (SPE_te > ucl_spe)).astype(int)
    for f_name in pd.unique(files):
        m = files == f_name
        or_pred[m] = flag_with_rule(or_raw[m].astype(float), 0.5, best_k)
    cmp_rows = []
    for name, p_ in [("OR 결합 (T² 또는 SPE)", or_pred),
                     ("결합 지표 φ = T²/UCL + SPE/UCL", te_pred)]:
        cmp_rows.append({"결합 방식": name,
                         "Precision": precision_score(y_te, p_, zero_division=0),
                         "Recall": recall_score(y_te, p_, zero_division=0),
                         "F1-score": f1_score(y_te, p_, zero_division=0),
                         "예측 이상": int(p_.sum())})
    cmp_df = pd.DataFrame(cmp_rows).round(4)
    print("\n결합 방식 비교 (같은 PCA·같은 관리한계, 연속 규칙 동일)")
    print(cmp_df.to_string(index=False))
    C.save(cmp_df, f"model_{PREFIX}_결합방식비교.csv")

    # ---- 기여도 분해 : 이상 시점에서 T²와 SPE 중 어느 쪽이 컸는가 ----
    mask = te_pred == 1
    if mask.sum():
        t2_part = (T2_te[mask] / ucl_t2)
        spe_part = (SPE_te[mask] / ucl_spe)
        contrib = pd.DataFrame([{
            "이상 판정 시점": int(mask.sum()),
            "T² 항 평균": round(float(t2_part.mean()), 3),
            "SPE 항 평균": round(float(spe_part.mean()), 3),
            "T²가 더 큰 시점 비율": round(float((t2_part > spe_part).mean()), 3),
        }])
        print("\n이상 판정 시점의 기여도 분해")
        print(contrib.to_string(index=False))
        C.save(contrib, f"model_{PREFIX}_기여도분해.csv")

    # ---- 시점별 점수 저장 (Dash 입력) ----
    out = test_df[["파일", "충방전", "판정", "시점", "label"]].copy()
    out["예측"] = te_pred
    out["결합지표_phi"] = np.round(phi_te, 4)
    out["Hotelling_T2"] = np.round(T2_te, 4)
    out["SPE"] = np.round(SPE_te, 4)
    out["T2_정규화"] = np.round(T2_te / ucl_t2, 4)
    out["SPE_정규화"] = np.round(SPE_te / ucl_spe, 4)
    C.save(out, f"model_{PREFIX}_이상점수.csv")

    # ---- 파일 단위 교차검증 (Out-of-Fold 통합, 단일 값) ----
    lab = df_all[df_all["파일"].str.startswith("Test")].reset_index(drop=True)
    lab_phi = np.zeros(len(lab))
    for f_name in lab["파일"].unique():
        m = (lab["파일"] == f_name).values
        lab_phi[m], _, _ = combined(lab.loc[m])

    def fit_predict(tr_part, te_part):
        tr_idx, te_idx = tr_part.index.values, te_part.index.values
        best_local = None
        for q in QUANTILES:
            t = np.quantile(phi_ok, q)
            for kk in KS:
                pr_pred = np.zeros(len(tr_part), dtype=int)
                for f_name in tr_part["파일"].unique():
                    mm = (tr_part["파일"] == f_name).values
                    pr_pred[mm] = flag_with_rule(lab_phi[tr_idx][mm], t, kk)
                pr = precision_score(tr_part["label"], pr_pred, zero_division=0)
                rc = recall_score(tr_part["label"], pr_pred, zero_division=0)
                f1 = f1_score(tr_part["label"], pr_pred, zero_division=0)
                key = (rc if pr >= 0.9 else -1, f1)
                if best_local is None or key > best_local[0]:
                    best_local = (key, t, kk)
        _, t, kk = best_local
        pred = np.zeros(len(te_part), dtype=int)
        for f_name in te_part["파일"].unique():
            mm = (te_part["파일"] == f_name).values
            pred[mm] = flag_with_rule(lab_phi[te_idx][mm], t, kk)
        return pred

    cv, fold_df, _ = C.oof_cross_validation(lab, fit_predict, model_name=MODEL_NAME)
    C.save(cv, f"model_{PREFIX}_교차검증.csv")
    C.save(fold_df, f"model_{PREFIX}_교차검증_fold별.csv")

    return results


if __name__ == "__main__":
    main()
