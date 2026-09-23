# -*- coding: utf-8 -*-
"""
모델 3 : PCA 기반 이상탐지 (비지도)  - 기획서 기술스택

정상 기준 데이터: preprocessed/train 10개 파일 (충전 5 + 방전 5) 56,805행
    → '정상 팩이 어떤 모습인가'만 학습한다. NG 라벨을 쓰지 않는다.
평가: Test03, Test04, Test06, Test07, Test08

이상 점수
    Q 통계량 (재구성 오차) : 주성분으로 복원되지 않는 정도
    Hotelling T2          : 정상 범위에서 벗어난 정도
    두 값 중 하나라도 임계값을 넘고, 그 상태가 연속 K시점 이상 지속되면 이상

실행
    python src/models/model_03_pca.py
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

MODEL_NAME = "PCA 이상탐지"
VAR_RATIO = 0.95        # 누적 설명분산 95%까지 주성분 사용
QUANTILES = (0.99, 0.999, 0.9999)
KS = (1, 10, 30, 60)    # 연속 초과 시점 수 후보


def flag_with_rule(Q, T2, q_thr, t_thr, k):
    """임계값 초과가 연속 k시점 이상 지속될 때만 이상으로 확정.
    단발성 잡음 1~2시점을 이상으로 보지 않는다 (교안 Ch46 : 단순 변동 vs 이상치)."""
    raw = ((Q > q_thr) | (T2 > t_thr)).astype(int)
    if k <= 1:
        return raw
    keep = pd.Series(raw).rolling(k, min_periods=k).min().fillna(0).values
    out = np.zeros_like(raw)
    for i in np.where(keep == 1)[0]:
        out[i - k + 1:i + 1] = 1
    return out


def main():
    df_all, report, cell_z = C.load_dataset()
    normal_df, train_df, test_df = C.get_splits(df_all)

    # 방전여부는 조건 변수라 이상 점수 계산에서 제외
    feats = [c for c in C.REL_FEATURES if c != "방전여부"]

    # ---- 정상 데이터로만 적합 ----
    scaler = StandardScaler().fit(normal_df[feats])
    pca = PCA(n_components=VAR_RATIO, random_state=C.RANDOM_STATE).fit(
        scaler.transform(normal_df[feats]))
    print(f"\n[PCA] 정상 기준 {len(normal_df):,}행으로 적합")
    print(f"      주성분 {pca.n_components_}개 / "
          f"누적 설명분산 {pca.explained_variance_ratio_.sum():.4f}")

    def scores(X):
        Z = scaler.transform(X)
        P = pca.transform(Z)
        Q = np.sum((Z - pca.inverse_transform(P)) ** 2, axis=1)
        T2 = np.sum(P ** 2 / pca.explained_variance_, axis=1)
        return Q, T2

    Q_ok, T2_ok = scores(normal_df[feats])

    # ---- 임계값·연속길이 탐색 : 학습 데이터에서만 ----
    # 기획서 목표가 Precision 0.7 이상이므로,
    # 학습에서 Precision 0.9 이상을 만족하는 설정 중 Recall이 가장 높은 조합 선택
    y_tr = train_df["label"].values
    tune, best = [], None
    for q in QUANTILES:
        qt, tt = np.quantile(Q_ok, q), np.quantile(T2_ok, q)
        for k in KS:
            pred = np.zeros(len(train_df), dtype=int)
            files = train_df["파일"].values
            for f_name in pd.unique(files):
                m = files == f_name
                Qf, T2f = scores(train_df.loc[m, feats])
                pred[m] = flag_with_rule(Qf, T2f, qt, tt, k)
            pr = precision_score(y_tr, pred, zero_division=0)
            rc = recall_score(y_tr, pred, zero_division=0)
            tune.append({"분위수": q, "연속 k": k, "Precision": pr, "Recall": rc,
                         "F1-score": f1_score(y_tr, pred, zero_division=0)})
            if pr >= 0.9 and (best is None or rc > best[0]):
                best = (rc, q, k, qt, tt)
    if best is None:
        r = max(tune, key=lambda d: d["F1-score"])
        best = (r["Recall"], r["분위수"], r["연속 k"],
                np.quantile(Q_ok, r["분위수"]), np.quantile(T2_ok, r["분위수"]))
    _, best_q, best_k, q_thr, t_thr = best
    C.save(pd.DataFrame(tune).round(4), "model_03_pca_임계값탐색.csv")
    print(f"[PCA] 선택: 분위수 {best_q}, 연속 {best_k}시점 "
          f"→ Q>{q_thr:.3f} 또는 T2>{t_thr:.3f}\n")

    def predict(frame):
        out = np.zeros(len(frame), dtype=int)
        Qs = np.zeros(len(frame))
        Ts = np.zeros(len(frame))
        files = frame["파일"].values
        for f_name in pd.unique(files):
            m = files == f_name
            Qf, T2f = scores(frame.loc[m, feats])
            out[m] = flag_with_rule(Qf, T2f, q_thr, t_thr, best_k)
            Qs[m], Ts[m] = Qf, T2f
        return out, Qs, Ts

    tr_pred, _, _ = predict(train_df)
    te_pred, Q_te, T2_te = predict(test_df)

    results = [C.evaluate(MODEL_NAME, train_df["label"], tr_pred, "학습"),
               C.evaluate(MODEL_NAME, test_df["label"], te_pred, "테스트")]
    ex = (test_df["파일"] != "Test07_NG_dchg").values
    results.append(C.evaluate(MODEL_NAME, test_df["label"].values[ex], te_pred[ex],
                              "테스트(Test07 제외)"))

    C.save(pd.DataFrame(results).round(4), "model_03_pca_성능.csv")
    C.save(C.evaluate_per_file(MODEL_NAME, test_df, te_pred), "model_03_pca_파일별.csv")

    # ---- 시점별 이상 점수 저장 (Dash 화면 입력) ----
    out = test_df[["파일", "충방전", "판정", "시점", "label"]].copy()
    out["PCA_예측"] = te_pred
    out["이상점수_Q"] = np.round(Q_te, 4)
    out["이상점수_T2"] = np.round(T2_te, 4)
    C.save(out, "model_03_pca_이상점수.csv")

    # ---- 이상 셀 순위 (셀 전압 Heatmap용) ----
    ranks = []
    for prefix, (z, label, cv_cols) in cell_z.items():
        if label.sum() == 0:
            continue
        score = z[label == 1].mean(axis=0)
        for rank, idx in enumerate(np.argsort(score)[::-1][:5], 1):
            ranks.append({"파일": prefix, "순위": rank, "셀": cv_cols[idx],
                          "이상점수": round(float(score[idx]), 3)})
    cell_rank = pd.DataFrame(ranks)
    print("\n이상 점수 상위 셀")
    print(cell_rank[cell_rank["순위"] <= 2].to_string(index=False))
    C.save(cell_rank, "model_03_pca_이상셀순위.csv")

    return results


if __name__ == "__main__":
    main()
