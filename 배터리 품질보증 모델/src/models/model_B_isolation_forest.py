# -*- coding: utf-8 -*-
"""
모델 B : Isolation Forest  (비지도 이상탐지)

Random Forest와 같은 트리 기반이지만 라벨을 쓰지 않는다.
무작위로 변수와 분할점을 골라 데이터를 계속 쪼갤 때,
이상치는 주변에 비슷한 값이 없어 몇 번 쪼개지 않아도 혼자 남는다(고립된다).
그 '고립되기까지 필요한 분할 횟수'가 짧을수록 이상 점수가 높다.

    정상 데이터 → 다른 값들 사이에 묻혀 있어 고립에 많은 분할이 필요
    이상 데이터 → 몇 번만 쪼개도 혼자 남음

데이터
    학습 : data/preprocessed/train 10개 파일 (정상 56,805행)만 사용 - 라벨 미사용
    평가 : Test03, Test04, Test06, Test07, Test08

임계값
    sklearn 기본(decision_function < 0)과
    정상 데이터 점수의 분위수 기준을 함께 비교해, 학습 데이터에서 고른다.

실행
    python src/models/model_B_isolation_forest.py
"""

import os
import sys

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import common_data as C

MODEL_NAME = "Isolation Forest"
PREFIX = "B_isolation_forest"

N_ESTIMATORS = 200
MAX_SAMPLES = 256          # 논문 권장값. 작을수록 이상치가 더 잘 고립된다.
QUANTILES = (0.99, 0.999, 0.9999)
KS = (1, 10, 30)           # 연속 초과 시점 수 (단발 잡음 억제)


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


def predict_by_file(frame, score_fn, thr, k):
    """파일(시험)별로 시간 순서가 이어지므로 파일 단위로 규칙을 적용한다."""
    out = np.zeros(len(frame), dtype=int)
    sc = np.zeros(len(frame))
    files = frame["파일"].values
    for f_name in pd.unique(files):
        m = files == f_name
        s = score_fn(frame.loc[m])
        sc[m] = s
        out[m] = flag_with_rule(s, thr, k)
    return out, sc


def main(df_all=None, cell_z=None):
    if df_all is None:
        df_all, _, cell_z = C.load_dataset()
    normal_df, train_df, test_df = C.get_splits(df_all)

    feats = [c for c in C.REL_FEATURES if c != "방전여부"]

    # ---- 정상 데이터로만 학습 (라벨 미사용) ----
    scaler = StandardScaler().fit(normal_df[feats])
    iso = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
                          contamination="auto", random_state=C.RANDOM_STATE,
                          n_jobs=-1)
    iso.fit(scaler.transform(normal_df[feats]))
    print(f"\n[Isolation Forest] 정상 {len(normal_df):,}행으로 학습 "
          f"(트리 {N_ESTIMATORS}개, 표본 {MAX_SAMPLES})")

    def score(frame):
        # score_samples는 정상일수록 큰 값 → 부호를 뒤집어 '이상 점수'로 쓴다
        return -iso.score_samples(scaler.transform(frame[feats]))

    s_ok = score(normal_df)

    # ---- 임계값·연속길이 탐색 : 학습 데이터에서만 ----
    y_tr = train_df["label"].values
    tune, best = [], None

    # (1) sklearn 기본 기준 : decision_function < 0
    base_thr = -iso.offset_ * -1.0      # score_samples 기준으로 환산한 기본 임계값
    base_thr = -iso.offset_
    for k in KS:
        pred, _ = predict_by_file(train_df, score, base_thr, k)
        pr = precision_score(y_tr, pred, zero_division=0)
        rc = recall_score(y_tr, pred, zero_division=0)
        tune.append({"기준": "sklearn 기본", "분위수": "-", "연속 k": k, "임계값": base_thr,
                     "Precision": pr, "Recall": rc,
                     "F1-score": f1_score(y_tr, pred, zero_division=0)})
        if pr >= 0.9 and (best is None or rc > best[0]):
            best = (rc, base_thr, k, "sklearn 기본")

    # (2) 정상 데이터 점수의 분위수 기준
    for q in QUANTILES:
        thr = np.quantile(s_ok, q)
        for k in KS:
            pred, _ = predict_by_file(train_df, score, thr, k)
            pr = precision_score(y_tr, pred, zero_division=0)
            rc = recall_score(y_tr, pred, zero_division=0)
            tune.append({"기준": "정상 분위수", "분위수": q, "연속 k": k, "임계값": thr,
                         "Precision": pr, "Recall": rc,
                         "F1-score": f1_score(y_tr, pred, zero_division=0)})
            if pr >= 0.9 and (best is None or rc > best[0]):
                best = (rc, thr, k, f"정상 {q} 분위수")

    tune_df = pd.DataFrame(tune).round(4)
    C.save(tune_df, f"model_{PREFIX}_임계값탐색.csv")
    if best is None:
        r = tune_df.loc[tune_df["F1-score"].idxmax()]
        best = (r["Recall"], r["임계값"], int(r["연속 k"]), str(r["기준"]))
    _, thr, k, thr_name = best
    print(f"[Isolation Forest] 선택: {thr_name}, 연속 {k}시점 → 점수 > {thr:.4f}\n")

    tr_pred, _ = predict_by_file(train_df, score, thr, k)
    te_pred, te_score = predict_by_file(test_df, score, thr, k)

    results = [C.evaluate(MODEL_NAME, train_df["label"], tr_pred, "학습"),
               C.evaluate(MODEL_NAME, test_df["label"], te_pred, "테스트")]

    C.save(pd.DataFrame(results).round(4), f"model_{PREFIX}_성능.csv")
    C.save(C.evaluate_per_file(MODEL_NAME, test_df, te_pred), f"model_{PREFIX}_파일별.csv")

    # ---- 파일 단위 교차검증 (Out-of-Fold 통합, 단일 값) ----
    # 비지도 모델이라 학습되는 것은 '정상 기준' 뿐이고, 라벨이 쓰이는 곳은
    # 임계값 선택 단계다. 따라서 교차검증도 그 절차를 검증한다.
    # fold마다 남은 시험 파일에서 임계값을 다시 고르고, 빠진 파일에 적용한다.
    lab = df_all[df_all["파일"].str.startswith("Test")].reset_index(drop=True)
    lab_score = np.zeros(len(lab))
    for f_name in lab["파일"].unique():
        m = (lab["파일"] == f_name).values
        lab_score[m] = score(lab.loc[m])

    def fit_predict(tr_part, te_part):
        tr_idx, te_idx = tr_part.index.values, te_part.index.values
        best_local = None
        for q in QUANTILES:
            t = np.quantile(s_ok, q)
            for kk in KS:
                pr_pred = np.zeros(len(tr_part), dtype=int)
                for f_name in tr_part["파일"].unique():
                    mm = (tr_part["파일"] == f_name).values
                    pr_pred[mm] = flag_with_rule(lab_score[tr_idx][mm], t, kk)
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
            pred[mm] = flag_with_rule(lab_score[te_idx][mm], t, kk)
        return pred

    cv, fold_df, _ = C.oof_cross_validation(lab, fit_predict, model_name=MODEL_NAME)
    C.save(cv, f"model_{PREFIX}_교차검증.csv")
    C.save(fold_df, f"model_{PREFIX}_교차검증_fold별.csv")

    out = test_df[["파일", "충방전", "판정", "시점", "label"]].copy()
    out["예측"] = te_pred
    out["이상점수"] = np.round(te_score, 4)
    C.save(out, f"model_{PREFIX}_이상점수.csv")

    return results


if __name__ == "__main__":
    main()
