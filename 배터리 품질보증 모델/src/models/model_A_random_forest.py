# -*- coding: utf-8 -*-
"""
모델 A : Random Forest  (지도학습 / 기준점 Baseline)

라벨(정상·이상)을 직접 학습하는 방식. 나머지 세 모델(Isolation Forest,
PCA, LSTM Autoencoder)은 모두 라벨 없이 '정상'만 학습하는 비지도 방식이므로,
이 모델은 "라벨이 있을 때 어디까지 되는가"를 보여주는 기준점 역할을 한다.

데이터
    정상 기준 : data/preprocessed/train 10개 파일 (충전 5 + 방전 5) 56,805행
    학습      : 정상 10개 + NG 2개(Test05, Test09)
    평가      : Test03, Test04, Test06, Test07, Test08

교안 연계 : Part 5 Ch40 코드 40-5 (RandomForestClassifier)

실행
    python src/models/model_A_random_forest.py
"""

import os
import sys

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import common_data as C

MODEL_NAME = "Random Forest"
PREFIX = "A_random_forest"

N_ESTIMATORS = 100
MAX_DEPTH = 5


def main(df_all=None, cell_z=None):
    if df_all is None:
        df_all, _, cell_z = C.load_dataset()
    normal_df, train_df, test_df = C.get_splits(df_all)

    # 절대 전압·온도 수준을 뺀 상대 Feature
    # (시험마다 온도 조건이 달라 절대값을 넣으면 '시험 종류'를 외운다)
    feats = C.REL_FEATURES
    X_train, y_train = train_df[feats], train_df["label"]
    X_test, y_test = test_df[feats], test_df["label"]

    print(f"\n학습 {X_train.shape} (정상 {int((y_train==0).sum()):,} / 이상 {int((y_train==1).sum()):,})")
    print(f"평가 {X_test.shape} (정상 {int((y_test==0).sum()):,} / 이상 {int((y_test==1).sum()):,})\n")

    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                                   class_weight="balanced",
                                   random_state=C.RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]      # 이상 확률 = 이상 점수

    results = [C.evaluate(MODEL_NAME, y_train, model.predict(X_train), "학습"),
               C.evaluate(MODEL_NAME, y_test, pred, "테스트")]

    C.save(pd.DataFrame(results).round(4), f"model_{PREFIX}_성능.csv")
    C.save(C.evaluate_per_file(MODEL_NAME, test_df, pred), f"model_{PREFIX}_파일별.csv")

    imp = (pd.DataFrame({"Feature": feats, "중요도": model.feature_importances_})
           .sort_values("중요도", ascending=False).round(4))
    print("\n변수 중요도")
    print(imp.head(6).to_string(index=False))
    C.save(imp, f"model_{PREFIX}_변수중요도.csv")

    out = test_df[["파일", "충방전", "판정", "시점", "label"]].copy()
    out["예측"] = pred
    out["이상확률"] = np.round(proba, 4)
    C.save(out, f"model_{PREFIX}_이상점수.csv")

    # ---- 파일 단위 교차검증 (Out-of-Fold 통합, 단일 값) ----
    lab = df_all[df_all["파일"].str.startswith("Test")].reset_index(drop=True)

    def fit_predict(tr_part, te_part):
        m = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                                   class_weight="balanced",
                                   random_state=C.RANDOM_STATE, n_jobs=-1)
        m.fit(tr_part[feats], tr_part["label"])
        return m.predict(te_part[feats])

    cv, fold_df, _ = C.oof_cross_validation(lab, fit_predict, model_name=MODEL_NAME)
    C.save(cv, f"model_{PREFIX}_교차검증.csv")
    C.save(fold_df, f"model_{PREFIX}_교차검증_fold별.csv")

    return results


if __name__ == "__main__":
    main()
