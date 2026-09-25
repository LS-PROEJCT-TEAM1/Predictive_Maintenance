# -*- coding: utf-8 -*-
"""
모델 2 : Random Forest (랜덤포레스트)  - 교안 Part 5 Ch40

정상 기준 데이터: preprocessed/train 10개 파일 (충전 5 + 방전 5)
학습: 정상 10개 + NG 2개(Test05, Test09)
평가: Test03, Test04, Test06, Test07, Test08

실행
    python src/models/model_02_random_forest.py
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
N_ESTIMATORS = 100      # 교안 코드 40-5
MAX_DEPTH = 5


def main():
    df_all, report, _ = C.load_dataset()
    normal_df, train_df, test_df = C.get_splits(df_all)

    feats = C.REL_FEATURES
    X_train, y_train = train_df[feats], train_df["label"]
    X_test, y_test = test_df[feats], test_df["label"]

    print(f"학습 {X_train.shape} (정상 {int((y_train==0).sum()):,} / 이상 {int((y_train==1).sum()):,})")
    print(f"테스트 {X_test.shape}\n")

    # ---- 학습 (교안 코드 40-5) ----
    # 정상 데이터가 이상보다 많아 class_weight로 불균형을 보정한다(표 40-5).
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                                   class_weight="balanced",
                                   random_state=C.RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)

    results = [C.evaluate(MODEL_NAME, y_train, model.predict(X_train), "학습"),
               C.evaluate(MODEL_NAME, y_test, model.predict(X_test), "테스트")]
    pred = model.predict(X_test)
    ex = (test_df["파일"] != "Test07_NG_dchg").values
    results.append(C.evaluate(MODEL_NAME, y_test[ex], pred[ex], "테스트(Test07 제외)"))

    C.save(pd.DataFrame(results).round(4), "model_02_random_forest_성능.csv")
    C.save(C.evaluate_per_file(MODEL_NAME, test_df, pred),
           "model_02_random_forest_파일별.csv")

    imp = (pd.DataFrame({"Feature": feats, "중요도": model.feature_importances_})
           .sort_values("중요도", ascending=False).round(4))
    print("\n변수 중요도")
    print(imp.head(6).to_string(index=False))
    C.save(imp, "model_02_random_forest_변수중요도.csv")

    # ---- 파일 단위 교차검증 (성능이 얼마나 흔들리는지) ----
    print("\n파일 단위 GroupKFold 교차검증 (5-fold)")
    labelled = df_all[df_all["파일"].str.startswith("Test")]
    gkf = GroupKFold(n_splits=5)
    sc = []
    for tr, te in gkf.split(labelled[feats], labelled["label"], groups=labelled["파일"]):
        m = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                                   class_weight="balanced",
                                   random_state=C.RANDOM_STATE, n_jobs=-1)
        m.fit(labelled[feats].iloc[tr], labelled["label"].iloc[tr])
        p = m.predict(labelled[feats].iloc[te])
        yt = labelled["label"].iloc[te]
        sc.append([accuracy_score(yt, p), precision_score(yt, p, zero_division=0),
                   recall_score(yt, p, zero_division=0), f1_score(yt, p, zero_division=0)])
    s = np.array(sc)
    cv = pd.DataFrame([{"Accuracy": f"{s[:,0].mean():.4f}±{s[:,0].std():.4f}",
                        "Precision": f"{s[:,1].mean():.4f}±{s[:,1].std():.4f}",
                        "Recall": f"{s[:,2].mean():.4f}±{s[:,2].std():.4f}",
                        "F1-score": f"{s[:,3].mean():.4f}±{s[:,3].std():.4f}"}])
    print(cv.to_string(index=False))
    C.save(cv, "model_02_random_forest_교차검증.csv")

    return results


if __name__ == "__main__":
    main()
