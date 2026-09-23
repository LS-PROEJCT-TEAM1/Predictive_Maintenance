# -*- coding: utf-8 -*-
"""
모델 1 : Decision Tree (의사결정나무)  - 교안 Part 5 Ch40

정상 기준 데이터: preprocessed/train 10개 파일 (충전 5 + 방전 5)
학습: 정상 10개 + NG 2개(Test05, Test09)
평가: Test03, Test04, Test06, Test07, Test08

실행
    python src/models/model_01_decision_tree.py
"""

import os
import sys

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import common_data as C

MODEL_NAME = "Decision Tree"
MAX_DEPTH = 4           # 교안 코드 40-4 : 너무 깊으면 과적합


def main():
    df_all, report, _ = C.load_dataset()
    normal_df, train_df, test_df = C.get_splits(df_all)

    # 절대 전압·온도 수준을 뺀 상대 Feature 사용
    # (시험마다 온도 조건이 달라 절대값을 쓰면 '시험 종류'를 외운다)
    feats = C.REL_FEATURES
    X_train, y_train = train_df[feats], train_df["label"]
    X_test, y_test = test_df[feats], test_df["label"]

    print(f"학습 {X_train.shape} (정상 {int((y_train==0).sum()):,} / 이상 {int((y_train==1).sum()):,})")
    print(f"테스트 {X_test.shape} (정상 {int((y_test==0).sum()):,} / 이상 {int((y_test==1).sum()):,})\n")

    # ---- 학습 (교안 코드 40-4) ----
    model = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=C.RANDOM_STATE)
    model.fit(X_train, y_train)

    # ---- 평가 (교안 Ch39 : 학습 성능과 테스트 성능 비교) ----
    results = [C.evaluate(MODEL_NAME, y_train, model.predict(X_train), "학습"),
               C.evaluate(MODEL_NAME, y_test, model.predict(X_test), "테스트")]
    pred = model.predict(X_test)

    # Test07은 NG 팩이지만 라벨이 마지막 227시점에만 있어 별도로도 본다
    ex = (test_df["파일"] != "Test07_NG_dchg").values
    results.append(C.evaluate(MODEL_NAME, y_test[ex], pred[ex], "테스트(Test07 제외)"))

    C.save(pd.DataFrame(results).round(4), "model_01_decision_tree_성능.csv")
    C.save(C.evaluate_per_file(MODEL_NAME, test_df, pred),
           "model_01_decision_tree_파일별.csv")

    # ---- 변수 중요도 + 규칙 (교안 Ch40 : 판단 구조가 직관적이다) ----
    imp = (pd.DataFrame({"Feature": feats, "중요도": model.feature_importances_})
           .sort_values("중요도", ascending=False).round(4))
    print("\n변수 중요도")
    print(imp.head(6).to_string(index=False))
    C.save(imp, "model_01_decision_tree_변수중요도.csv")

    print("\n학습된 판정 규칙 (상위 일부)")
    rules = export_text(model, feature_names=list(feats), max_depth=3)
    print("\n".join(rules.split("\n")[:25]))
    with open(os.path.join(C.OUT, "model_01_decision_tree_규칙.txt"), "w",
              encoding="utf-8") as fp:
        fp.write(rules)

    return results


if __name__ == "__main__":
    main()
