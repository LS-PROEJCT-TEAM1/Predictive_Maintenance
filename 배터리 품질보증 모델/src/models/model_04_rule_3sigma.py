# -*- coding: utf-8 -*-
"""
모델 4 : 3σ 규칙 기반 이상탐지  - 교안 Part 6 Ch46 (코드 46-5 / 46-6)

정상 기준 데이터: preprocessed/train 10개 파일 (충전 5 + 방전 5) 56,805행
    → 변수별 임계값 = 정상 데이터의 평균 + 3 × 표준편차
평가: Test03, Test04, Test06, Test07, Test08

불량 유형이 서로 다르므로(온도 불균형 / 셀 전압 이탈) 변수 하나로는 부족하다.
변수별 단독 성능을 먼저 보고, 유형별 규칙을 OR로 묶는다.

실행
    python src/models/model_04_rule_3sigma.py
"""

import os
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import common_data as C

MODEL_NAME = "3σ 규칙"
SIGMA = 3.0

# 단독 성능을 확인할 변수
CANDIDATES = ["모듈온도_편차", "모듈온도_표준편차", "셀전압_최대z", "이상셀_개수",
              "셀전압_편차", "셀전압_표준편차"]

# 최종 규칙에 쓸 변수 (서로 다른 불량 유형을 담당)
RULE_COLS = ["모듈온도_편차", "셀전압_최대z"]


def main():
    df_all, report, _ = C.load_dataset()
    normal_df, train_df, test_df = C.get_splits(df_all)

    y_test = test_df["label"]

    # ---- 임계값 산출 : 정상 데이터만 사용 (교안 코드 46-5) ----
    print(f"\n정상 기준 {len(normal_df):,}행으로 임계값 산출 "
          f"(평균 + {SIGMA:.0f} × 표준편차)\n")
    thr_rows = []
    for c in CANDIDATES:
        mean, std = normal_df[c].mean(), normal_df[c].std()
        up = mean + SIGMA * std
        # IQR 기준도 함께 계산 (교안 코드 46-6)
        q1, q3 = normal_df[c].quantile(.25), normal_df[c].quantile(.75)
        up_iqr = q3 + 1.5 * (q3 - q1)

        p3 = (test_df[c] > up).astype(int).values
        pi = (test_df[c] > up_iqr).astype(int).values
        thr_rows.append({
            "변수": c, "정상 평균": mean, "정상 표준편차": std,
            "3σ 임계값": up, "IQR 임계값": up_iqr,
            "3σ Precision": precision_score(y_test, p3, zero_division=0),
            "3σ Recall": recall_score(y_test, p3, zero_division=0),
            "3σ F1": f1_score(y_test, p3, zero_division=0),
            "IQR Precision": precision_score(y_test, pi, zero_division=0),
            "IQR Recall": recall_score(y_test, pi, zero_division=0),
        })
    thr_df = pd.DataFrame(thr_rows).round(4)
    print("변수별 단독 규칙 성능 (테스트 기준)")
    print(thr_df[["변수", "3σ 임계값", "3σ Precision", "3σ Recall", "3σ F1"]]
          .to_string(index=False))
    C.save(thr_df, "model_04_rule_변수별_임계값.csv")

    # ---- 최종 규칙 : 유형별 규칙 OR 결합 ----
    thr = {c: normal_df[c].mean() + SIGMA * normal_df[c].std() for c in RULE_COLS}
    print("\n최종 규칙:", " 또는 ".join(f"{c} > {v:.4f}" for c, v in thr.items()))

    def predict(frame):
        flag = np.zeros(len(frame), dtype=bool)
        for c, v in thr.items():
            flag |= (frame[c] > v).values
        return flag.astype(int)

    tr_pred, te_pred = predict(train_df), predict(test_df)
    print()
    results = [C.evaluate(MODEL_NAME, train_df["label"], tr_pred, "학습"),
               C.evaluate(MODEL_NAME, y_test, te_pred, "테스트")]
    ex = (test_df["파일"] != "Test07_NG_dchg").values
    results.append(C.evaluate(MODEL_NAME, y_test.values[ex], te_pred[ex],
                              "테스트(Test07 제외)"))

    C.save(pd.DataFrame(results).round(4), "model_04_rule_성능.csv")
    C.save(C.evaluate_per_file(MODEL_NAME, test_df, te_pred), "model_04_rule_파일별.csv")

    # ---- 판정 근거 저장 (어느 규칙에 걸렸는지) ----
    out = test_df[["파일", "충방전", "판정", "시점", "label"]].copy()
    out["규칙_예측"] = te_pred
    for c, v in thr.items():
        out[f"{c}_초과"] = (test_df[c] > v).astype(int).values
        out[f"{c}_값"] = test_df[c].round(4).values
    C.save(out, "model_04_rule_판정근거.csv")

    return results


if __name__ == "__main__":
    main()
