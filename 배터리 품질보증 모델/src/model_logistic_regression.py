# -*- coding: utf-8 -*-
"""
모델: LogisticRegression (지도학습) - 참고용 성능 상한선(upper bound).

RandomForest와 마찬가지로 라벨을 직접 보고 학습하는 지도학습 모델이라
비지도 모델과 "같은 조건" 비교는 아니지만, 가장 단순한 선형 분류기 기준선으로
같이 리포트한다. (RandomForest보다도 단순한 하한/기준선 역할)

라벨 있는 9개 파일 중 5개(SUP_TRAIN_TAGS)로 학습하고 나머지 4개(SUP_EVAL_TAGS)로 평가한다.

LogisticRegression은 solver(lbfgs)가 수렴할 때까지 반복 최적화를 하는 모델이라
max_iter(최대 반복 횟수)가 epoch와 가장 비슷한 역할을 한다.

실행: python model_logistic_regression.py [--train-limit 10] [--max-iter 1000]
결과: output/metrics_logistic_regression.csv
"""

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression

import common as C


def parse_args():
    parser = argparse.ArgumentParser(description="LogisticRegression 이상탐지 비교 모델 (지도학습 기준선)")
    parser.add_argument("--train-limit", type=int, default=10,
                        help="정규화 기준(ref_std)을 만들 때 쓸 팩 개수 (기본 10, MTadGAN --train-limit과 동일)")
    parser.add_argument("--max-iter", type=int, default=1000,
                        help="solver 최대 반복 횟수 (기본 1000) - epoch와 가장 비슷한 역할")
    return parser.parse_args()


def main():
    args = parse_args()
    train_limit = None if args.train_limit is not None and args.train_limit <= 0 else args.train_limit

    print("데이터 로딩 및 정규화 중... (train_limit=%s, max_iter=%d)"
          % (train_limit, args.max_iter))
    _, norm_data = C.load_all_normalized(train_limit=train_limit)

    X_train_list, y_train_list = [], []
    for tag in C.SUP_TRAIN_TAGS:
        feat, label, _ = norm_data[tag]
        X_train_list.append(feat.values)
        y_train_list.append(label)
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    print("  학습 데이터: %s (이상 %d개, %.1f%%)" %
          (X_train.shape, y_train.sum(), 100 * y_train.mean()))

    clf = LogisticRegression(max_iter=args.max_iter, class_weight="balanced")
    clf.fit(X_train, y_train)

    rows = []
    for tag in C.SUP_EVAL_TAGS:
        feat, label, _ = norm_data[tag]
        pred = clf.predict(feat.values)
        score = clf.predict_proba(feat.values)[:, 1]  # 이상(1) 클래스 확률
        metrics = C.evaluate_and_report(pred, label, score, tag, "logistic_regression")
        rows.append(metrics)

    C.print_and_save("logistic_regression", rows)
    print("\n(참고: 지도학습 기준선 - 비지도 모델과 직접 비교 대상이 아님, 학습 5개/평가 4개로 분리)")


if __name__ == "__main__":
    main()
