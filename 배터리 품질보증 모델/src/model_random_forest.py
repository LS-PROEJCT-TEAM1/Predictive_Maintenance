# -*- coding: utf-8 -*-
"""
모델: RandomForest (지도학습) - 참고용 성능 상한선(upper bound).

라벨을 직접 보고 학습하는 지도학습 모델이라 MTadGAN 등 비지도 모델과
"같은 조건"에서 비교되는 것은 아니다. 다만 "라벨을 알고 있다면 어디까지
잡아낼 수 있는가"를 보여주는 참고선으로 같이 리포트한다.

라벨 있는 9개 파일 중 5개(SUP_TRAIN_TAGS)로 학습하고 나머지 4개(SUP_EVAL_TAGS)로 평가한다.

RandomForest도 반복학습(gradient descent) 대신 트리를 몇 개 만드는지
(n_estimators)로 "얼마나 학습을 많이 하는지"를 조절한다 - epoch와 비슷한 역할.

실행: python model_random_forest.py [--train-limit 10] [--n-estimators 200]
결과: output/metrics_random_forest.csv
"""

import argparse

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import common as C


def parse_args():
    parser = argparse.ArgumentParser(description="RandomForest 이상탐지 비교 모델 (지도학습 상한선)")
    parser.add_argument("--train-limit", type=int, default=10,
                        help="정규화 기준(ref_std)을 만들 때 쓸 팩 개수 (기본 10, MTadGAN --train-limit과 동일)")
    parser.add_argument("--n-estimators", type=int, default=200,
                        help="트리 개수 (기본 200) - epoch와 비슷한 '얼마나 학습을 많이 하는지' 역할")
    return parser.parse_args()


def main():
    args = parse_args()
    train_limit = None if args.train_limit is not None and args.train_limit <= 0 else args.train_limit

    print("데이터 로딩 및 정규화 중... (train_limit=%s, n_estimators=%d)"
          % (train_limit, args.n_estimators))
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

    clf = RandomForestClassifier(n_estimators=args.n_estimators,
                                  class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)

    rows = []
    for tag in C.SUP_EVAL_TAGS:
        feat, label, _ = norm_data[tag]
        pred = clf.predict(feat.values)
        score = clf.predict_proba(feat.values)[:, 1]  # 이상(1) 클래스 확률
        metrics = C.evaluate_and_report(pred, label, score, tag, "random_forest")
        rows.append(metrics)

    C.print_and_save("random_forest", rows)
    print("\n(참고: 지도학습 상한선 - 비지도 모델과 직접 비교 대상이 아님, 학습 5개/평가 4개로 분리)")


if __name__ == "__main__":
    main()
