# -*- coding: utf-8 -*-
"""
모델: Isolation Forest (sklearn) - 비지도 이상탐지.

정상 데이터(1000_chg/1000_dchg)로만 학습한 뒤, 각 샘플이 얼마나 "쉽게 고립되는지"
(트리에서 분할 횟수가 적을수록 이상치)로 이상 여부를 판정한다.

라벨을 쓰지 않는 완전 비지도 모델이므로 라벨 있는 9개 파일 전체에서 평가한다.

Isolation Forest는 반복학습(gradient descent) 대신 트리를 몇 개 만드는지
(n_estimators)로 "얼마나 학습을 많이 하는지"를 조절한다 - epoch와 비슷한 역할.

실행: python model_isolation_forest.py [--train-limit 10] [--n-estimators 200]
결과: output/metrics_isolation_forest.csv
"""

import argparse

from sklearn.ensemble import IsolationForest

import common as C


def parse_args():
    parser = argparse.ArgumentParser(description="Isolation Forest 이상탐지 비교 모델")
    parser.add_argument("--train-limit", type=int, default=10,
                        help="학습에 쓸 팩 개수 (기본 10, 0 또는 음수면 전체 사용, MTadGAN --train-limit과 동일)")
    parser.add_argument("--n-estimators", type=int, default=200,
                        help="트리 개수 (기본 200) - epoch와 비슷한 '얼마나 학습을 많이 하는지' 역할")
    parser.add_argument("--contamination", type=float, default=0.05,
                        help="이상치 비율 추정값 (기본 0.05)")
    return parser.parse_args()


def main():
    args = parse_args()
    train_limit = None if args.train_limit is not None and args.train_limit <= 0 else args.train_limit

    print("데이터 로딩 및 정규화 중... (train_limit=%s, n_estimators=%d)"
          % (train_limit, args.n_estimators))
    train_feats, norm_data = C.load_all_normalized(train_limit=train_limit)

    models = {}
    for mode in ("chg", "dchg"):
        m = IsolationForest(n_estimators=args.n_estimators,
                             contamination=args.contamination, random_state=42)
        m.fit(train_feats[mode].values)
        models[mode] = m
        print("  [%s] 학습 완료 (n_estimators=%d, contamination=%s)"
              % (mode, args.n_estimators, args.contamination))

    rows = []
    for tag, (feat, label, mode) in norm_data.items():
        model = models[mode]
        raw_pred = model.predict(feat.values)  # 1=정상, -1=이상
        pred = (raw_pred == -1).astype(int)
        score = -model.decision_function(feat.values)  # 클수록 이상 (부호 반전)
        metrics = C.evaluate_and_report(pred, label, score, tag, "isolation_forest")
        rows.append(metrics)

    C.print_and_save("isolation_forest", rows)


if __name__ == "__main__":
    main()
