# -*- coding: utf-8 -*-
"""
모델: PCA 재구성오차(SPE) 기반 이상탐지 - 가장 단순한 비지도 베이스라인.

Hotelling T^2 + SPE 모델과 같은 PCA를 쓰지만, 그 모델은 주성분 공간 "안"에서의
분포 이탈(T^2)과 "밖"으로의 이탈(SPE)을 함께 본다. 이 모델은 그중 SPE(재구성오차)
하나만 써서 가장 단순한 형태의 PCA 이상탐지를 구현한다 - 다른 모델들이 "PCA를
더 정교하게 쓰면 얼마나 좋아지는지"를 비교할 수 있는 하한선 역할.

정상 데이터(1000_chg/1000_dchg)로 PCA를 학습한 뒤, 각 샘플을 주성분 공간에
투영했다가 원래 차원으로 복원할 때 생기는 오차(재구성오차)가 임계값(학습 데이터
기준 99th percentile)을 넘으면 이상으로 판정한다.

라벨을 쓰지 않는 완전 비지도 모델이므로 라벨 있는 9개 파일 전체에서 평가한다.
실행: python model_pca.py [--train-limit 10] [--alpha 0.01]
결과: output/metrics_pca.csv
"""

import argparse

import numpy as np
from sklearn.decomposition import PCA

import common as C


class PCAReconstruction:
    def __init__(self, n_components=3, alpha=0.01):
        self.n_components = n_components
        self.alpha = alpha
        self.pca = None
        self.limit = None

    def fit(self, X):
        n_comp = min(self.n_components, X.shape[1])
        self.pca = PCA(n_components=n_comp)
        scores = self.pca.fit_transform(X)
        recon = self.pca.inverse_transform(scores)
        spe = np.sum((X - recon) ** 2, axis=1)
        self.limit = np.percentile(spe, 100 * (1 - self.alpha))
        return self

    def spe(self, X):
        scores = self.pca.transform(X)
        recon = self.pca.inverse_transform(scores)
        return np.sum((X - recon) ** 2, axis=1)

    def predict(self, X):
        return (self.spe(X) > self.limit).astype(int)


def parse_args():
    parser = argparse.ArgumentParser(description="PCA 재구성오차(SPE) 이상탐지 비교 모델")
    parser.add_argument("--train-limit", type=int, default=10,
                        help="학습에 쓸 팩 개수 (기본 10, 0 또는 음수면 전체 사용, MTadGAN --train-limit과 동일)")
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="임계값 유의수준 (기본 0.01 = 99th percentile)")
    return parser.parse_args()


def main():
    args = parse_args()
    train_limit = None if args.train_limit is not None and args.train_limit <= 0 else args.train_limit

    print("데이터 로딩 및 정규화 중... (train_limit=%s, alpha=%s)" % (train_limit, args.alpha))
    train_feats, norm_data = C.load_all_normalized(train_limit=train_limit)

    models = {}
    for mode in ("chg", "dchg"):
        m = PCAReconstruction(n_components=3, alpha=args.alpha)
        m.fit(train_feats[mode].values)
        models[mode] = m
        print("  [%s] 재구성오차(SPE) 임계값(99pct)=%.4f" % (mode, m.limit))

    rows = []
    for tag, (feat, label, mode) in norm_data.items():
        model = models[mode]
        pred = model.predict(feat.values)
        score = model.spe(feat.values)
        metrics = C.evaluate_and_report(pred, label, score, tag, "pca")
        rows.append(metrics)

    C.print_and_save("pca", rows)
    print("\n(참고: Hotelling T^2+SPE와 같은 PCA를 쓰지만 SPE만 단독으로 판정 - 가장 단순한 PCA 베이스라인)")


if __name__ == "__main__":
    main()
