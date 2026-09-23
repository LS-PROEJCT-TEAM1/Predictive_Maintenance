# -*- coding: utf-8 -*-
"""
모델: Hotelling T^2 + SPE(Q) 통계적 공정관리(SPC) 기반 이상탐지.

PCA로 정상(1000_chg/1000_dchg) 데이터의 주성분 공간을 학습한 뒤,
- T^2: 주성분 공간 안에서 정상 분포로부터 얼마나 떨어져 있는지
- SPE(Q): 주성분 공간 밖(재구성 오차)으로 얼마나 벗어났는지
두 통계량이 각각의 관리한계(99th percentile, 학습 데이터 기준)를 넘으면 이상으로 판정한다.

라벨을 쓰지 않는 완전 비지도 모델이므로 라벨 있는 9개 파일 전체에서 평가한다.
실행: python model_hotelling_t2_spe.py [--train-limit 10] [--alpha 0.01]
결과: output/metrics_hotelling_t2_spe.csv
"""

import argparse

import numpy as np
from sklearn.decomposition import PCA

import common as C


class HotellingSPE:
    def __init__(self, n_components=3, alpha=0.01):
        self.n_components = n_components
        self.alpha = alpha
        self.pca = None
        self.t2_limit = None
        self.spe_limit = None

    def fit(self, X):
        n_comp = min(self.n_components, X.shape[1])
        self.pca = PCA(n_components=n_comp)
        scores = self.pca.fit_transform(X)
        eigvals = self.pca.explained_variance_
        t2 = np.sum((scores ** 2) / np.maximum(eigvals, 1e-12), axis=1)
        recon = self.pca.inverse_transform(scores)
        spe = np.sum((X - recon) ** 2, axis=1)
        self.t2_limit = np.percentile(t2, 100 * (1 - self.alpha))
        self.spe_limit = np.percentile(spe, 100 * (1 - self.alpha))
        return self

    def _stats(self, X):
        scores = self.pca.transform(X)
        eigvals = self.pca.explained_variance_
        t2 = np.sum((scores ** 2) / np.maximum(eigvals, 1e-12), axis=1)
        recon = self.pca.inverse_transform(scores)
        spe = np.sum((X - recon) ** 2, axis=1)
        return t2, spe

    def predict(self, X):
        t2, spe = self._stats(X)
        return ((t2 > self.t2_limit) | (spe > self.spe_limit)).astype(int)

    def score(self, X):
        """T2/SPE 각각을 자기 한계치로 나눠 더한 값 - 그래프/비교용 연속 이상 점수."""
        t2, spe = self._stats(X)
        return t2 / self.t2_limit + spe / self.spe_limit


def parse_args():
    parser = argparse.ArgumentParser(description="Hotelling T^2+SPE 이상탐지 비교 모델")
    parser.add_argument("--train-limit", type=int, default=10,
                        help="학습에 쓸 팩 개수 (기본 10, 0 또는 음수면 전체 사용, MTadGAN --train-limit과 동일)")
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="관리한계 유의수준 (기본 0.01 = 99th percentile)")
    return parser.parse_args()


def main():
    args = parse_args()
    train_limit = None if args.train_limit is not None and args.train_limit <= 0 else args.train_limit

    print("데이터 로딩 및 정규화 중... (train_limit=%s, alpha=%s)" % (train_limit, args.alpha))
    train_feats, norm_data = C.load_all_normalized(train_limit=train_limit)

    models = {}
    for mode in ("chg", "dchg"):
        m = HotellingSPE(n_components=3, alpha=args.alpha)
        m.fit(train_feats[mode].values)
        models[mode] = m
        print("  [%s] T2 한계=%.3f  SPE 한계=%.3f" % (mode, m.t2_limit, m.spe_limit))

    rows = []
    for tag, (feat, label, mode) in norm_data.items():
        model = models[mode]
        pred = model.predict(feat.values)
        score = model.score(feat.values)
        metrics = C.evaluate_and_report(pred, label, score, tag, "hotelling_t2_spe")
        rows.append(metrics)

    C.print_and_save("hotelling_t2_spe", rows)


if __name__ == "__main__":
    main()
