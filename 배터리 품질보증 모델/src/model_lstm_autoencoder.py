# -*- coding: utf-8 -*-
"""
모델: LSTM Autoencoder - 비지도 이상탐지 (MTadGAN의 ablation 성격).

MTadGAN에서 GAN/critic 구조를 빼고 인코더-디코더(LSTM)만 남긴 구조.
정상 데이터로만 재구성 학습을 한 뒤, 재구성 오차가 큰 구간을 이상으로 판정한다.
GAN의 critic score 없이 순수 재구성 오차만 쓰기 때문에 MTadGAN과의 비교를 통해
"critic 구조가 성능에 실제로 얼마나 기여하는지"를 볼 수 있다.

TensorFlow가 설치되어 있어야 실행된다 (사용자 PC에서 실행 권장).
라벨을 쓰지 않는 완전 비지도 모델이므로 라벨 있는 9개 파일 전체에서 평가한다.
실행: python model_lstm_autoencoder.py [--epochs 30] [--train-limit 10]
결과: output/metrics_lstm_autoencoder.csv
"""

import argparse

import numpy as np

import common as C

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    _HAS_TF = True
except ImportError:
    _HAS_TF = False


def build_autoencoder(win_size, n_features, latent_dim=20):
    inputs = layers.Input(shape=(win_size, n_features))
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inputs)
    x = layers.LSTM(latent_dim)(x)
    x = layers.RepeatVector(win_size)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    outputs = layers.TimeDistributed(layers.Dense(n_features))(x)
    ae = models.Model(inputs, outputs)
    ae.compile(optimizer="adam", loss="mse")
    return ae


def reconstruction_scores(ae, windows):
    recon = ae.predict(windows, verbose=0)
    return np.mean((windows - recon) ** 2, axis=(1, 2))


def parse_args():
    parser = argparse.ArgumentParser(description="LSTM Autoencoder 이상탐지 비교 모델")
    parser.add_argument("--epochs", type=int, default=30,
                        help="학습 epoch 수 (기본 30, MTadGAN과 동일한 기본값)")
    parser.add_argument("--train-limit", type=int, default=10,
                        help="학습에 쓸 팩 개수 (기본 10, 0 또는 음수면 전체 사용)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not _HAS_TF:
        print("TensorFlow가 설치되어 있지 않아 이 모델은 건너뜁니다.")
        print("(사용자 PC에서 'pip install tensorflow' 후 다시 실행하세요.)")
        return
    train_limit = None if args.train_limit is not None and args.train_limit <= 0 else args.train_limit

    print("데이터 로딩 및 정규화 중... (epochs=%d, train_limit=%s)" % (args.epochs, train_limit))
    train_feats, norm_data = C.load_all_normalized(train_limit=train_limit)

    models_by_mode = {}
    thresholds = {}
    for mode in ("chg", "dchg"):
        # 팩(파일) 경계를 넘어 윈도우가 만들어지면 안 되므로, 팩별로 따로 윈도우를
        # 만든 뒤 이어붙인다 (train_feats["chg_files"] = 팩별 정규화 결과 리스트).
        per_pack_windows = [C.make_windows(f.values, win_size=C.SEQ_WIN_SIZE)
                            for f in train_feats["%s_files" % mode]]
        per_pack_windows = [w for w in per_pack_windows if len(w) > 0]
        X_train = np.concatenate(per_pack_windows, axis=0)
        ae = build_autoencoder(C.SEQ_WIN_SIZE, X_train.shape[-1])
        print("  [%s] 학습 중... (windows=%d)" % (mode, len(X_train)))
        ae.fit(X_train, X_train, epochs=args.epochs, batch_size=64, verbose=0)
        train_scores = reconstruction_scores(ae, X_train)
        thresholds[mode] = float(np.percentile(train_scores, 99))
        models_by_mode[mode] = ae
        print("  [%s] 학습 완료, 임계값(99pct)=%.6f" % (mode, thresholds[mode]))

    rows = []
    for tag, (feat, label, mode) in norm_data.items():
        windows = C.make_windows(feat.values, win_size=C.SEQ_WIN_SIZE)
        if len(windows) == 0:
            print("  건너뜀 (윈도우 부족): %s" % tag)
            continue
        scores = reconstruction_scores(models_by_mode[mode], windows)
        point_scores = C.window_scores_to_pointwise(scores, len(feat), win_size=C.SEQ_WIN_SIZE)
        pred = (point_scores > thresholds[mode]).astype(int)
        metrics = C.evaluate_and_report(pred, label, point_scores, tag, "lstm_autoencoder")
        rows.append(metrics)

    C.print_and_save("lstm_autoencoder", rows)


if __name__ == "__main__":
    main()
