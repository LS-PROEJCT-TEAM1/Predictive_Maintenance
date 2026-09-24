# -*- coding: utf-8 -*-
"""
모델 D : LSTM Autoencoder  (시계열 재구성 오차 기반 이상탐지)

앞의 세 모델은 '한 시점의 값'을 본다. 이 모델은 충·방전이 진행되면서
값이 어떤 순서로 변해가는지, 그 '패턴'을 학습한다.

    Encoder : 30시점 구간을 읽어 압축된 상태 벡터로 요약
    Decoder : 그 벡터만으로 원래 30시점을 복원
    → 정상 패턴은 잘 복원되고, 처음 보는 패턴은 복원이 안 된다.
      복원 실패 정도(재구성 오차)가 이상 점수다.

정상 데이터만 학습하므로 라벨이 필요 없다.

데이터
    학습 : data/preprocessed/train 10개 파일 (정상 56,805행), 라벨 미사용
           학습:검증 = 8:2 (검증 손실로 과적합 확인)
    평가 : Test03, Test04, Test06, Test07, Test08

환경
    TensorFlow 2.x (Keras 2 / Keras 3 모두 동작하도록 작성)

실행
    python src/models/model_D_lstm_autoencoder.py
    python src/models/model_D_lstm_autoencoder.py --epochs 20
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import common_data as C

MODEL_NAME = "LSTM Autoencoder"
PREFIX = "D_lstm_autoencoder"

WINDOW = 30          # 한 번에 보는 시점 수 (초 단위 데이터이므로 30초 구간)
STRIDE_TRAIN = 5     # 학습 윈도우 간격 (전부 쓰면 중복이 많아 학습이 느려진다)
LATENT = 32          # 압축 차원
EPOCHS = 15
BATCH = 128
QUANTILES = (0.99, 0.999, 0.9999)
KS = (1, 10, 30)


def make_windows(values, window, stride):
    """(N, F) 배열을 (개수, window, F) 윈도우로 자른다. 시작 인덱스도 함께 반환."""
    idx = np.arange(0, len(values) - window + 1, stride)
    if len(idx) == 0:
        return np.empty((0, window, values.shape[1])), idx
    out = np.stack([values[i:i + window] for i in idx])
    return out, idx


def build_model(window, n_features, latent):
    """LSTM Autoencoder 구조 정의. Keras 2 / 3 공통 API만 사용한다."""
    import tensorflow as tf
    from tensorflow.keras.layers import (Input, LSTM, RepeatVector,
                                         TimeDistributed, Dense)
    from tensorflow.keras.models import Model

    x = Input(shape=(window, n_features))
    # Encoder : 시계열 → 상태 벡터
    h = LSTM(64, activation="tanh", return_sequences=True)(x)
    h = LSTM(latent, activation="tanh", return_sequences=False)(h)
    # Decoder : 상태 벡터 → 시계열 복원
    h = RepeatVector(window)(h)
    h = LSTM(latent, activation="tanh", return_sequences=True)(h)
    h = LSTM(64, activation="tanh", return_sequences=True)(h)
    y = TimeDistributed(Dense(n_features))(h)

    model = Model(x, y)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def flag_with_rule(score, thr, k):
    raw = (score > thr).astype(int)
    if k <= 1:
        return raw
    keep = pd.Series(raw).rolling(k, min_periods=k).min().fillna(0).values
    out = np.zeros_like(raw)
    for i in np.where(keep == 1)[0]:
        out[i - k + 1:i + 1] = 1
    return out


def point_errors(model, values, window):
    """시점별 재구성 오차.

    윈도우 단위로 예측한 뒤, 각 시점이 포함된 모든 윈도우의 오차를 평균낸다.
    윈도우가 겹치므로 한 시점이 여러 윈도우에 들어간다.
    """
    n = len(values)
    if n < window:
        return np.zeros(n)
    X, idx = make_windows(values, window, 1)
    pred = model.predict(X, batch_size=512, verbose=0)
    err = np.mean((X - pred) ** 2, axis=2)          # (윈도우, 시점)

    acc = np.zeros(n)
    cnt = np.zeros(n)
    for j, start in enumerate(idx):
        acc[start:start + window] += err[j]
        cnt[start:start + window] += 1
    cnt[cnt == 0] = 1
    return acc / cnt


def main(df_all=None, cell_z=None, epochs=EPOCHS):
    if df_all is None:
        df_all, _, cell_z = C.load_dataset()
    normal_df, train_df, test_df = C.get_splits(df_all)

    feats = [c for c in C.REL_FEATURES if c != "방전여부"]
    scaler = StandardScaler().fit(normal_df[feats])

    # ---- 정상 데이터에서 학습 윈도우 생성 (파일별로 시간 순서 유지) ----
    Xs = []
    for f_name in normal_df["파일"].unique():
        v = scaler.transform(normal_df.loc[normal_df["파일"] == f_name, feats])
        w, _ = make_windows(v, WINDOW, STRIDE_TRAIN)
        if len(w):
            Xs.append(w)
    X = np.concatenate(Xs).astype("float32")
    rng = np.random.RandomState(C.RANDOM_STATE)
    rng.shuffle(X)
    print(f"\n[LSTM AE] 학습 윈도우 {X.shape} (window={WINDOW}, stride={STRIDE_TRAIN})")

    import tensorflow as tf
    tf.random.set_seed(C.RANDOM_STATE)
    np.random.seed(C.RANDOM_STATE)

    model = build_model(WINDOW, X.shape[2], LATENT)
    print(f"[LSTM AE] 파라미터 {model.count_params():,}개 / epochs={epochs}\n")

    hist = model.fit(X, X, epochs=epochs, batch_size=BATCH,
                     validation_split=0.2, shuffle=True, verbose=2)

    hist_df = pd.DataFrame({"epoch": np.arange(1, len(hist.history["loss"]) + 1),
                            "train_loss": hist.history["loss"],
                            "val_loss": hist.history["val_loss"]}).round(6)
    C.save(hist_df, f"model_{PREFIX}_학습곡선.csv")
    print(f"\n최종 학습 손실 {hist_df['train_loss'].iloc[-1]:.6f} / "
          f"검증 손실 {hist_df['val_loss'].iloc[-1]:.6f}")

    # 저장 (Keras 2 / 3 모두 .keras 포맷 지원)
    try:
        model.save(os.path.join(C.OUT, f"model_{PREFIX}.keras"))
    except Exception as e:
        print("(모델 저장 생략:", e, ")")

    # ---- 시점별 재구성 오차 ----
    def errors_by_file(frame):
        out = np.zeros(len(frame))
        files = frame["파일"].values
        for f_name in pd.unique(files):
            m = files == f_name
            v = scaler.transform(frame.loc[m, feats])
            out[m] = point_errors(model, v, WINDOW)
        return out

    print("\n재구성 오차 계산 중...")
    err_ok = errors_by_file(normal_df)
    err_tr = errors_by_file(train_df)
    err_te = errors_by_file(test_df)

    # ---- 임계값 탐색 : 학습 데이터에서만 ----
    y_tr = train_df["label"].values
    tune, best = [], None
    for q in QUANTILES:
        thr = np.quantile(err_ok, q)
        for k in KS:
            pred = np.zeros(len(train_df), dtype=int)
            files = train_df["파일"].values
            for f_name in pd.unique(files):
                m = files == f_name
                pred[m] = flag_with_rule(err_tr[m], thr, k)
            pr = precision_score(y_tr, pred, zero_division=0)
            rc = recall_score(y_tr, pred, zero_division=0)
            tune.append({"분위수": q, "연속 k": k, "임계값": thr,
                         "Precision": pr, "Recall": rc,
                         "F1-score": f1_score(y_tr, pred, zero_division=0)})
            if pr >= 0.9 and (best is None or rc > best[0]):
                best = (rc, q, k, thr)
    tune_df = pd.DataFrame(tune).round(6)
    C.save(tune_df, f"model_{PREFIX}_임계값탐색.csv")
    if best is None:
        r = tune_df.loc[tune_df["F1-score"].idxmax()]
        best = (r["Recall"], r["분위수"], int(r["연속 k"]), r["임계값"])
    _, best_q, best_k, thr = best
    print(f"[LSTM AE] 임계값: 정상 {best_q} 분위수, 연속 {best_k}시점 "
          f"→ 재구성오차 > {thr:.6f}\n")

    def predict(frame, err):
        out = np.zeros(len(frame), dtype=int)
        files = frame["파일"].values
        for f_name in pd.unique(files):
            m = files == f_name
            out[m] = flag_with_rule(err[m], thr, best_k)
        return out

    tr_pred = predict(train_df, err_tr)
    te_pred = predict(test_df, err_te)

    results = [C.evaluate(MODEL_NAME, train_df["label"], tr_pred, "학습"),
               C.evaluate(MODEL_NAME, test_df["label"], te_pred, "테스트")]

    C.save(pd.DataFrame(results).round(4), f"model_{PREFIX}_성능.csv")
    C.save(C.evaluate_per_file(MODEL_NAME, test_df, te_pred), f"model_{PREFIX}_파일별.csv")

    # ---- 파일 단위 교차검증 (Out-of-Fold 통합, 단일 값) ----
    # Autoencoder 자체는 정상 데이터로만 학습되므로 fold마다 다시 학습하지 않는다.
    # 라벨이 쓰이는 임계값 선택 절차만 fold별로 다시 수행해 검증한다.
    lab = df_all[df_all["파일"].str.startswith("Test")].reset_index(drop=True)
    lab_err = errors_by_file(lab)

    def fit_predict(tr_part, te_part):
        tr_idx, te_idx = tr_part.index.values, te_part.index.values
        best_local = None
        for q in QUANTILES:
            t = np.quantile(err_ok, q)
            for kk in KS:
                pr_pred = np.zeros(len(tr_part), dtype=int)
                for f_name in tr_part["파일"].unique():
                    mm = (tr_part["파일"] == f_name).values
                    pr_pred[mm] = flag_with_rule(lab_err[tr_idx][mm], t, kk)
                pr = precision_score(tr_part["label"], pr_pred, zero_division=0)
                rc = recall_score(tr_part["label"], pr_pred, zero_division=0)
                f1 = f1_score(tr_part["label"], pr_pred, zero_division=0)
                key = (rc if pr >= 0.9 else -1, f1)
                if best_local is None or key > best_local[0]:
                    best_local = (key, t, kk)
        _, t, kk = best_local
        pred = np.zeros(len(te_part), dtype=int)
        for f_name in te_part["파일"].unique():
            mm = (te_part["파일"] == f_name).values
            pred[mm] = flag_with_rule(lab_err[te_idx][mm], t, kk)
        return pred

    cv, fold_df, _ = C.oof_cross_validation(lab, fit_predict, model_name=MODEL_NAME)
    C.save(cv, f"model_{PREFIX}_교차검증.csv")
    C.save(fold_df, f"model_{PREFIX}_교차검증_fold별.csv")

    out = test_df[["파일", "충방전", "판정", "시점", "label"]].copy()
    out["예측"] = te_pred
    out["재구성오차"] = np.round(err_te, 6)
    C.save(out, f"model_{PREFIX}_이상점수.csv")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    a = ap.parse_args()
    main(epochs=a.epochs)
