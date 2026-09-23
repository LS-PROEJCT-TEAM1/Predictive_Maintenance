# -*- coding: utf-8 -*-
# """
# 전자부품(배터리팩) 품질보증 AI 데이터셋 - MTadGAN 분석 실습
# KAMP 분석실습 가이드북 [단계 ①] ~ [단계 ⑧] 전체 재현 (단일 스크립트)

# 출처: 중소벤처기업부, Korea AI Manufacturing Platform(KAMP),
#       전자부품(배터리팩) 품질보증 AI 데이터셋, 스마트제조혁신추진단
#       (㈜인터엑스, 네스트필드㈜), 2022.12.23., www.kamp-ai.kr

# 디렉터리 구조 (프로젝트 루트 기준)
#     data/raw_data/train/1000_chg.csv ...
#     data/raw_data/test/Test07_NG_dchg.csv ...
#     data/preprocessed/train/, data/preprocessed/test/
#     checkpoints/, networks/, output/

# 실행 예시
#     # 전처리 + 학습 + 테스트 + 성능평가 (가이드북 기본 설정)
#     python battery_quality_mtadgan.py --mode all

#     # 학습만 / 테스트만
#     python battery_quality_mtadgan.py --mode train
#     python battery_quality_mtadgan.py --mode predict

#     # 다른 테스트 파일로 평가
#     python battery_quality_mtadgan.py --mode predict \
#         --test-file Test05_NG_chg.csv --test-label Test05_NG_chg_Label.csv

# 필요 패키지 (가이드북 부록 3)
#     tensorflow==2.5.0, scikit-learn, pandas, numpy, scipy, matplotlib,
#     plotly, pyts, pydot, pydotplus, graphviz
# """

import argparse
import collections
import glob
import math
import os
import warnings

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from scipy import integrate, stats
from scipy.optimize import fmin

import matplotlib
matplotlib.use("Agg")           # 서버/CLI 환경에서도 그림 저장이 되도록
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

warnings.simplefilter(action="ignore", category=FutureWarning)


# =============================================================================
# 0. 경로 및 하이퍼파라미터  ([코드 51] 하이퍼파라미터 설정)
# =============================================================================

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATHS = {
    "raw_train":  os.path.join(ROOT, "data", "raw_data", "train"),
    "raw_test":   os.path.join(ROOT, "data", "raw_data", "test"),
    "pre_train":  os.path.join(ROOT, "data", "preprocessed", "train"),
    "pre_test":   os.path.join(ROOT, "data", "preprocessed", "test"),
    "ckpt":       os.path.join(ROOT, "checkpoints"),
    "network":    os.path.join(ROOT, "networks"),
    "output":     os.path.join(ROOT, "output"),
}

# 가이드북 [단계 ⑤] 하이퍼파라미터
WIN_SIZE = 10          # 시계열 입력 윈도우 크기
FEATURES_DIM = 3       # PCA로 축소한 피쳐 차원수 (원본 208차원 -> 3차원)

PARAMS = {
    "plot_network": False,          # pydot/graphviz 설치 시 True 로 두면 구조도 저장
    "epochs": 30,
    "batch_size": 64,
    "n_critic": 5,
    "learning_rate": 0.0005,        # [코드 67]
    "latent_dim": 20,
    "shape": [WIN_SIZE, FEATURES_DIM],
    "encoder_input_shape": [WIN_SIZE, FEATURES_DIM],
    "encoder_reshape_shape": [20, 1],
    "generator_input_shape": [20, 1],
    "generator_reshape_shape": [WIN_SIZE, 1],
    "critic_x_input_shape": [WIN_SIZE, FEATURES_DIM],
    "critic_z_input_shape": [20, 1],
}

# 차분/평활/시차 설정 ([코드 53]) - 가이드북은 모두 0 (원 데이터 그대로 사용)
DIFFS_N, LAGS_N, SMOOTH_N = 0, 0, 0

Args = collections.namedtuple(
    "Args", "signal_file timest_form anomaly_file mode aggregate_interval regate_interval"
)


def ensure_dirs():
    for key in ("pre_train", "pre_test", "ckpt", "network", "output"):
        os.makedirs(PATHS[key], exist_ok=True)


# =============================================================================
# [단계 ①] 데이터 준비 - 전처리 함수들
#          ([코드 19] ~ [코드 29], 부록 4 데이터 품질 전처리)
# =============================================================================

def removeConstant(df, n=1):
    # """속성의 값이 n개 이하인(=사실상 상수인) 속성을 제거한다. [코드 19]"""
    return df[[col for col in df if df[col].nunique() > n]]


def handleMissingValue(data):
    # """결측치 처리. [코드 21]

    # - 범주형: 최빈값(mode)으로 대체
    # - 수치형: 앞/뒤 시점 값의 선형보간(평균), 처음/끝은 가장 가까운 유효값
    # """
    df = data.copy()
    categorical = df.columns[df.dtypes == "object"]

    for col in categorical:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    rows, cols = np.where(df.isnull())
    for j in range(len(rows)):
        r, c = rows[j], cols[j]
        if r == 0:                                   # 맨 앞 결측
            s = df.iloc[:, c]
            df.iat[r, c] = df.iat[s.notna().idxmax(), c]
        elif r == len(df) - 1:                       # 맨 뒤 결측
            s = df.iloc[:, c]
            df.iat[r, c] = df.iat[s.notna()[::-1].idxmax(), c]
        else:                                        # 중간 결측 -> 보간
            low = df.iat[r - 1, c]
            high = df.iat[r + 1, c]
            try:
                high_is_nan = math.isnan(float(high))
            except (TypeError, ValueError):
                high_is_nan = True
            df.iat[r, c] = low if high_is_nan else (low + high) / 2.0
    return df


def identify_outliers(df, c):
    # """IQR 기준 이상치 검출. c x IQR 범위를 벗어나면 이상치. [코드 23]"""
    constant = float(c)
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    mask = ((df.lt(q1 - constant * iqr)) | (df.gt(q3 + constant * iqr))).any(axis=1)
    return pd.DataFrame(df[mask])


def remove_outliers(df, c):
    # """검출된 이상치 행을 제거한다. [코드 24]"""
    df = pd.DataFrame(df).copy()
    outliers = identify_outliers(df, c)
    df.drop(outliers.index, inplace=True)
    return df


def select_voltage_temperature(df):
    # """셀 전압(M01CV01~M16CV11) / 온도(M01T01~M16T02) 속성만 추출. [코드 28], [코드 34]

    # 가이드북은 컬럼 인덱스(18:226, 4:212, 1:209)로 잘라내지만, 파일마다
    # 상수 컬럼 제거 결과가 달라 인덱스가 흔들리므로 컬럼명 패턴으로 선택한다.
    # """
    cols = [c for c in df.columns
            if c.startswith("M") and ("CV" in c or "T0" in c or "T1" in c)]
    return df[cols]


def preprocess_file(src_path, dst_path, iqr_constant=4.0, verbose=True):
    # """원 데이터 1개 파일에 대한 [단계 ①] 전체 전처리."""
    data = pd.read_csv(src_path)
    if verbose:
        print("\n=== 전처리: %s ===" % os.path.basename(src_path))
        print("원 데이터 shape        :", data.shape)
        print("결측치 총 개수         :", int(data.isna().sum().sum()))
        print("중복 행 개수           :", int(data.duplicated().sum()))

    data1 = removeConstant(data, 1)                      # [코드 19]
    data2 = handleMissingValue(data1)                    # [코드 22]

    numeric = data2.select_dtypes(include=[np.number])
    n_outliers = len(identify_outliers(numeric, iqr_constant))   # [코드 25]

    df_vt = select_voltage_temperature(data2)            # [코드 28]
    df_vt.to_csv(dst_path, index=False)                  # [코드 29]

    if verbose:
        print("상수 속성 제거 후 shape:", data1.shape)
        print("잔여 결측치            :", int(data2.isna().sum().sum()))
        print("이상치(4 x IQR) 행 수  :", n_outliers)
        print("전압/온도 추출 후 shape:", df_vt.shape)
        print("저장                   :", dst_path)
    return df_vt


def run_preprocess(train_file, test_file, test_label):
    # """학습 1개 + 테스트 1개 파일 전처리. 라벨 파일은 그대로 복사/확인."""
    ensure_dirs()
    src_train = os.path.join(PATHS["raw_train"], train_file)
    preprocess_file(src_train, os.path.join(PATHS["pre_train"], train_file))

    # 파일명에 공백이 섞여 있는 경우(Test03_OK_chg .csv)까지 커버
    src_test = os.path.join(PATHS["raw_test"], test_file)
    if not os.path.exists(src_test):
        cand = glob.glob(os.path.join(PATHS["raw_test"],
                                      test_file.replace(".csv", "*.csv")))
        if not cand:
            raise FileNotFoundError(src_test)
        src_test = cand[0]
    preprocess_file(src_test, os.path.join(PATHS["pre_test"], test_file))

    label_path = os.path.join(PATHS["pre_test"], test_label)
    if not os.path.exists(label_path):
        raise FileNotFoundError("라벨 파일이 없습니다: %s" % label_path)
    print("\n라벨 파일 확인:", label_path,
          pd.read_csv(label_path).shape)


# =============================================================================
# [단계 ②~③] 데이터 구조 탐색(PCA) 및 모델 입력 생성
#             ([코드 48], [코드 53] ~ [코드 58])
# =============================================================================

def diff_smooth_df(df, lags_n, diffs_n, smooth_n, diffs_abs=False, abs_features=False):
    # """차분 / 평활 / 시차(lag) 특징 생성. [코드 53]"""
    if diffs_n >= 1:
        df = df.diff(diffs_n).dropna()
        if diffs_abs:
            df = abs(df)
    if smooth_n >= 2:
        df = df.rolling(smooth_n).mean().dropna()
    if lags_n >= 1:
        df_columns_new = [f"{col}_lag{n}" for n in range(lags_n + 1) for col in df.columns]
        df = pd.concat([df.shift(n) for n in range(lags_n + 1)], axis=1).dropna()
        df.columns = df_columns_new
    df = df.reindex(sorted(df.columns), axis=1)
    if abs_features:
        df = abs(df)
    return df


def pca_reduce(df_raw, features_dim=FEATURES_DIM, tag=""):
    # """PCA로 차원을 features_dim 으로 축소하고 'date' 인덱스를 붙인다. [코드 54], [코드 77]"""
    data_1 = diff_smooth_df(df_raw, LAGS_N, DIFFS_N, SMOOTH_N)
    pca = PCA(n_components=features_dim)
    reduced = pca.fit_transform(data_1)

    rows = []
    for i in range(len(reduced)):
        rows.append([i + 1] + [reduced[i][j] for j in range(features_dim)])
    df = pd.DataFrame(rows,
                      columns=["date"] + ["pca_%d" % (i + 1) for i in range(features_dim)])

    print("\n[PCA] %s 축소 후 shape = %s / 설명분산비 = %s"
          % (tag, df.shape, np.round(pca.explained_variance_ratio_, 4)))
    print(df.head(3).to_string(index=False))
    return df


def time_segments_aggregate(X, interval, time_column, method=("mean",)):
    # """주어진 시간 구간 단위로 값을 집합화한다. [코드 55]"""
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    X = X.sort_values(time_column).set_index(time_column)
    if isinstance(method, str):
        method = [method]

    start_ts = X.index.values[0]
    max_ts = X.index.values[-1]
    values, index = [], []
    while start_ts <= max_ts:
        end_ts = start_ts + interval
        subset = X.loc[start_ts:end_ts - 1]
        aggregated = [getattr(subset, agg)(skipna=True).values for agg in method]
        values.append(np.concatenate(aggregated))
        index.append(start_ts)
        start_ts = end_ts
    return np.asarray(values), np.asarray(index)


def rolling_window_sequences(X, index, window_size, target_size, step_size,
                             target_column, drop=None, drop_windows=False):
    # """시계열을 윈도우 단위 시퀀스로 묶는다. [코드 57]"""
    out_X, out_y, X_index, y_index = [], [], [], []
    target = X[:, target_column]

    if drop_windows:
        if hasattr(drop, "__len__") and (not isinstance(drop, str)):
            if len(drop) != len(X):
                raise Exception("Arrays `drop` and `X` must be of the same length.")
        else:
            if isinstance(drop, float) and np.isnan(drop):
                drop = np.isnan(X)
            else:
                drop = X == drop

    start = 0
    max_start = len(X) - window_size - target_size + 1
    while start < max_start:
        end = start + window_size
        if drop_windows:
            drop_window = drop[start:end + target_size]
            to_drop = np.where(drop_window)[0]
            if to_drop.size:
                start += to_drop[-1] + 1
                continue
        out_X.append(X[start:end])
        out_y.append(target[end:end + target_size])
        X_index.append(index[start])
        y_index.append(index[end])
        start += step_size
    return (np.asarray(out_X), np.asarray(out_y),
            np.asarray(X_index), np.asarray(y_index))


def build_model_input(df_signal, aggregate_interval=1, tag=""):
    # """PCA -> 시간집합화 -> [-1,1] 정규화 -> 윈도우 묶음. [코드 54]~[코드 58]"""
    df = pca_reduce(df_signal, FEATURES_DIM, tag)

    X, index = time_segments_aggregate(df, interval=aggregate_interval, time_column="date")
    X = SimpleImputer().fit_transform(X)
    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)      # [코드 56]

    X, y, X_index, y_index = rolling_window_sequences(
        X, index, window_size=WIN_SIZE, target_size=1, step_size=1, target_column=0)
    print("[윈도우] %s X shape = %s, y shape = %s" % (tag, X.shape, y.shape))
    return X, y, X_index, y_index


# =============================================================================
# [단계 ④~⑤] MTadGAN 모델 정의 및 학습
#             ([코드 60] ~ [코드 73])
# =============================================================================

def _import_tf():
    # """TensorFlow는 학습/예측 시점에만 불러온다(전처리만 돌릴 때 불필요)."""
    import tensorflow as tf
    return tf


def build_networks():
    # """encoder / generator / critic_x / critic_z 및 조합 모델 생성."""
    tf = _import_tf()
    from tensorflow.keras import backend as K
    from tensorflow.keras.layers import (Activation, Bidirectional, Conv1D, Dense,
                                         Dropout, Flatten, Input, Layer, LeakyReLU,
                                         LSTM, Reshape, TimeDistributed, UpSampling1D)
    from tensorflow.keras.models import Model

    # ----- [코드 59] GPU 확인 -----
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    print("GPU:", gpus if gpus else "없음 (CPU로 학습)")

    batch_size = PARAMS["batch_size"]
    latent_dim = PARAMS["latent_dim"]
    shape = PARAMS["shape"]

    # ----- [코드 60] RandomWeightedAverage -----
    class RandomWeightedAverage(Layer):
        def __init__(self, batch_size):
            super().__init__()
            self.batch_size = batch_size

        def call(self, inputs, **kwargs):
            # Keras 3 에는 K.random_uniform 이 없으므로 tf.random.uniform 을 쓴다.
            alpha = tf.random.uniform((self.batch_size, 1, 1))
            return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

    # ----- [코드 61] encoder -----
    def build_encoder_layer(input_shape, encoder_reshape_shape):
        x = Input(shape=input_shape)
        model = tf.keras.models.Sequential([
            Bidirectional(LSTM(units=WIN_SIZE, return_sequences=True)),
            Flatten(),
            Dense(20),
            Reshape(target_shape=encoder_reshape_shape)])
        return Model(x, model(x))

    # ----- [코드 62] generator -----
    def build_generator_layer(input_shape, generator_reshape_shape):
        x = Input(shape=input_shape)
        model = tf.keras.models.Sequential([
            Flatten(),
            Dense(WIN_SIZE),
            Reshape(target_shape=generator_reshape_shape),
            Bidirectional(LSTM(units=64, return_sequences=True), merge_mode="concat"),
            Dropout(rate=0.2),
            UpSampling1D(size=1),
            Bidirectional(LSTM(units=64, return_sequences=True), merge_mode="concat"),
            Dropout(rate=0.2),
            TimeDistributed(Dense(FEATURES_DIM)),
            Activation(activation="tanh")])
        return Model(x, model(x))

    # ----- [코드 63] critic_x -----
    k_size = 5 if WIN_SIZE >= 30 else 2

    def build_critic_x_layer(input_shape):
        x = Input(shape=input_shape)
        layers = []
        for _ in range(4):
            layers += [Conv1D(filters=64, kernel_size=k_size),
                       LeakyReLU(alpha=0.2),
                       Dropout(rate=0.25)]
        layers += [Flatten(), Dense(units=1)]
        return Model(x, tf.keras.models.Sequential(layers)(x))

    # ----- [코드 64] critic_z -----
    def build_critic_z_layer(input_shape):
        x = Input(shape=input_shape)
        model = tf.keras.models.Sequential([
            Flatten(),
            Dense(units=100), LeakyReLU(alpha=0.2), Dropout(rate=0.2),
            Dense(units=100), LeakyReLU(alpha=0.2), Dropout(rate=0.2),
            Dense(units=1)])
        return Model(x, model(x))

    encoder = build_encoder_layer(PARAMS["encoder_input_shape"],
                                  PARAMS["encoder_reshape_shape"])
    generator = build_generator_layer(PARAMS["generator_input_shape"],
                                      PARAMS["generator_reshape_shape"])
    critic_x = build_critic_x_layer(PARAMS["critic_x_input_shape"])
    critic_z = build_critic_z_layer(PARAMS["critic_z_input_shape"])

    # ----- [코드 68] 입출력 구조 -----
    z = Input(shape=(latent_dim, 1))
    x = Input(shape=shape)
    x_ = generator(z)
    z_ = encoder(x)
    critic_x_model = Model(inputs=[x, z],
                           outputs=[critic_x(x), critic_x(x_),
                                    RandomWeightedAverage(batch_size)([x, x_])])
    critic_z_model = Model(inputs=[x, z],
                           outputs=[critic_z(z), critic_z(z_),
                                    RandomWeightedAverage(batch_size)([z, z_])])

    z_gen = Input(shape=(latent_dim, 1))
    x_gen = Input(shape=shape)
    x_gen_ = generator(z_gen)
    z_gen_ = encoder(x_gen)
    x_gen_rec = generator(z_gen_)
    encoder_generator_model = Model([x_gen, z_gen],
                                    [critic_x(x_gen_), critic_z(z_gen_), x_gen_rec])

    return dict(tf=tf, K=K, encoder=encoder, generator=generator,
                critic_x=critic_x, critic_z=critic_z,
                critic_x_model=critic_x_model, critic_z_model=critic_z_model,
                encoder_generator_model=encoder_generator_model)


def _is_keras3():
    # """Keras 3 (TF 2.16 이상) 여부. 체크포인트 확장자/저장 방식이 다르다."""
    try:
        import keras
        return int(keras.__version__.split(".")[0]) >= 3
    except Exception:
        return False


def _ckpt_names():
    # """Keras 3 는 save_weights 파일명이 반드시 '.weights.h5' 여야 한다."""
    base = ["critic_x_model", "critic_z_model", "encoder_generator_model"]
    ext = ".weights.h5" if _is_keras3() else ".h5"
    return [(b, b + ext) for b in base]


def load_weights_if_any(net):
    # """저장된 체크포인트가 있으면 불러온다. [코드 69]"""
    loaded = 0
    for key, fname in _ckpt_names():
        path = os.path.join(PATHS["ckpt"], fname)
        if os.path.isfile(path):
            net[key].load_weights(path)
            print("load weights:", fname)
            loaded += 1
    if loaded == 0:
        print("저장된 체크포인트 없음 - 무작위 초기화 상태에서 시작")
    return loaded


def save_weights(net):
    for key, fname in _ckpt_names():
        path = os.path.join(PATHS["ckpt"], fname)
        try:
            net[key].save_weights(path, save_format="h5")   # Keras 2
        except TypeError:
            net[key].save_weights(path)                      # Keras 3
    print("체크포인트 저장 완료:", PATHS["ckpt"])


def plot_networks(net):
    # """네트워크 구조도 저장 ([코드 69]). pydot/graphviz 없으면 건너뛴다."""
    if not PARAMS["plot_network"]:
        return
    try:
        from tensorflow.keras.utils import plot_model
        for key, fname in [("critic_x_model", "critic_x_model_tf2.png"),
                           ("critic_z_model", "critic_z_model_tf2.png"),
                           ("encoder_generator_model", "enc_gen_model_tf2.png")]:
            plot_model(net[key], to_file=os.path.join(PATHS["network"], fname),
                       show_shapes=True, expand_nested=True)
        print("네트워크 구조도 저장:", PATHS["network"])
    except Exception as e:
        print("네트워크 구조도 생략 (pydot/graphviz 미설치):", e)


def train_mtadgan(X, net, epochs=None):
    # """MTadGAN 학습 루프. [코드 70] ~ [코드 73]"""
    tf, K = net["tf"], net["K"]
    epochs = epochs if epochs is not None else PARAMS["epochs"]
    batch_size = PARAMS["batch_size"]
    n_critics = PARAMS["n_critic"]
    latent_dim = PARAMS["latent_dim"]
    shape = PARAMS["shape"]

    # TF 2.9+ 의 새 optimizer 는 하나의 optimizer 로 여러 모델의 가중치를
    # 번갈아 갱신하면 'cannot recognize variable' 오류가 난다.
    # (Keras 3 에는 tf.keras.optimizers.legacy 자체가 없다.)
    # 따라서 모델마다 별도의 optimizer 를 둔다 - Keras 2 / 3 모두 동작.
    optimizer_cx = tf.keras.optimizers.Adam(PARAMS["learning_rate"])
    optimizer_cz = tf.keras.optimizers.Adam(PARAMS["learning_rate"])
    optimizer_eg = tf.keras.optimizers.Adam(PARAMS["learning_rate"])

    def wasserstein_loss(y_true, y_pred):        # [코드 65]
        return K.mean(y_true * y_pred)

    def _gradient_penalty(critic, interpolated):
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = critic(interpolated, training=True)
        grads = gp_tape.gradient(pred, interpolated)[0]
        grads = tf.square(grads)
        ddx = tf.sqrt(1e-8 + tf.reduce_sum(grads, axis=np.arange(1, len(grads.shape))))
        return tf.reduce_mean((ddx - 1.0) ** 2)

    @tf.function
    def critic_x_train_on_batch(x, z, valid, fake):          # [코드 70]
        with tf.GradientTape() as tape:
            valid_x, fake_x, interpolated = net["critic_x_model"]([x, z], training=True)
            loss = tf.reduce_mean(wasserstein_loss(valid, valid_x))
            loss += tf.reduce_mean(wasserstein_loss(fake, fake_x))
            loss += _gradient_penalty(net["critic_x"], interpolated) * 10.0
        grads = tape.gradient(loss, net["critic_x_model"].trainable_weights)
        optimizer_cx.apply_gradients(zip(grads, net["critic_x_model"].trainable_weights))
        return loss

    @tf.function
    def critic_z_train_on_batch(x, z, valid, fake):          # [코드 71]
        with tf.GradientTape() as tape:
            valid_z, fake_z, interpolated = net["critic_z_model"]([x, z], training=True)
            loss = tf.reduce_mean(wasserstein_loss(valid, valid_z))
            loss += tf.reduce_mean(wasserstein_loss(fake, fake_z))
            loss += _gradient_penalty(net["critic_z"], interpolated) * 10.0
        grads = tape.gradient(loss, net["critic_z_model"].trainable_weights)
        optimizer_cz.apply_gradients(zip(grads, net["critic_z_model"].trainable_weights))
        return loss

    @tf.function
    def enc_gen_train_on_batch(x, z, valid):                 # [코드 72]
        with tf.GradientTape() as tape:
            fake_gen_x, fake_gen_z, x_gen_rec = net["encoder_generator_model"](
                [x, z], training=True)
            xs = tf.squeeze(x)
            xr = tf.squeeze(x_gen_rec)
            loss = tf.reduce_mean(wasserstein_loss(valid, fake_gen_x))
            loss += tf.reduce_mean(wasserstein_loss(valid, fake_gen_z))
            loss += tf.keras.losses.MSE(xs, xr) * 10
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, net["encoder_generator_model"].trainable_weights)
        optimizer_eg.apply_gradients(zip(grads, net["encoder_generator_model"].trainable_weights))
        return loss

    X = X.reshape((-1, shape[0], shape[1]))
    X_ = np.copy(X)
    fake = np.ones((batch_size, 1), dtype=np.float32)
    valid = -np.ones((batch_size, 1), dtype=np.float32)

    history = []
    for epoch in range(1, epochs + 1):
        np.random.shuffle(X_)
        epoch_g_loss, epoch_cx_loss, epoch_cz_loss = [], [], []
        minibatches_size = batch_size * n_critics
        num_minibatches = int(X_.shape[0] // minibatches_size)

        for i in range(num_minibatches):
            minibatch = X_[i * minibatches_size:(i + 1) * minibatches_size]

            net["critic_x"].trainable = True
            net["critic_z"].trainable = True
            net["generator"].trainable = False
            net["encoder"].trainable = False
            for j in range(n_critics):
                x = minibatch[j * batch_size:(j + 1) * batch_size]
                z = np.random.normal(size=(batch_size, latent_dim, 1))
                epoch_cx_loss.append(critic_x_train_on_batch(x, z, valid, fake))
                epoch_cz_loss.append(critic_z_train_on_batch(x, z, valid, fake))

            net["critic_x"].trainable = False
            net["critic_z"].trainable = False
            net["generator"].trainable = True
            net["encoder"].trainable = True
            epoch_g_loss.append(enc_gen_train_on_batch(x, z, valid))

        cx = float(np.mean(np.array(epoch_cx_loss)))
        cz = float(np.mean(np.array(epoch_cz_loss)))
        g = float(np.mean(np.array(epoch_g_loss)))
        history.append((epoch, cx, cz, g))
        print("Epoch: {}/{}, [Dx loss: {:.5f}] [Dz loss: {:.5f}] [G loss: {:.5f}]"
              .format(epoch, epochs, cx, cz, g))

    save_weights(net)
    return history


def predict(X, net):
    # """encoder-generator 재현값 y_hat 과 critic_x 점수를 반환한다. [코드 74]"""
    X = X.reshape((-1, PARAMS["shape"][0], PARAMS["shape"][1]))
    z_ = net["encoder"].predict(X)
    y_hat = net["generator"].predict(z_)
    critic = net["critic_x"].predict(X)
    return y_hat, critic


# =============================================================================
# [단계 ⑥] 이상 탐지 - Anomaly 클래스  ([코드 81])
# =============================================================================

class Anomaly(object):
    # """시계열 이상탐지 함수 모음.
    # 참고: https://arxiv.org/pdf/1802.04431.pdf
    # """

    # ---------- threshold ----------
    def _deltas(self, errors, epsilon, mean, std):
        below = errors[errors <= epsilon]
        if not len(below):
            return 0, 0
        return mean - below.mean(), std - below.std()

    def _count_above(self, errors, epsilon):
        above = errors > epsilon
        total_above = len(errors[above])
        above = pd.Series(above)
        change = above != above.shift(1)
        return total_above, sum(above & change)

    def _z_cost(self, z, errors, mean, std):
        epsilon = mean + z * std
        delta_mean, delta_std = self._deltas(errors, epsilon, mean, std)
        above, consecutive = self._count_above(errors, epsilon)
        numerator = -(delta_mean / mean + delta_std / std)
        denominator = above + consecutive ** 2
        if denominator == 0:
            return np.inf
        return numerator / denominator

    def _find_threshold(self, errors, z_range):
        mean, std = errors.mean(), errors.std()
        min_z, max_z = z_range
        best_z, best_cost = min_z, np.inf
        for z in range(min_z, max_z):
            best = fmin(self._z_cost, z, args=(errors, mean, std),
                        full_output=True, disp=False)
            z_opt, cost = best[0:2]
            if cost < best_cost:
                best_z, best_cost = z_opt[0], cost      # 가이드북의 best_cost 미갱신 버그 수정
        return mean + best_z * std

    def _fixed_threshold(self, errors, k=3.0):
        return errors.mean() + k * errors.std()

    # ---------- sequence ----------
    def _find_sequences(self, errors, epsilon, anomaly_padding):
        above = pd.Series(errors > epsilon)
        index_above = np.argwhere(above.values)
        for idx in index_above.flatten():
            above[max(0, idx - anomaly_padding):
                  min(idx + anomaly_padding + 1, len(above))] = True

        shift = above.shift(1).fillna(False)
        change = above != shift
        max_below = 0 if above.all() else max(errors[~above])

        index = above.index
        starts = index[above & change].tolist()
        ends = (index[~above & change] - 1).tolist()
        if len(ends) == len(starts) - 1:
            ends.append(len(above) - 1)
        return np.array([starts, ends]).T, max_below

    def _get_max_errors(self, errors, sequences, max_below):
        max_errors = [{"max_error": max_below, "start": -1, "stop": -1}]
        for start, stop in sequences:
            max_errors.append({"start": start, "stop": stop,
                               "max_error": max(errors[start:stop + 1])})
        max_errors = pd.DataFrame(max_errors).sort_values("max_error", ascending=False)
        return max_errors.reset_index(drop=True)

    def _prune_anomalies(self, max_errors, min_percent):
        next_error = max_errors["max_error"].shift(-1).iloc[:-1]
        max_error = max_errors["max_error"].iloc[:-1]
        increase = (max_error - next_error) / max_error
        too_small = increase < min_percent
        last_index = -1 if too_small.all() else max_error[~too_small].index[-1]
        return max_errors[["start", "stop", "max_error"]].iloc[0:last_index + 1].values

    def _compute_scores(self, pruned_anomalies, errors, threshold, window_start):
        anomalies = []
        denominator = errors.mean() + errors.std()
        for row in pruned_anomalies:
            score = (row[2] - threshold) / denominator
            anomalies.append([row[0] + window_start, row[1] + window_start, score])
        return anomalies

    def _merge_sequences(self, sequences):
        if len(sequences) == 0:
            return np.array([])
        sorted_sequences = sorted(sequences, key=lambda entry: entry[0])
        new_sequences = [sorted_sequences[0]]
        score = [sorted_sequences[0][2]]
        weights = [sorted_sequences[0][1] - sorted_sequences[0][0]]
        for sequence in sorted_sequences[1:]:
            prev = new_sequences[-1]
            if sequence[0] <= prev[1] + 1:
                score.append(sequence[2])
                weights.append(sequence[1] - sequence[0])
                if sum(weights) == 0:
                    weighted_average = float(np.mean(score))
                else:
                    weighted_average = float(np.average(score, weights=weights))
                new_sequences[-1] = (prev[0], max(prev[1], sequence[1]), weighted_average)
            else:
                score = [sequence[2]]
                weights = [sequence[1] - sequence[0]]
                new_sequences.append(sequence)
        return np.array(new_sequences)

    def _find_window_sequences(self, window, z_range, anomaly_padding, min_percent,
                               window_start, fixed_threshold):
        if fixed_threshold:
            threshold = self._fixed_threshold(window)
        else:
            threshold = self._find_threshold(window, z_range)
        window_sequences, max_below = self._find_sequences(window, threshold, anomaly_padding)
        max_errors = self._get_max_errors(window, window_sequences, max_below)
        pruned = self._prune_anomalies(max_errors, min_percent)
        return self._compute_scores(pruned, window, threshold, window_start)

    def find_anomalies(self, errors, index, z_range=(0, 10), window_size=None,
                       window_size_portion=None, window_step_size=None,
                       window_step_size_portion=None, min_percent=0.1,
                       anomaly_padding=50, lower_threshold=False,
                       fixed_threshold=True, verbose=False):
        # """이상 구간(start, end, score) 리스트를 찾는다. [코드 81 (10)]"""
        window_size = window_size or len(errors)
        if window_size_portion:
            window_size = np.ceil(len(errors) * window_size_portion).astype("int")
        window_step_size = window_step_size or window_size
        if window_step_size_portion:
            window_step_size = np.ceil(window_size * window_step_size_portion).astype("int")

        window_start, window_end = 0, 0
        sequences = []
        while window_end < len(errors):
            window_end = window_start + window_size
            window = errors[window_start:window_end]
            sequences.extend(self._find_window_sequences(
                window, z_range, anomaly_padding, min_percent, window_start, fixed_threshold))
            if lower_threshold:
                mean = window.mean()
                inverted = mean - (window - mean)
                sequences.extend(self._find_window_sequences(
                    inverted, z_range, anomaly_padding, min_percent,
                    window_start, fixed_threshold))
            window_start += window_step_size

        sequences = self._merge_sequences(sequences)
        anomalies = []
        for start, stop, score in sequences:
            if verbose:
                print("start %d  stop %d  score %.4f" % (start, stop, score))
            anomalies.append([index[int(start)], index[int(stop)], score])
        return anomalies

    # ---------- score ----------
    def _compute_critic_score(self, critics, smooth_window):
        # """판별자 점수로부터 이상 점수 배열 계산. [코드 81 (11)]"""
        critics = np.asarray(critics)
        l_q, u_q = np.quantile(critics, 0.25), np.quantile(critics, 0.75)
        in_range = np.logical_and(critics >= l_q, critics <= u_q)
        critic_mean = np.mean(critics[in_range])
        critic_std = np.std(critics)
        z_scores = np.absolute((critics - critic_mean) / critic_std) + 1
        return pd.Series(z_scores).rolling(
            smooth_window, center=True, min_periods=smooth_window // 2).mean().values

    def _point_wise_error(self, y, y_hat):
        return np.sum(abs(y - y_hat), axis=-1)

    def _area_error(self, y, y_hat, score_window=10):
        smooth_y = pd.Series(y).rolling(
            score_window, center=True,
            min_periods=score_window // 2).apply(integrate.trapezoid)
        smooth_y_hat = pd.Series(y_hat).rolling(
            score_window, center=True,
            min_periods=score_window // 2).apply(integrate.trapezoid)
        return abs(smooth_y - smooth_y_hat)

    def _dtw_error(self, y, y_hat, score_window=10):
        from pyts.metrics import dtw
        length_dtw = (score_window // 2) * 2 + 1
        half = length_dtw // 2
        y_pad = np.pad(y, (half, half), "constant", constant_values=(0, 0))
        y_hat_pad = np.pad(y_hat, (half, half), "constant", constant_values=(0, 0))
        similarity = []
        i = 0
        while i < len(y) - length_dtw:
            similarity.append(dtw(y_pad[i:i + length_dtw].flatten(),
                                  y_hat_pad[i:i + length_dtw].flatten()))
            i += 1
        return [0] * half + similarity + [0] * (len(y) - len(similarity) - half)

    def _reconstruction_errors(self, y, y_hat, step_size=1, score_window=10,
                               smoothing_window=0.01, smooth=True,
                               rec_error_type="point"):
        # """재현 오차 배열 계산. [코드 81 (16)]"""
        if isinstance(smoothing_window, float):
            smoothing_window = min(math.trunc(len(y) * smoothing_window), 200)

        true = [y[i][0] for i in range(len(y))]
        for it in range(len(y[-1]) - 1):            # 마지막 윈도우의 잔여 구간 포함
            true.append(y[-1][it + 1])

        predictions, predictions_vs = [], []
        pred_length = y_hat.shape[1]
        num_errors = y_hat.shape[1] + step_size * (y_hat.shape[0] - 1)
        for i in range(num_errors):
            intermediate = []
            for j in range(max(0, i - num_errors + pred_length), min(i + 1, pred_length)):
                intermediate.append(y_hat[i - j, j])
            if intermediate:
                arr = np.asarray(intermediate)
                predictions.append(np.average(intermediate, axis=0))
                predictions_vs.append([[np.min(arr), np.percentile(arr, 25),
                                        np.percentile(arr, 50), np.percentile(arr, 75),
                                        np.max(arr)]])

        true = np.asarray(true)
        predictions = np.asarray(predictions)
        predictions_vs = np.asarray(predictions_vs)

        if rec_error_type.lower() == "point":
            errors = self._point_wise_error(true, predictions)
        elif rec_error_type.lower() == "area":
            errors = self._area_error(true, predictions, score_window)
        elif rec_error_type.lower() == "dtw":
            errors = self._dtw_error(true, predictions, score_window)
        else:
            raise ValueError("unknown rec_error_type: %s" % rec_error_type)

        if smooth:
            errors = pd.Series(errors).rolling(
                smoothing_window, center=True,
                min_periods=smoothing_window // 2).mean().values
        return errors, predictions_vs

    def score_anomalies(self, y, y_hat, critic, index, score_window=10,
                        critic_smooth_window=None, error_smooth_window=None,
                        smooth=True, rec_error_type="point", comb="mult",
                        lambda_rec=0.5):
        # """판별자 점수 + 재현 오차를 결합한 최종 이상 점수. [코드 81 (17)]"""
        critic_smooth_window = critic_smooth_window or max(math.trunc(y.shape[0] * 0.01), 1)
        error_smooth_window = error_smooth_window or max(math.trunc(y.shape[0] * 0.01), 1)
        step_size = 1

        true_index = list(index)
        true = [y[i][0] for i in range(len(y))]
        for it in range(len(y[-1]) - 1):
            true.append(y[-1][it + 1])
            true_index.append(index[-1] + it + 1)
        true_index = np.array(true_index)

        critic_extended = []
        for c in critic:
            critic_extended.extend(np.repeat(c, y_hat.shape[1]).tolist())
        critic_extended = np.asarray(critic_extended).reshape((-1, y_hat.shape[1]))

        critic_kde_max = []
        pred_length = y_hat.shape[1]
        num_errors = y_hat.shape[1] + step_size * (y_hat.shape[0] - 1)
        for i in range(num_errors):
            critic_intermediate = []
            for j in range(max(0, i - num_errors + pred_length), min(i + 1, pred_length)):
                critic_intermediate.append(critic_extended[i - j, j])
            if len(critic_intermediate) > 1:
                discr = np.asarray(critic_intermediate)
                try:
                    critic_kde_max.append(
                        discr[np.argmax(stats.gaussian_kde(discr)(critic_intermediate))])
                except (np.linalg.LinAlgError, ValueError):
                    critic_kde_max.append(np.median(discr))
            else:
                critic_kde_max.append(np.median(np.asarray(critic_intermediate)))

        critic_scores = self._compute_critic_score(critic_kde_max, critic_smooth_window)
        rec_scores, predictions = self._reconstruction_errors(
            y, y_hat, step_size, score_window, error_smooth_window, smooth, rec_error_type)
        rec_scores = stats.zscore(rec_scores)
        rec_scores = np.clip(rec_scores, a_min=0, a_max=None) + 1

        if comb == "mult":
            final_scores = np.multiply(critic_scores, rec_scores)
        elif comb == "sum":
            final_scores = (1 - lambda_rec) * (critic_scores - 1) + lambda_rec * (rec_scores - 1)
        elif comb == "rec":
            final_scores = rec_scores
        else:
            raise ValueError('comb must be "mult", "sum", or "rec"')

        true = [[t] for t in true]
        return final_scores, true_index, true, predictions


# =============================================================================
# [단계 ⑦~⑧] 테스트셋 결과 분석 및 알고리즘 평가  ([코드 84] ~ [코드 86])
# =============================================================================

def evaluate(final_scores, anomalies, labels, tag, X_windows):
    # """Accuracy / Precision / Recall / F-score 계산 및 이상구간 비교. [코드 84], [코드 85]"""
    pred_length = len(final_scores)

    avg = float(np.average(final_scores))
    sigma = math.sqrt(float(np.sum((final_scores - avg) ** 2)) / pred_length)
    z_score = (final_scores - avg) / sigma          # [단계 ⑦] Z score

    pred_bin = [0] * pred_length
    for a in anomalies:
        s, e = int(a[0]), int(a[1])
        for k in range(max(0, s - 1), min(e, pred_length)):
            pred_bin[k] = 1

    n_pred = min(pred_length, len(labels))
    pred = np.array(pred_bin[:n_pred])
    gt = np.array(labels[:n_pred])

    tp = int(np.sum((pred == 1) & (gt == 1)))
    tn = int(np.sum((pred == 0) & (gt == 0)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))

    accuracy = (tp + tn) / n_pred
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    print("\n" + "=" * 62)
    print("[단계 ⑧] 알고리즘 평가 - %s" % tag)
    print("=" * 62)
    print("TP=%d, TN=%d, FP=%d, FN=%d (n=%d)" % (tp, tn, fp, fn, n_pred))
    print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}"
          .format(accuracy, precision, recall, f1))

    # ----- 이상 구간(sequence) 리스트 [코드 85] -----
    def to_sequences(binary):
        seqs, begin, inside = [], 0, False
        for k, v in enumerate(binary):
            if v == 1 and not inside:
                begin, inside = k, True
            elif v == 0 and inside:
                seqs.append((begin, k - 1))
                inside = False
        if inside:
            seqs.append((begin, len(binary) - 1))
        return seqs

    anomaly_gt = to_sequences(gt)
    anomaly_pred = to_sequences(pred)
    print("\ngt   구간:", anomaly_gt)
    print("pred 구간:", anomaly_pred)

    # ----- 결과 그래프 [코드 86] -----
    plot_result(X_windows, z_score, [anomaly_gt, anomaly_pred], n_pred, tag)

    return dict(accuracy=accuracy, precision=precision, recall=recall, f_score=f1,
                tp=tp, tn=tn, fp=fp, fn=fn, n=n_pred,
                anomaly_gt=anomaly_gt, anomaly_pred=anomaly_pred)


def plot_result(X_windows, z_score, anomaly_sets, length_anom, tag):
    # """PCA 성분 / Z score / 실제·예측 이상구간 비교 그래프. [코드 86]"""
    register_matplotlib_converters()
    max_len = min(length_anom - 10, len(X_windows))
    if max_len <= 0:
        return

    time = range(max_len)
    z2 = z_score[:max_len]
    signal = np.array([X_windows[k, 1] for k in range(max_len)])

    fig = plt.figure(figsize=(30, 12))
    fig.add_subplot(111)
    plt.plot(time, 3 * signal[:, 0], label="3*PCA1")
    plt.plot(time, 3 * signal[:, 1], label="3*PCA2")
    plt.plot(time, z2, label="Z score")
    plt.legend(loc=0, fontsize=30)

    colors = ["red", "blue"]
    for i, anomaly in enumerate(anomaly_sets):
        for t1, t2 in anomaly:
            plt.axvspan(t1, t2, color=colors[i % len(colors)], alpha=0.2)

    plt.title("%s : Red = True Anomaly, Blue = Predicted Anomaly" % tag, size=34)
    plt.ylabel("PCA1, PCA2, Z_score", size=30)
    plt.xlabel("Time", size=30)
    plt.xticks(size=26)
    plt.yticks(size=26)
    plt.xlim([0, max_len - 1])
    out = os.path.join(PATHS["output"], "anomaly_%s.png" % tag.replace(" ", "_"))
    plt.savefig(out, bbox_inches="tight", dpi=70)
    plt.close()
    print("결과 그래프 저장:", out)


# =============================================================================
# 메인 파이프라인
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="배터리팩 품질보증 - MTadGAN 분석실습 (KAMP 가이드북)")
    parser.add_argument("--mode", default="all",
                        choices=["preprocess", "train", "predict", "all"])
    parser.add_argument("--train-file", default="1000_chg.csv")
    parser.add_argument("--test-file", default="Test07_NG_dchg.csv")
    parser.add_argument("--test-label", default="Test07_NG_dchg_Label.csv")
    parser.add_argument("--epochs", type=int, default=PARAMS["epochs"])
    parser.add_argument("--aggregate-interval", type=int, default=1)
    parser.add_argument("--rec-error-type", default="point",
                        choices=["point", "area", "dtw"])
    parser.add_argument("--comb", default="mult", choices=["mult", "sum", "rec"])
    parser.add_argument("--anomaly-padding", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None,
                        help="난수 시드 고정 (GAN 학습은 시드마다 결과 편차가 큼)")
    args_cli = parser.parse_args()

    if args_cli.seed is not None:
        os.environ["PYTHONHASHSEED"] = str(args_cli.seed)
        np.random.seed(args_cli.seed)
        try:
            import tensorflow as tf
            tf.random.set_seed(args_cli.seed)
        except Exception:
            pass
        print("random seed =", args_cli.seed)

    ensure_dirs()
    PARAMS["epochs"] = args_cli.epochs

    print("=" * 62)
    print("배터리팩 품질보증 AI - MTadGAN")
    print("win_size = %d, features_dim = %d, epochs = %d"
          % (WIN_SIZE, FEATURES_DIM, PARAMS["epochs"]))
    print("=" * 62)

    # ---------- [단계 ①] 데이터 준비 ----------
    if args_cli.mode in ("preprocess", "all"):
        run_preprocess(args_cli.train_file, args_cli.test_file, args_cli.test_label)
    if args_cli.mode == "preprocess":
        return

    net = build_networks()
    load_weights_if_any(net)
    plot_networks(net)

    # ---------- [단계 ②③⑤] 학습 ----------
    if args_cli.mode in ("train", "all"):
        train_args = Args(
            signal_file=os.path.join(PATHS["pre_train"], args_cli.train_file),
            timest_form=0, anomaly_file="", mode="train",
            aggregate_interval=args_cli.aggregate_interval, regate_interval=1)
        print("\n[학습 데이터] %s" % train_args.signal_file)

        df_train = pd.read_csv(train_args.signal_file)
        X_train, _, _, _ = build_model_input(
            df_train, train_args.aggregate_interval, tag="train")
        train_mtadgan(X_train, net, epochs=PARAMS["epochs"])

    # ---------- [단계 ⑥⑦⑧] 테스트 ----------
    if args_cli.mode in ("predict", "all"):
        test_args = Args(
            signal_file=os.path.join(PATHS["pre_test"], args_cli.test_file),
            timest_form=0,
            anomaly_file=os.path.join(PATHS["pre_test"], args_cli.test_label),
            mode="predict",
            aggregate_interval=args_cli.aggregate_interval, regate_interval=1)
        print("\n[테스트 데이터] %s" % test_args.signal_file)
        print("[라벨 데이터]   %s" % test_args.anomaly_file)

        df_test = pd.read_csv(test_args.signal_file)
        X_test, _, X_index, _ = build_model_input(
            df_test, test_args.aggregate_interval, tag="test")

        y_hat, critic = predict(X_test, net)             # [코드 82]
        print("y_hat.shape =", y_hat.shape, " critic.shape =", critic.shape)

        anomaly = Anomaly()
        final_scores, true_index, _, _ = anomaly.score_anomalies(
            X_test, y_hat, critic, X_index,
            rec_error_type=args_cli.rec_error_type, comb=args_cli.comb)
        final_scores = np.array(final_scores)
        print("final_scores.shape =", final_scores.shape)

        anomalies = anomaly.find_anomalies(
            final_scores, true_index, anomaly_padding=args_cli.anomaly_padding)
        print("탐지된 이상 구간 수:", len(anomalies))

        known = pd.read_csv(test_args.anomaly_file)
        labels = known["label"].values

        tag = args_cli.test_file.replace(".csv", "")
        result = evaluate(final_scores, anomalies, labels, tag, X_test)

        pd.DataFrame([{k: v for k, v in result.items()
                       if k not in ("anomaly_gt", "anomaly_pred")}]).to_csv(
            os.path.join(PATHS["output"], "metrics_%s.csv" % tag), index=False)
        pd.DataFrame({"index": true_index[:len(final_scores)],
                      "anomaly_score": final_scores}).to_csv(
            os.path.join(PATHS["output"], "scores_%s.csv" % tag), index=False)
        print("지표/점수 저장:", PATHS["output"])


if __name__ == "__main__":
    main()
