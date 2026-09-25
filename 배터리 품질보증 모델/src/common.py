# -*- coding: utf-8 -*-
"""
공통 모듈 - 모델별 비교 스크립트들이 공유하는 데이터 로딩/정규화/평가 함수.

모델 하나당 파일 하나(model_*.py)로 짜고, 전처리·정규화·평가 방식만 여기서
통일한다. 이렇게 해야 "같은 조건"에서 비교한 게 되고, 각 파일이 서로 다른
방식으로 전처리하면 숫자를 나란히 놓고 비교할 수 없다.

battery_quality_mtadgan.py 의 removeConstant/handleMissingValue/
select_voltage_temperature 를 재사용한다.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import battery_quality_mtadgan as bq


# =============================================================================
# 파일 목록 - 라벨이 있는 9개 파일을 학습용 5개 / 평가용 4개로 명확히 분리한다.
# 비지도 모델(라벨 불필요)은 9개 전체에서 평가하고,
# 지도학습 모델(RandomForest, LogisticRegression)은 학습 5개 -> 평가 4개로 나눠 쓴다.
# =============================================================================

SUP_TRAIN_NG = [("Test05_NG_chg", "Test05_NG_chg.csv", "Test05_NG_chg_Label.csv"),
                ("Test06_NG_chg", "Test06_NG_chg.csv", "Test06_NG_chg_Label.csv"),
                ("Test08_NG_chg", "Test08_NG_chg.csv", "Test08_NG_chg_Label.csv")]
SUP_TRAIN_OK = [("Test01_OK_chg", "Test01_OK_chg.csv", "Test01_OK_chg_Label.csv"),
                ("Test02_OK_dchg", "Test02_OK_dchg.csv", "Test02_OK_dchg_Label.csv")]
EVAL_NG = [("Test07_NG_dchg", "Test07_NG_dchg.csv", "Test07_NG_dchg_Label.csv"),
           ("Test09_NG_dchg", "Test09_NG_dchg.csv", "Test09_NG_dchg_Label.csv")]
EVAL_OK = [("Test03_OK_chg", "Test03_OK_chg.csv", "Test03_OK_chg_Label.csv"),
           ("Test04_OK_dchg", "Test04_OK_dchg.csv", "Test04_OK_dchg_Label.csv")]
ALL_LABELED = SUP_TRAIN_NG + SUP_TRAIN_OK + EVAL_NG + EVAL_OK

NG_TAGS_ALL = [t[0] for t in SUP_TRAIN_NG + EVAL_NG]      # 비지도 평균 낼 때 쓰는 NG 5개
OK_TAGS_ALL = [t[0] for t in SUP_TRAIN_OK + EVAL_OK]      # 오탐률 확인용 OK 4개
SUP_TRAIN_TAGS = [t[0] for t in SUP_TRAIN_NG + SUP_TRAIN_OK]
SUP_EVAL_TAGS = [t[0] for t in EVAL_NG + EVAL_OK]


def get_mode(tag):
    """파일명에 dchg가 있으면 방전, 없으면 충전."""
    return "dchg" if "dchg" in tag else "chg"


# =============================================================================
# 데이터 로딩 - 원본 csv -> 상수열 제거 -> 결측치 처리 -> 전압/온도 컬럼만 추출
# =============================================================================

def _resolve_path(dir_path, name):
    """파일명이 정확히 일치하지 않아도 찾아준다.
    (예: 실제 파일이 "Test03_OK_chg .csv"처럼 확장자 앞에 공백이 들어있는 경우)"""
    candidates = [name]
    stem, ext = os.path.splitext(name)
    candidates.append(stem + " " + ext)   # 공백이 껴 있는 경우
    candidates.append(stem.rstrip() + ext)  # 반대로 공백이 없어야 하는 경우
    for cand in candidates:
        p = os.path.join(dir_path, cand)
        if os.path.exists(p):
            return p
    if os.path.isdir(dir_path):
        target = name.replace(" ", "").lower()
        for fname in os.listdir(dir_path):
            if fname.replace(" ", "").lower() == target:
                return os.path.join(dir_path, fname)
    return os.path.join(dir_path, name)  # 못 찾으면 원래 경로 반환 (호출부에서 존재 여부 재확인)


def load_signal(raw_path):
    df = pd.read_csv(raw_path)
    df1 = bq.removeConstant(df, 1)
    df2 = bq.handleMissingValue(df1)
    return bq.select_voltage_temperature(df2)


def load_signal_and_label(raw_path, label_path):
    feat = load_signal(raw_path)
    label = pd.read_csv(label_path)["label"].values
    n = min(len(feat), len(label))
    return feat.iloc[:n].reset_index(drop=True), np.asarray(label[:n])


def align_columns(dfs):
    """여러 파일의 전압/온도 컬럼 교집합으로 통일 (배터리팩마다 상수열이 달라 컬럼수가 다름)."""
    common = set(dfs[0].columns)
    for d in dfs[1:]:
        common &= set(d.columns)
    common = sorted(common)
    return [d[common] for d in dfs], common


def normalize_with_ref_scale(df, ref_std, floor_frac=0.1):
    """파일 자신의 평균/표준편차로 정규화하되, 표준편차는 학습 파일(ref_std)의
    floor_frac(기본 10%) 밑으로는 내려가지 않게 바닥을 깐다.

    이유: 신호가 거의 평평한 파일에서 자기 자신의 std로만 나누면 미세한 측정
    잡음까지 z-score로 크게 확대되어 전부 이상으로 오판된다. 반대로 학습 파일의
    std를 그대로 강제로 씌우면 배터리팩 개체차 때문에 다른 팩에는 안 맞는다.
    절충: 파일 자신의 변동을 쓰되, 최소한의 바닥만 학습 파일에서 빌려온다.
    """
    mean = df.mean()
    own_std = df.std()
    floor = ref_std.reindex(df.columns).replace(0, 1e-8) * floor_frac
    std = np.maximum(own_std.values, floor.values)
    std = pd.Series(std, index=df.columns).replace(0, 1e-8)
    return (df - mean) / std


def discover_train_files(train_dir):
    """raw_data/train 폴더 안에서 "*_chg.csv" / "*_dchg.csv" 패턴의 파일들을
    전부 찾아 모드별로 나눠준다 (1000_chg.csv, 1001_chg.csv, ... 처럼 여러 팩).
    """
    chg_files, dchg_files = [], []
    if not os.path.isdir(train_dir):
        return chg_files, dchg_files
    for fname in sorted(os.listdir(train_dir)):
        lower = fname.strip().lower()
        if lower.endswith("_dchg.csv"):
            dchg_files.append(fname)
        elif lower.endswith("_chg.csv"):
            chg_files.append(fname)
    return chg_files, dchg_files


def load_train_pool(train_dir, filenames, mode, limit=None):
    """모드별 학습 파일들을 전부 불러와 각자 자기 평균으로 중심화(centering)한 뒤
    하나로 이어붙인다.

    팩(배터리 개체)마다 전압/온도의 절대적인 기준선이 달라서, 원본 그대로 이어
    붙이면 "팩 간 기준선 차이"가 "정상 변동"으로 잘못 섞여 들어간다. 각 파일을
    자기 자신의 평균으로 먼저 중심화하면 기준선 차이는 제거되고, 여러 팩의
    "변동 패턴"만 모여서 훨씬 더 대표성 있는 정상 분포를 만들 수 있다.
    """
    if limit is not None:
        filenames = filenames[:limit]
    pieces = []
    for fname in filenames:
        path = os.path.join(train_dir, fname)
        try:
            feat = load_signal(path)
        except Exception as e:
            print("  [%s] 로딩 실패, 건너뜀: %s (%s)" % (mode, fname, e))
            continue
        centered = feat - feat.mean()
        pieces.append(centered)
        print("  [%s] 학습팩 로딩: %-20s shape=%s" % (mode, fname, feat.shape))
    if not pieces:
        raise RuntimeError("모드 '%s'에 대해 로딩된 학습 파일이 하나도 없습니다." % mode)
    return pieces


CACHE_PATH_TEMPLATE = "_cache_common_data_%s.pkl"


def load_all_normalized(train_limit=None, use_cache=True, force_reload=False):
    """raw_data/train 안의 모든 팩(1000_chg/1000_dchg, 1001_chg/1001_dchg, ...)을
    모드별로 모아 정상 분포를 만들고, 라벨 있는 9개 파일을 정규화까지 끝낸 상태로 반환한다.

    train_limit: 모드별로 사용할 학습 파일 개수를 제한 (None이면 전부 사용).
                 팩 수가 많아 시간이 오래 걸릴 때 빠르게 테스트해보기 위한 옵션.
    use_cache:   결과를 output/ 밑에 캐시해서, 같은 조건으로 여러 model_*.py를
                 실행할 때 매번 처음부터 다시 로딩하지 않도록 한다.

    반환:
      train_feats: {"chg": 정규화된 학습 데이터 풀, "dchg": 정규화된 학습 데이터 풀}
      norm_data:   {tag: (정규화된 feature df, label array, mode)}
    """
    bq.ensure_dirs()
    cache_key = "all" if train_limit is None else str(train_limit)
    cache_path = os.path.join(bq.PATHS["output"], CACHE_PATH_TEMPLATE % cache_key)
    if use_cache and not force_reload and os.path.exists(cache_path):
        try:
            import pickle
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            tf_check = cached["train_feats"]
            required_keys = {"chg", "dchg", "chg_files", "dchg_files"}
            if not required_keys.issubset(tf_check.keys()):
                raise ValueError("캐시 구조가 예전 버전입니다 (필요한 키 없음).")
            if tf_check["chg"].shape[1] == 0 or tf_check["dchg"].shape[1] == 0:
                raise ValueError("캐시된 학습 데이터의 컬럼 수가 0입니다 (이전 버전의 버그로 생성된 캐시).")
            print("캐시에서 불러옴: %s" % cache_path)
            return tf_check, cached["norm_data"]
        except Exception as e:
            print("캐시가 유효하지 않아 다시 계산합니다: %s" % e)

    chg_files, dchg_files = discover_train_files(bq.PATHS["raw_train"])
    print("학습 파일 발견: chg %d개, dchg %d개" % (len(chg_files), len(dchg_files)))
    if not chg_files and not dchg_files:
        raise RuntimeError("raw_data/train 폴더에서 *_chg.csv / *_dchg.csv 파일을 찾지 못했습니다: %s"
                            % bq.PATHS["raw_train"])

    train_pool_chg = load_train_pool(bq.PATHS["raw_train"], chg_files, "chg", limit=train_limit)
    train_pool_dchg = load_train_pool(bq.PATHS["raw_train"], dchg_files, "dchg", limit=train_limit)

    labeled = {}
    for tag, raw_name, label_name in ALL_LABELED:
        raw_path = _resolve_path(bq.PATHS["raw_test"], raw_name)
        label_path = _resolve_path(bq.PATHS["pre_test"], label_name)
        if not os.path.exists(raw_path):
            print("  건너뜀 (원본 없음): %s" % tag)
            continue
        feat, label = load_signal_and_label(raw_path, label_path)
        labeled[tag] = (feat, label)
        print("  %-16s shape=%s  anomaly=%d/%d" % (tag, feat.shape, label.sum(), len(label)))

    # 컬럼 기준은 "라벨 있는 9개 파일"의 교집합으로 고정한다. (기존에 검증된 208개)
    # 학습 팩이 ~100개나 되면 그중 하나라도 컬럼 구성이 특이한 파일이 섞여 있을 때
    # 전체 교집합을 그 파일 하나가 0으로 만들어버릴 수 있어서, 라벨 파일 기준으로
    # 고정해두고 학습 팩들은 그 기준에 맞춰 재정렬(부족하면 0으로 채움)한다.
    labeled_dfs = [v[0] for v in labeled.values()]
    if not labeled_dfs:
        raise RuntimeError("라벨 있는 파일을 하나도 찾지 못해 기준 컬럼을 정할 수 없습니다.")
    _, common_cols = align_columns(labeled_dfs)
    print("공통 컬럼수 (라벨 파일 기준): %d" % len(common_cols))

    for tag in labeled:
        feat, label = labeled[tag]
        labeled[tag] = (feat[common_cols], label)

    def _reindex_train_pool(pool, tag_prefix):
        kept = []
        for i, df in enumerate(pool):
            overlap = len(set(df.columns) & set(common_cols))
            coverage = overlap / len(common_cols)
            if coverage < 0.7:
                print("  [%s] 컬럼 겹침 부족(%.0f%%)으로 학습팩 제외: 인덱스 %d" %
                      (tag_prefix, coverage * 100, i))
                continue
            kept.append(df.reindex(columns=common_cols, fill_value=0.0))
        return kept

    train_pool_chg = _reindex_train_pool(train_pool_chg, "chg")
    train_pool_dchg = _reindex_train_pool(train_pool_dchg, "dchg")
    if not train_pool_chg:
        raise RuntimeError("컬럼 기준을 만족하는 chg 학습 팩이 하나도 없습니다.")
    if not train_pool_dchg:
        raise RuntimeError("컬럼 기준을 만족하는 dchg 학습 팩이 하나도 없습니다.")

    n_chg, n_dchg = len(train_pool_chg), len(train_pool_dchg)

    train_chg_raw = pd.concat(train_pool_chg, axis=0, ignore_index=True)
    train_dchg_raw = pd.concat(train_pool_dchg, axis=0, ignore_index=True)
    print("학습 풀 크기: chg=%s (%d개 팩), dchg=%s (%d개 팩)" %
          (train_chg_raw.shape, n_chg, train_dchg_raw.shape, n_dchg))

    ref_std = {"chg": train_chg_raw.std(), "dchg": train_dchg_raw.std()}
    train_feats = {"chg": normalize_with_ref_scale(train_chg_raw, ref_std["chg"]),
                   "dchg": normalize_with_ref_scale(train_dchg_raw, ref_std["dchg"])}

    # 시퀀스 모델(LSTM Autoencoder, USAD)은 팩 경계를 넘어 윈도우를 만들면 안 되므로
    # 팩별로 나눠진 정규화 결과도 같이 넣어둔다 (train_feats["chg_files"] 처럼 접근).
    train_feats["chg_files"] = [normalize_with_ref_scale(f, ref_std["chg"]) for f in train_pool_chg]
    train_feats["dchg_files"] = [normalize_with_ref_scale(f, ref_std["dchg"]) for f in train_pool_dchg]

    norm_data = {}
    for tag, (feat, label) in labeled.items():
        mode = get_mode(tag)
        norm_data[tag] = (normalize_with_ref_scale(feat, ref_std[mode]), label, mode)

    if use_cache:
        try:
            import pickle
            os.makedirs(bq.PATHS["output"], exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump({"train_feats": train_feats, "norm_data": norm_data}, f)
            print("캐시 저장: %s" % cache_path)
        except Exception as e:
            print("캐시 저장 실패 (무시하고 계속): %s" % e)

    return train_feats, norm_data


# =============================================================================
# 공통 평가 (point-wise) - 모든 모델이 이 함수로 Accuracy/Precision/Recall/F-score를 낸다
# =============================================================================

def evaluate_pointwise(pred, gt):
    pred = np.asarray(pred).astype(int)
    gt = np.asarray(gt).astype(int)
    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]

    tp = int(np.sum((pred == 1) & (gt == 1)))
    tn = int(np.sum((pred == 0) & (gt == 0)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))

    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(n=n, tp=tp, tn=tn, fp=fp, fn=fn,
                accuracy=round(accuracy, 4), precision=round(precision, 4),
                recall=round(recall, 4), f_score=round(f1, 4))


# =============================================================================
# battery_quality_mtadgan.py 의 evaluate()/plot_result() 와 같은 형식으로 결과를
# 출력하는 공통 평가 함수. MTadGAN과 나란히 비교하려면 출력 형식이 같아야 하므로
# 다른 모델 파일들도 이 함수를 쓴다.
# =============================================================================

def to_sequences(binary):
    """0/1 배열을 (시작, 끝) 구간 리스트로 변환한다. (battery_quality_mtadgan.evaluate와 동일)"""
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


def evaluate_and_report(pred, gt, score, tag, model_name, plot=True):
    """battery_quality_mtadgan.evaluate()와 같은 형식(TP/TN/FP/FN, Accuracy/Precision/
    Recall/F-score, gt/pred 구간, 그래프)으로 출력한다.

    pred, gt: 0/1 배열 (동일 길이로 맞춰짐)
    score:    이상 점수(연속값) - Z score로 정규화해서 그래프에 쓴다.
    """
    pred = np.asarray(pred).astype(int)
    gt = np.asarray(gt).astype(int)
    score = np.asarray(score, dtype=float)
    n = min(len(pred), len(gt), len(score))
    pred, gt, score = pred[:n], gt[:n], score[:n]

    tp = int(np.sum((pred == 1) & (gt == 1)))
    tn = int(np.sum((pred == 0) & (gt == 0)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))

    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    print("\n" + "=" * 62)
    print("[%s] 알고리즘 평가 - %s" % (model_name, tag))
    print("=" * 62)
    print("TP=%d, TN=%d, FP=%d, FN=%d (n=%d)" % (tp, tn, fp, fn, n))
    print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}"
          .format(accuracy, precision, recall, f1))

    anomaly_gt = to_sequences(gt)
    anomaly_pred = to_sequences(pred)
    print("\ngt   구간:", anomaly_gt)
    print("pred 구간:", anomaly_pred)

    if plot:
        std = score.std()
        z_score = (score - score.mean()) / std if std > 1e-12 else np.zeros_like(score)
        _plot_anomaly_result(z_score, [anomaly_gt, anomaly_pred], n, tag, model_name)

    return dict(test_file=tag, n=n, tp=tp, tn=tn, fp=fp, fn=fn,
                accuracy=round(accuracy, 4), precision=round(precision, 4),
                recall=round(recall, 4), f_score=round(f1, 4),
                anomaly_gt=anomaly_gt, anomaly_pred=anomaly_pred)


def _plot_anomaly_result(z_score, anomaly_sets, length, tag, model_name):
    """battery_quality_mtadgan.plot_result()와 같은 스타일 (Z score + 실제/예측 구간)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    max_len = min(length, len(z_score))
    if max_len <= 0:
        return

    fig = plt.figure(figsize=(30, 12))
    fig.add_subplot(111)
    plt.plot(range(max_len), z_score[:max_len], label="Z score (anomaly score)")
    plt.legend(loc=0, fontsize=30)

    colors = ["red", "blue"]
    for i, anomaly in enumerate(anomaly_sets):
        for t1, t2 in anomaly:
            plt.axvspan(t1, t2, color=colors[i % len(colors)], alpha=0.2)

    plt.title("[%s] %s : Red = True Anomaly, Blue = Predicted Anomaly" % (model_name, tag), size=34)
    plt.ylabel("Z_score", size=30)
    plt.xlabel("Time", size=30)
    plt.xticks(size=26)
    plt.yticks(size=26)
    plt.xlim([0, max_len - 1])
    out = os.path.join(bq.PATHS["output"], "anomaly_%s_%s.png" % (model_name, tag.replace(" ", "_")))
    plt.savefig(out, bbox_inches="tight", dpi=70)
    plt.close()
    print("결과 그래프 저장:", out)


SEQ_WIN_SIZE = 10  # LSTM Autoencoder / USAD 같은 시퀀스 모델의 윈도우 크기 (MTadGAN과 동일하게 맞춤)


def make_windows(X, win_size=SEQ_WIN_SIZE, step=1):
    """(T, D) 배열을 (N, win_size, D) 겹치는 윈도우들로 자른다.
    시퀀스 모델(LSTM Autoencoder, USAD)에서 공통으로 쓴다."""
    X = np.asarray(X)
    n = len(X)
    if n < win_size:
        return np.empty((0, win_size, X.shape[1]))
    starts = range(0, n - win_size + 1, step)
    return np.stack([X[s:s + win_size] for s in starts])


def window_scores_to_pointwise(scores, n_points, win_size=SEQ_WIN_SIZE, step=1):
    """윈도우 단위 이상 점수를 시점(point) 단위로 되돌린다.
    각 시점은 자신을 포함하는 모든 윈도우 점수의 평균을 받는다."""
    sums = np.zeros(n_points)
    counts = np.zeros(n_points)
    for i, s in enumerate(scores):
        start = i * step
        end = start + win_size
        sums[start:end] += s
        counts[start:end] += 1
    counts[counts == 0] = 1
    return sums / counts


def print_and_save(model_name, rows):
    """모델 1개의 파일별 결과를 표로 찍고 output/metrics_<model>.csv 로 저장한다."""
    df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("[%s] 파일별 Accuracy / Precision / Recall / F-score" % model_name)
    print("=" * 70)
    cols = ["test_file", "n", "tp", "fp", "fn", "accuracy", "precision", "recall", "f_score"]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))

    ng_avg = df[df["test_file"].isin(NG_TAGS_ALL) | df["test_file"].isin(SUP_EVAL_TAGS)]
    if len(ng_avg):
        print("\n평균 (NG 파일 기준):")
        print(ng_avg[["precision", "recall", "f_score"]].mean().round(4).to_string())

    out_dir = bq.PATHS["output"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "metrics_%s.csv" % model_name)
    save_cols = [c for c in df.columns if c not in ("anomaly_gt", "anomaly_pred")]
    df[save_cols].to_csv(out_path, index=False)
    print("\n저장: %s" % out_path)
    return df
