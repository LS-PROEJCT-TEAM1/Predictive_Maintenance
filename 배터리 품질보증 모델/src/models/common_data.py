# -*- coding: utf-8 -*-
"""
BatteryFlow AI - 배터리 품질 모듈 / 모델 공통 데이터 모듈

모델별 스크립트(model_01~04)가 공통으로 쓰는 데이터 로딩·Feature·평가 함수.
이 파일 자체는 실행하지 않고, 각 모델 스크립트에서 import 한다.

데이터 구성
    정상 기준 데이터 : data/preprocessed/train/  (10개 파일, 라벨 없음 = 전부 정상)
        1000~1004_chg  (충전 5개)
        1000~1004_dchg (방전 5개)
    라벨 데이터      : data/preprocessed/test/Test0X_..._Label.csv
        학습용 NG : Test05(충전), Test09(방전)
        테스트    : Test03, Test04 (정상) / Test06, Test07, Test08 (불량)

교안 연계
    Part 2 Ch13 Feature Engineering / Part 5 Ch36 학습·테스트 분리
    Part 5 Ch39 성능 평가 / Part 6 Ch43 시계열 Feature
"""

import os
import re
import glob

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRE_TRAIN = os.path.join(ROOT, "data", "preprocessed", "train")
PRE_TEST = os.path.join(ROOT, "data", "preprocessed", "test")
RAW_TEST = os.path.join(ROOT, "data", "raw_data", "test")
OUT = os.path.join(ROOT, "output", "models")
os.makedirs(OUT, exist_ok=True)

RANDOM_STATE = 42

# 정상 기준으로 사용할 학습 파일 10개 (충전 5 + 방전 5)
TRAIN_FILES = [
    ("1000_chg", "충전"), ("1001_chg", "충전"), ("1002_chg", "충전"),
    ("1003_chg", "충전"), ("1004_chg", "충전"),
    ("1000_dchg", "방전"), ("1001_dchg", "방전"), ("1002_dchg", "방전"),
    ("1003_dchg", "방전"), ("1004_dchg", "방전"),
]

# 라벨이 있는 시험 파일 (지도학습의 NG 클래스 / 최종 평가용)
TEST_FILES = [
    # (파일 접두사, 충방전, 판정, 용도)
    ("Test05_NG_chg",  "충전", "NG", "train"),
    ("Test09_NG_dchg", "방전", "NG", "train"),
    ("Test03_OK_chg",  "충전", "OK", "test"),
    ("Test04_OK_dchg", "방전", "OK", "test"),
    ("Test06_NG_chg",  "충전", "NG", "test"),
    ("Test07_NG_dchg", "방전", "NG", "test"),
    ("Test08_NG_chg",  "충전", "NG", "test"),
]

FEATURES = [
    "셀전압_평균", "셀전압_최소", "셀전압_최대", "셀전압_편차", "셀전압_표준편차",
    "이상셀_개수", "셀전압_최대z",
    "모듈온도_평균", "모듈온도_최소", "모듈온도_최대", "모듈온도_편차", "모듈온도_표준편차",
    "셀전압평균_변화율", "셀전압편차_이동평균잔차", "모듈온도최대_변화율", "방전여부",
]

# 절대 수준(전압·온도의 절대값)을 뺀 '상대 특성'.
# 팩·계절·설비가 바뀌어도 의미가 유지되는 변수만 남긴 조합.
REL_FEATURES = [
    "셀전압_편차", "셀전압_표준편차", "이상셀_개수", "셀전압_최대z",
    "모듈온도_편차", "모듈온도_표준편차",
    "셀전압평균_변화율", "셀전압편차_이동평균잔차", "모듈온도최대_변화율", "방전여부",
]


# =============================================================================
# 데이터 정제
# =============================================================================

def _clean(data, cv_cols, tp_cols):
    """물리 범위 이탈 처리 + 결측 보간.

    - 셀 전압 2.0~5.0V, 모듈 온도 -20~100℃ 밖은 센서 오류로 보고 결측 처리
    - 시계열이므로 앞뒤 시점 선형보간
    - 한 셀(열) 전체가 이탈이면 같은 시점 다른 셀의 중앙값으로 대체
    """
    v_bad = (data[cv_cols] < 2.0) | (data[cv_cols] > 5.0)
    t_bad = (data[tp_cols] < -20) | (data[tp_cols] > 100)
    n_bad = int(v_bad.sum().sum() + t_bad.sum().sum())
    data[cv_cols] = data[cv_cols].mask(v_bad)
    data[tp_cols] = data[tp_cols].mask(t_bad)

    data = data.interpolate(limit_direction="both").ffill().bfill()

    n_dead = 0
    for cols in (cv_cols, tp_cols):
        dead = [c for c in cols if data[c].isna().all()]
        n_dead += len(dead)
        if dead:
            med = data[[c for c in cols if c not in dead]].median(axis=1)
            for c in dead:
                data[c] = med
    return data, n_bad, n_dead


def _split_columns(df):
    cv = [c for c in df.columns if re.match(r"M\d+CV\d+$", c)]
    tp = [c for c in df.columns if re.match(r"M\d+T\d+$", c)]
    return cv, tp


# =============================================================================
# Feature Engineering (교안 Ch13, Ch43)
# =============================================================================

def make_features(data, cv_cols, tp_cols, mode):
    """208개 원 변수 → 해석 가능한 16개 지표로 요약. 셀 단위 z-score도 함께 반환."""
    V = data[cv_cols].values
    T = data[tp_cols].values

    v_mean, v_std = V.mean(axis=1), V.std(axis=1)

    # 같은 시점 다른 셀 대비 얼마나 벗어났는지 (셀 단위 z-score)
    z = np.abs(V - v_mean[:, None]) / np.where(v_std[:, None] == 0, 1e-9, v_std[:, None])

    f = pd.DataFrame({
        "셀전압_평균": v_mean,
        "셀전압_최소": V.min(axis=1),
        "셀전압_최대": V.max(axis=1),
        "셀전압_편차": V.max(axis=1) - V.min(axis=1),
        "셀전압_표준편차": v_std,
        "이상셀_개수": (z > 3).sum(axis=1),
        "셀전압_최대z": z.max(axis=1),
        "모듈온도_평균": T.mean(axis=1),
        "모듈온도_최소": T.min(axis=1),
        "모듈온도_최대": T.max(axis=1),
        "모듈온도_편차": T.max(axis=1) - T.min(axis=1),
        "모듈온도_표준편차": T.std(axis=1),
    })
    f["셀전압평균_변화율"] = f["셀전압_평균"].diff().fillna(0)
    f["셀전압편차_이동평균잔차"] = (f["셀전압_편차"]
                                 - f["셀전압_편차"].rolling(10, min_periods=1).mean())
    f["모듈온도최대_변화율"] = f["모듈온도_최대"].diff().fillna(0)
    f["방전여부"] = 1 if mode == "방전" else 0
    return f[FEATURES], z


# =============================================================================
# 데이터 적재
# =============================================================================

def _find_raw(prefix):
    p = os.path.join(RAW_TEST, prefix + ".csv")
    if os.path.exists(p):
        return p
    cand = glob.glob(os.path.join(RAW_TEST, prefix + "*.csv"))
    if not cand:
        raise FileNotFoundError(prefix)
    return cand[0]


def load_dataset(verbose=True):
    """정상 기준 10개 + 라벨 시험 7개를 하나의 DataFrame으로 합친다.

    반환 컬럼: FEATURES + [label, 파일, 충방전, 판정, 구분, 시점]
        구분 = "normal"(정상 기준 10개) / "train"(NG 학습용) / "test"(평가용)
    """
    frames, cell_z = [], {}
    rows = []

    # 1) 정상 기준 데이터 (preprocessed/train) - 라벨 없음 = 전부 정상(0)
    for prefix, mode in TRAIN_FILES:
        df = pd.read_csv(os.path.join(PRE_TRAIN, prefix + ".csv"))
        cv, tp = _split_columns(df)
        data, n_bad, n_dead = _clean(df[cv + tp].apply(pd.to_numeric, errors="coerce"), cv, tp)
        f, z = make_features(data, cv, tp, mode)
        f["label"] = 0
        f["파일"], f["충방전"], f["판정"], f["구분"] = prefix, mode, "OK", "normal"
        f["시점"] = np.arange(len(f))
        frames.append(f)
        cell_z[prefix] = (z, np.zeros(len(f), dtype=int), cv)
        rows.append({"파일": prefix, "충방전": mode, "판정": "OK", "구분": "normal",
                     "행": len(f), "범위이탈": n_bad, "센서고장 셀": n_dead, "이상 시점": 0})

    # 2) 라벨이 있는 시험 데이터 (raw_data/test + Label)
    for prefix, mode, grade, split in TEST_FILES:
        df = pd.read_csv(_find_raw(prefix))
        cv, tp = _split_columns(df)
        data, n_bad, n_dead = _clean(df[cv + tp].apply(pd.to_numeric, errors="coerce"), cv, tp)
        f, z = make_features(data, cv, tp, mode)
        label = pd.read_csv(os.path.join(PRE_TEST, prefix + "_Label.csv"))["label"].values
        n = min(len(f), len(label))
        f, label, z = f.iloc[:n].reset_index(drop=True), label[:n], z[:n]
        f["label"] = label
        f["파일"], f["충방전"], f["판정"], f["구분"] = prefix, mode, grade, split
        f["시점"] = np.arange(len(f))
        frames.append(f)
        cell_z[prefix] = (z, label, cv)
        rows.append({"파일": prefix, "충방전": mode, "판정": grade, "구분": split,
                     "행": n, "범위이탈": n_bad, "센서고장 셀": n_dead,
                     "이상 시점": int(label.sum())})

    df_all = pd.concat(frames, ignore_index=True)
    report = pd.DataFrame(rows)

    if verbose:
        print("=" * 78)
        print("데이터 적재 결과")
        print("=" * 78)
        print(report.to_string(index=False))
        print(f"\n정상 기준(preprocessed/train) {len(TRAIN_FILES)}개 파일 "
              f"{int((df_all['구분'] == 'normal').sum()):,}행")
        print(f"전체 {len(df_all):,}행 / 이상 시점 {int(df_all['label'].sum()):,}행\n")
    return df_all, report, cell_z


def get_splits(df_all):
    """모델 학습/평가에 쓰는 3개 묶음을 돌려준다.

    normal_df : 정상 기준 10개 파일 (비지도 모델의 기준 데이터)
    train_df  : 지도학습용 = 정상 10개 + NG 2개(Test05, Test09)
    test_df   : 평가용 5개 (Test03, 04, 06, 07, 08)
    """
    normal_df = df_all[df_all["구분"] == "normal"]
    train_df = df_all[df_all["구분"].isin(["normal", "train"])]
    test_df = df_all[df_all["구분"] == "test"]
    return normal_df, train_df, test_df


# =============================================================================
# 평가 (교안 Ch39 / Ch40 코드 40-6 형식)
# =============================================================================

def evaluate(model_name, y_true, y_pred, split_name="", show=True):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if show:
        print(f"[{model_name}] {split_name}")
        print(f"  Accuracy : {acc:.4f}  Precision : {pre:.4f}  "
              f"Recall : {rec:.4f}  F1-score : {f1:.4f}")
        print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    return {"모델": model_name, "구분": split_name, "Accuracy": acc, "Precision": pre,
            "Recall": rec, "F1-score": f1,
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}


def evaluate_per_file(model_name, test_df, y_pred):
    """테스트 파일별로 오류를 분해한다(어느 시험에서 틀리는지 확인)."""
    rows = []
    y_pred = np.asarray(y_pred)
    for f_name in test_df["파일"].unique():
        m = (test_df["파일"] == f_name).values
        yt, pp = test_df["label"].values[m], y_pred[m]
        tn, fp, fn, tp = confusion_matrix(yt, pp, labels=[0, 1]).ravel()
        rows.append({"모델": model_name, "파일": f_name, "행 수": int(m.sum()),
                     "실제 이상": int(yt.sum()), "예측 이상": int(pp.sum()),
                     "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
                     "Accuracy": round(accuracy_score(yt, pp), 4)})
    return pd.DataFrame(rows)


def save(df, name):
    path = os.path.join(OUT, name)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print("저장:", path)
    return path


# =============================================================================
# 파일 단위 교차검증 (Out-of-Fold 통합)
# =============================================================================

def oof_cross_validation(lab, fit_predict, n_splits=5, model_name="", show=True):
    """라벨이 있는 시험 파일을 5개 묶음으로 나눠 교차검증한다.

    fold별 점수를 평균내면 "0.61±0.48"처럼 편차가 커서 해석이 어렵다.
    대신 각 fold에서 '테스트로 빠진 시점'의 예측만 모아(Out-of-Fold)
    마지막에 지표를 한 번 계산한다. 모든 시점이 정확히 한 번씩 예측되므로
    결과가 단일 값으로 나온다.

    인자
        lab         : 라벨이 있는 시험 데이터 (파일 컬럼 포함, 인덱스 0부터)
        fit_predict : fit_predict(train_part, test_part) -> test_part의 예측 배열
                      지도학습이면 모델을 학습해 예측하고,
                      비지도면 임계값을 train_part에서 정해 test_part에 적용한다.
    반환
        cv(단일 값 1행), fold별 내역, out-of-fold 예측 배열
    """
    lab = lab.reset_index(drop=True)
    oof = np.zeros(len(lab), dtype=int)
    fold_rows = []

    for i, (tr, te) in enumerate(
            GroupKFold(n_splits=n_splits).split(lab, lab["label"], groups=lab["파일"]), 1):
        pred = np.asarray(fit_predict(lab.iloc[tr], lab.iloc[te]))
        oof[te] = pred
        yt = lab["label"].iloc[te]
        fold_rows.append({"fold": i,
                          "테스트 파일": ", ".join(sorted(lab["파일"].iloc[te].unique())),
                          "행 수": len(te),
                          "실제 이상": int(yt.sum()),
                          "Accuracy": accuracy_score(yt, pred),
                          "Precision": precision_score(yt, pred, zero_division=0),
                          "Recall": recall_score(yt, pred, zero_division=0),
                          "F1-score": f1_score(yt, pred, zero_division=0)})

    y_all = lab["label"]
    cv = pd.DataFrame([{"Accuracy": accuracy_score(y_all, oof),
                        "Precision": precision_score(y_all, oof, zero_division=0),
                        "Recall": recall_score(y_all, oof, zero_division=0),
                        "F1-score": f1_score(y_all, oof, zero_division=0)}]).round(4)
    if show:
        print(f"\n[{model_name}] 파일 단위 GroupKFold 교차검증 "
              f"({n_splits}-fold, Out-of-Fold 통합)")
        print(cv.to_string(index=False))
    return cv, pd.DataFrame(fold_rows).round(4), oof


def tune_threshold(y_true, score, quantiles, ks, ref_score, flag_fn,
                   min_precision=0.9):
    """정상 데이터 분위수 x 연속길이 조합에서 임계값을 고른다.

    기획서 목표가 Precision 0.7 이상이므로
    Precision min_precision 이상을 만족하는 조합 중 Recall이 가장 높은 것을 쓴다.
    조건을 만족하는 조합이 없으면 F1이 가장 높은 조합을 쓴다.
    """
    best, rows = None, []
    for q in quantiles:
        thr = np.quantile(ref_score, q)
        for k in ks:
            pred = flag_fn(score, thr, k)
            pr = precision_score(y_true, pred, zero_division=0)
            rc = recall_score(y_true, pred, zero_division=0)
            f1 = f1_score(y_true, pred, zero_division=0)
            rows.append({"분위수": q, "연속 k": k, "임계값": thr,
                         "Precision": pr, "Recall": rc, "F1-score": f1})
            if pr >= min_precision and (best is None or rc > best[0]):
                best = (rc, q, k, thr)
    if best is None:
        r = max(rows, key=lambda d: d["F1-score"])
        best = (r["Recall"], r["분위수"], r["연속 k"], r["임계값"])
    return best[1], best[2], best[3], pd.DataFrame(rows)
