# -*- coding: utf-8 -*-
"""
BatteryFlow AI Control Tower - [배터리 품질] 모듈
셀 전압 176 + 모듈 온도 32 = 208개 시계열 변수 기반 이상 셀·구간 탐지

담당: 최우찬 / 데이터: KAMP 전자부품(배터리팩) 품질보증 AI 데이터셋

기획서 연계
    - 목표      : 셀 전압·모듈 온도 패턴을 분석해 이상 가능성이 있는 셀과 구간을 탐지
    - 목표치    : PCA 기반 이상탐지에서 이상 셀 판별 Precision 0.7 이상
    - 화면 출력 : 셀 전압 Heatmap, 이상 셀·구간, 이상 점수

교안 연계
    - Part 2 Ch13 : 데이터 변환과 Feature Engineering
    - Part 2 Ch15/16 : EDA, 기술통계와 상관관계 분석
    - Part 3 Ch20/21 : Plotly Express / Graph Objects 시각화
    - Part 5 Ch36 : 학습/테스트 데이터 분리
    - Part 5 Ch39 : 모델 성능 평가와 과적합 이해
    - Part 5 Ch40 : 의사결정나무와 랜덤포레스트
    - Part 6 Ch43/46 : 시계열 Feature(이동평균 잔차), 이상치 탐지

실행
    python src/battery_quality_ml.py
"""

import os
import re
import glob
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

import plotly.express as px
import plotly.graph_objects as go

warnings.simplefilter(action="ignore", category=FutureWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_TEST = os.path.join(ROOT, "data", "raw_data", "test")
PRE_TEST = os.path.join(ROOT, "data", "preprocessed", "test")
OUT = os.path.join(ROOT, "output", "quality_ml")
os.makedirs(OUT, exist_ok=True)

RANDOM_STATE = 42

# =============================================================================
# 0. 분석 대상 파일 정의
#    학습/테스트를 "파일 단위"로 분리한다. 같은 시험 파일의 앞뒤 시점이
#    학습과 테스트에 섞이면 사실상 정답을 보고 학습하는 것과 같아
#    성능이 과도하게 높게 나온다(교안 Ch39 '데이터 누수').
# =============================================================================

FILES = [
    # (파일 접두사, 충방전 구분, 품질 판정, 학습/테스트 분리)
    ("Test01_OK_chg",   "충전", "OK", "train"),
    ("Test02_OK_dchg",  "방전", "OK", "train"),
    ("Test03_OK_chg",   "충전", "OK", "test"),
    ("Test04_OK_dchg",  "방전", "OK", "test"),
    ("Test05_NG_chg",   "충전", "NG", "train"),
    ("Test06_NG_chg",   "충전", "NG", "test"),
    ("Test07_NG_dchg",  "방전", "NG", "test"),
    ("Test08_NG_chg",   "충전", "NG", "test"),
    ("Test09_NG_dchg",  "방전", "NG", "train"),
]


# =============================================================================
# [단계 1] 데이터 정제 / 전처리
# =============================================================================

def find_raw_path(prefix):
    """파일명에 공백이 섞인 경우(Test03_OK_chg .csv)까지 찾아준다."""
    path = os.path.join(RAW_TEST, prefix + ".csv")
    if os.path.exists(path):
        return path
    cand = glob.glob(os.path.join(RAW_TEST, prefix + "*.csv"))
    if not cand:
        raise FileNotFoundError(prefix)
    return cand[0]


def clean_one_file(prefix, mode, grade, split, report):
    """원 데이터 1개 파일 정제.

    1) 셀 전압(M**CV**) 176개 / 모듈 온도(M**T**) 32개만 선택
       - 나머지 BMS 요약 컬럼(Voltage, Current, SOH, DV 등)은 OK 파일에서
         전부 0으로 기록되어 있어(미기록 상수 컬럼) 그대로 쓰면
         '파일 종류'를 그대로 알려주는 누수 변수가 된다. 제외한다.
    2) 결측치: 시계열이므로 앞뒤 시점 선형보간 후 남은 값은 전/후방 채움
    3) 중복 행: 개수만 기록(연속 측정에서는 동일값이 자연 발생)
    4) 물리적 불가능 값: 셀 전압 2.0~5.0V 밖은 센서 오류로 보고 결측 처리 후 보간
       (Test06은 최대 52V 기록이 있어 이 단계에서 정리된다)
    """
    df = pd.read_csv(find_raw_path(prefix))
    n_raw, c_raw = df.shape

    cv_cols = [c for c in df.columns if re.match(r"M\d+CV\d+$", c)]
    tp_cols = [c for c in df.columns if re.match(r"M\d+T\d+$", c)]
    data = df[cv_cols + tp_cols].apply(pd.to_numeric, errors="coerce")

    n_missing = int(data.isna().sum().sum())
    n_dup = int(df.duplicated().sum())

    bad_v = ((data[cv_cols] < 2.0) | (data[cv_cols] > 5.0)).sum().sum()
    data[cv_cols] = data[cv_cols].mask((data[cv_cols] < 2.0) | (data[cv_cols] > 5.0))
    bad_t = ((data[tp_cols] < -20) | (data[tp_cols] > 100)).sum().sum()
    data[tp_cols] = data[tp_cols].mask((data[tp_cols] < -20) | (data[tp_cols] > 100))

    data = data.interpolate(limit_direction="both").ffill().bfill()

    # 한 셀(열) 전체가 범위 이탈이면 시간 방향 보간으로는 채울 수 없다.
    # (Test06은 특정 셀이 전 구간 52V로 기록된 센서 고장 사례)
    # 이 경우 같은 시점 다른 셀의 중앙값으로 대체하고, 개수를 기록한다.
    n_dead = 0
    for cols in (cv_cols, tp_cols):
        dead = [c for c in cols if data[c].isna().all()]
        n_dead += len(dead)
        if dead:
            med = data[[c for c in cols if c not in dead]].median(axis=1)
            for c in dead:
                data[c] = med

    label = pd.read_csv(os.path.join(PRE_TEST, prefix + "_Label.csv"))["label"].values
    n = min(len(data), len(label))
    data, label = data.iloc[:n].reset_index(drop=True), label[:n]

    report.append({
        "파일": prefix, "충방전": mode, "판정": grade, "구분": split,
        "원본 행": n_raw, "원본 열": c_raw,
        "사용 열(셀전압+온도)": len(cv_cols) + len(tp_cols),
        "결측치": n_missing, "중복 행": n_dup,
        "전압 범위이탈": int(bad_v), "온도 범위이탈": int(bad_t),
        "센서고장 셀(열 전체 이탈)": n_dead,
        "최종 행": n, "이상 시점 수": int(label.sum()),
        "이상 비율(%)": round(100 * label.mean(), 1),
    })
    return data, label, cv_cols, tp_cols


# =============================================================================
# [단계 2] Feature Engineering (교안 Part 2 Ch13, Part 6 Ch43)
#          208개 원 변수를 그대로 쓰지 않고, 현장에서 해석 가능한
#          '셀 전압 분포 / 모듈 온도 분포 / 시간 변화' 지표로 요약한다.
# =============================================================================

FEATURES = [
    "셀전압_평균", "셀전압_최소", "셀전압_최대", "셀전압_편차", "셀전압_표준편차",
    "이상셀_개수", "셀전압_최대z",
    "모듈온도_평균", "모듈온도_최소", "모듈온도_최대", "모듈온도_편차", "모듈온도_표준편차",
    "셀전압평균_변화율", "셀전압편차_이동평균잔차", "모듈온도최대_변화율", "방전여부",
]

# 절대 수준(전압/온도의 절대값)을 제외한 '상대 특성' Feature.
# 팩·계절·설비가 달라도 의미가 유지되는 변수만 남긴 조합이다.
REL_FEATURES = [
    "셀전압_편차", "셀전압_표준편차", "이상셀_개수", "셀전압_최대z",
    "모듈온도_편차", "모듈온도_표준편차",
    "셀전압평균_변화율", "셀전압편차_이동평균잔차", "모듈온도최대_변화율", "방전여부",
]


def make_features(data, cv_cols, tp_cols, mode):
    V = data[cv_cols].values
    T = data[tp_cols].values

    v_mean, v_std = V.mean(axis=1), V.std(axis=1)
    t_mean = T.mean(axis=1)

    # 같은 시점의 다른 셀 대비 얼마나 벗어났는지 (셀 단위 z-score)
    z = np.abs(V - v_mean[:, None]) / np.where(v_std[:, None] == 0, 1e-9, v_std[:, None])

    f = pd.DataFrame({
        "셀전압_평균": v_mean,
        "셀전압_최소": V.min(axis=1),
        "셀전압_최대": V.max(axis=1),
        "셀전압_편차": V.max(axis=1) - V.min(axis=1),      # 셀 간 전압 불균형
        "셀전압_표준편차": v_std,
        "이상셀_개수": (z > 3).sum(axis=1),                 # 3σ 이탈 셀 수
        "셀전압_최대z": z.max(axis=1),
        "모듈온도_평균": t_mean,
        "모듈온도_최소": T.min(axis=1),
        "모듈온도_최대": T.max(axis=1),
        "모듈온도_편차": T.max(axis=1) - T.min(axis=1),     # 모듈 간 온도 불균형
        "모듈온도_표준편차": T.std(axis=1),
    })

    # 시계열 Feature (교안 Ch43 / Ch46 : 이동평균 잔차)
    f["셀전압평균_변화율"] = f["셀전압_평균"].diff().fillna(0)
    ma = f["셀전압_편차"].rolling(10, min_periods=1).mean()
    f["셀전압편차_이동평균잔차"] = f["셀전압_편차"] - ma
    f["모듈온도최대_변화율"] = f["모듈온도_최대"].diff().fillna(0)
    f["방전여부"] = 1 if mode == "방전" else 0
    return f[FEATURES], z


# =============================================================================
# [단계 3] 모델 평가 함수 (교안 Ch39 / Ch40 코드 40-6 형식)
# =============================================================================

def evaluate_classification_model(model_name, y_true, y_pred, split_name=""):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    print(f"[{model_name}] {split_name}")
    print(f"  Accuracy : {acc:.4f}  Precision : {pre:.4f}  "
          f"Recall : {rec:.4f}  F1-score : {f1:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    return {"모델": model_name, "구분": split_name, "Accuracy": acc,
            "Precision": pre, "Recall": rec, "F1-score": f1,
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}


def main():
    print("=" * 70)
    print("BatteryFlow AI Control Tower - 배터리 품질 모듈")
    print("=" * 70)

    # ---------------- 단계 1~2 : 정제 + Feature ----------------
    clean_report, frames = [], []
    cell_z_store = {}
    for prefix, mode, grade, split in FILES:
        data, label, cv_cols, tp_cols = clean_one_file(
            prefix, mode, grade, split, clean_report)
        feat, z = make_features(data, cv_cols, tp_cols, mode)
        feat["label"] = label
        feat["파일"] = prefix
        feat["충방전"] = mode
        feat["판정"] = grade
        feat["구분"] = split
        feat["시점"] = np.arange(len(feat))
        frames.append(feat)
        cell_z_store[prefix] = (z, label, cv_cols)

    report_df = pd.DataFrame(clean_report)
    df = pd.concat(frames, ignore_index=True)
    report_df.to_csv(os.path.join(OUT, "01_전처리_요약.csv"),
                     index=False, encoding="utf-8-sig")
    print("\n[단계 1] 데이터 정제 요약")
    print(report_df.to_string(index=False))
    print(f"\n전체 데이터: {df.shape[0]}행 / Feature {len(FEATURES)}개 "
          f"/ 이상 시점 비율 {100 * df['label'].mean():.1f}%")

    # ---------------- 단계 2 : 충방전별 EDA ----------------
    eda = (df.groupby(["충방전", "판정"])[
        ["셀전압_편차", "셀전압_표준편차", "이상셀_개수",
         "모듈온도_최대", "모듈온도_편차"]]
        .mean().round(4).reset_index())
    eda.to_csv(os.path.join(OUT, "02_충방전별_EDA.csv"),
               index=False, encoding="utf-8-sig")
    print("\n[단계 2] 충방전 구분 x 품질 판정별 평균 특성")
    print(eda.to_string(index=False))

    corr = df[FEATURES + ["label"]].corr()["label"].drop("label").sort_values(
        key=abs, ascending=False)
    corr.round(4).to_csv(os.path.join(OUT, "02_label_상관계수.csv"),
                         encoding="utf-8-sig")
    print("\n이상 여부(label)와의 상관계수 상위 5개")
    print(corr.head(5).round(4).to_string())

    # ---------------- 단계 3 : 학습/테스트 분리 (교안 Ch36) ----------------
    train_df = df[df["구분"] == "train"]
    test_df = df[df["구분"] == "test"]
    X_train, y_train = train_df[FEATURES], train_df["label"]
    X_test, y_test = test_df[FEATURES], test_df["label"]
    print(f"\n[단계 3] 학습 {X_train.shape} / 테스트 {X_test.shape}")
    print("학습 파일:", sorted(train_df['파일'].unique()))
    print("테스트 파일:", sorted(test_df['파일'].unique()))

    results = []

    # ---------------- 모델 1 : Decision Tree (교안 코드 40-4) ----------------
    # 전체 Feature 사용 (절대 전압/온도 수준 포함)
    tree_all = DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)
    tree_all.fit(X_train, y_train)
    results.append(evaluate_classification_model(
        "Decision Tree(전체)", y_train, tree_all.predict(X_train), "학습"))
    results.append(evaluate_classification_model(
        "Decision Tree(전체)", y_test, tree_all.predict(X_test), "테스트"))

    # ---------------- 모델 2 : Random Forest (교안 코드 40-5) ----------------
    forest_all = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=RANDOM_STATE)
    forest_all.fit(X_train, y_train)
    results.append(evaluate_classification_model(
        "Random Forest(전체)", y_train, forest_all.predict(X_train), "학습"))
    results.append(evaluate_classification_model(
        "Random Forest(전체)", y_test, forest_all.predict(X_test), "테스트"))

    # ---------------- 모델 1-2, 2-2 : 상대 Feature만 사용 ----------------
    # 절대 온도/전압 수준은 시험 조건(계절, 설비, 팩)에 따라 달라지므로
    # 새 팩에 그대로 적용하기 어렵다. 셀 간 불균형과 시간 변화처럼
    # '팩 내부에서의 상대적 특성'만 남겨 일반화 성능을 비교한다.
    tree_model = DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)
    tree_model.fit(X_train[REL_FEATURES], y_train)
    results.append(evaluate_classification_model(
        "Decision Tree(상대)", y_train, tree_model.predict(X_train[REL_FEATURES]), "학습"))
    results.append(evaluate_classification_model(
        "Decision Tree(상대)", y_test, tree_model.predict(X_test[REL_FEATURES]), "테스트"))

    forest_model = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=RANDOM_STATE)
    forest_model.fit(X_train[REL_FEATURES], y_train)
    results.append(evaluate_classification_model(
        "Random Forest(상대)", y_train, forest_model.predict(X_train[REL_FEATURES]), "학습"))
    rf_pred = forest_model.predict(X_test[REL_FEATURES])
    results.append(evaluate_classification_model(
        "Random Forest(상대)", y_test, rf_pred, "테스트"))

    # ---------------- 모델 3 : PCA 기반 이상탐지 (비지도) ----------------
    # 정상(OK) 학습 데이터만으로 '정상 패턴'을 학습하고,
    # 그 패턴으로 재현되지 않는 정도(Q통계량=재구성오차)와
    # 정상 범위에서 벗어난 정도(Hotelling T2)로 이상 점수를 만든다.
    pca_feats = [c for c in REL_FEATURES if c != "방전여부"]
    ok_train = train_df[train_df["판정"] == "OK"]
    scaler = StandardScaler().fit(ok_train[pca_feats])
    Z_ok = scaler.transform(ok_train[pca_feats])

    pca = PCA(n_components=0.95, random_state=RANDOM_STATE).fit(Z_ok)
    print(f"\n[PCA] 주성분 {pca.n_components_}개 / "
          f"누적 설명분산 {pca.explained_variance_ratio_.sum():.4f}")

    def pca_scores(X_raw):
        Z = scaler.transform(X_raw)
        P = pca.transform(Z)
        Q = np.sum((Z - pca.inverse_transform(P)) ** 2, axis=1)   # 재구성 오차
        T2 = np.sum(P ** 2 / pca.explained_variance_, axis=1)     # Hotelling T2
        return Q, T2

    Q_ok, T2_ok = pca_scores(ok_train[pca_feats])

    # 임계값(정상 데이터 분위수)과 '연속 초과 길이'를 학습 데이터에서만 탐색한다.
    # 단발성 잡음 1~2 시점을 이상으로 보지 않고, 일정 시간 이상 지속될 때만
    # 이상 구간으로 판정하는 현장 기준을 반영한다(교안 Ch46 : 단순 변동 vs 이상치).
    def flag_with_rule(Q, T2, q_thr, t_thr, k):
        raw = ((Q > q_thr) | (T2 > t_thr)).astype(int)
        if k <= 1:
            return raw
        keep = pd.Series(raw).rolling(k, min_periods=k).min().fillna(0).values
        out = np.zeros_like(raw)
        for i in np.where(keep == 1)[0]:
            out[i - k + 1:i + 1] = 1
        return out

    best = None
    tune_rows = []
    for q in (0.99, 0.999, 0.9999):
        qt, tt = np.quantile(Q_ok, q), np.quantile(T2_ok, q)
        for k in (1, 10, 30, 60):
            preds = []
            for f_name in train_df["파일"].unique():
                mask = train_df["파일"] == f_name
                Qf, T2f = pca_scores(train_df.loc[mask, pca_feats])
                preds.append(pd.Series(flag_with_rule(Qf, T2f, qt, tt, k),
                                       index=train_df.index[mask]))
            pred = pd.concat(preds).reindex(train_df.index).values
            f1 = f1_score(y_train, pred, zero_division=0)
            pr = precision_score(y_train, pred, zero_division=0)
            rc = recall_score(y_train, pred, zero_division=0)
            tune_rows.append({"분위수": q, "연속 k": k, "Precision": pr,
                              "Recall": rc, "F1-score": f1})
            # 기획서 목표가 'Precision 0.7 이상'이므로,
            # 학습 데이터에서 Precision 0.9 이상을 만족하는 설정 중
            # Recall이 가장 높은 조합을 고른다(오경보 우선 억제).
            if pr >= 0.9 and (best is None or rc > best[0]):
                best = (rc, q, k, qt, tt)
    if best is None:                      # 조건을 만족하는 설정이 없으면 F1 최대
        r = max(tune_rows, key=lambda d: d["F1-score"])
        best = (r["Recall"], r["분위수"], r["연속 k"],
                np.quantile(Q_ok, r["분위수"]), np.quantile(T2_ok, r["분위수"]))
    pd.DataFrame(tune_rows).round(4).to_csv(
        os.path.join(OUT, "03b_PCA_임계값_탐색.csv"), index=False, encoding="utf-8-sig")
    _, best_q, best_k, q_thr, t_thr = best
    print(f"[PCA] 학습 데이터 기준 선택: 분위수 {best_q}, 연속 {best_k}시점 "
          f"→ Q={q_thr:.3f}, T2={t_thr:.3f}")

    def pca_predict(X_raw, files=None):
        Q, T2 = pca_scores(X_raw)
        if files is None:
            return flag_with_rule(Q, T2, q_thr, t_thr, best_k), Q, T2
        # 파일(시험)별로 시간 순서가 이어지므로 파일 단위로 규칙을 적용한다.
        out = np.zeros(len(Q), dtype=int)
        files = np.asarray(files)
        for f_name in pd.unique(files):
            m = files == f_name
            out[m] = flag_with_rule(Q[m], T2[m], q_thr, t_thr, best_k)
        return out, Q, T2

    pca_train_pred, _, _ = pca_predict(X_train[pca_feats], train_df["파일"].values)
    pca_test_pred, Q_test, T2_test = pca_predict(X_test[pca_feats], test_df["파일"].values)
    results.append(evaluate_classification_model(
        "PCA 이상탐지", y_train, pca_train_pred, "학습"))
    results.append(evaluate_classification_model(
        "PCA 이상탐지", y_test, pca_test_pred, "테스트"))

    # ---------------- 모델 4 : 교안 규칙 기반 이상탐지 (Part 6 Ch46) ----------------
    # 코드 46-5 방식(평균 + 3 x 표준편차)을 정상(OK) 학습 데이터에만 적용해
    # 변수별 임계값을 만들고, 불량 유형별 규칙을 OR로 묶는다.
    #   - 모듈온도_편차 : 과열/냉각 불균형 유형 (Test08)
    #   - 셀전압_최대z  : 특정 셀 전압 이탈 유형 (Test06 센서고장, Test07 급락)
    RULE_COLS = ["모듈온도_편차", "셀전압_최대z"]
    rule_thr = {c: ok_train[c].mean() + 3 * ok_train[c].std() for c in RULE_COLS}
    print("\n[교안 규칙] 정상 데이터 기준 3σ 임계값:",
          {c: round(v, 3) for c, v in rule_thr.items()})

    def rule_predict(frame):
        flag = np.zeros(len(frame), dtype=bool)
        for c, v in rule_thr.items():
            flag |= (frame[c] > v).values
        return flag.astype(int)

    rule_train_pred = rule_predict(X_train)
    rule_pred = rule_predict(X_test)
    results.append(evaluate_classification_model(
        "교안 규칙(3σ OR)", y_train, rule_train_pred, "학습"))
    results.append(evaluate_classification_model(
        "교안 규칙(3σ OR)", y_test, rule_pred, "테스트"))

    # 변수별 단독 규칙 성능도 기록 (어느 변수가 어떤 불량을 잡는지 확인)
    single = []
    for c in ["모듈온도_편차", "셀전압_최대z", "이상셀_개수", "셀전압_편차",
              "셀전압_표준편차", "모듈온도_표준편차"]:
        v = ok_train[c].mean() + 3 * ok_train[c].std()
        p_ = (X_test[c] > v).astype(int).values
        single.append({"변수": c, "임계값(3σ)": round(v, 4),
                       "Precision": precision_score(y_test, p_, zero_division=0),
                       "Recall": recall_score(y_test, p_, zero_division=0),
                       "F1-score": f1_score(y_test, p_, zero_division=0)})
    pd.DataFrame(single).round(4).to_csv(
        os.path.join(OUT, "03e_교안_단일변수_3시그마.csv"),
        index=False, encoding="utf-8-sig")

    # ---------------- 단계 4 : 파일 단위 교차검증 (안정성 확인) ----------------
    print("\n[단계 4] 파일 단위 GroupKFold 교차검증 (5-fold)")
    gkf = GroupKFold(n_splits=5)
    cv_rows = []
    for name, feats, model in [
            ("Decision Tree(전체)", FEATURES, DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)),
            ("Random Forest(전체)", FEATURES, RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE)),
            ("Decision Tree(상대)", REL_FEATURES, DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)),
            ("Random Forest(상대)", REL_FEATURES, RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE))]:
        scores = []
        for tr, te in gkf.split(df[feats], df["label"], groups=df["파일"]):
            m = model.__class__(**model.get_params())
            m.fit(df[feats].iloc[tr], df["label"].iloc[tr])
            p = m.predict(df[feats].iloc[te])
            scores.append([accuracy_score(df["label"].iloc[te], p),
                           precision_score(df["label"].iloc[te], p, zero_division=0),
                           recall_score(df["label"].iloc[te], p, zero_division=0),
                           f1_score(df["label"].iloc[te], p, zero_division=0)])
        s = np.array(scores)
        cv_rows.append({"모델": name,
                        "Accuracy(평균)": s[:, 0].mean(), "Accuracy(표준편차)": s[:, 0].std(),
                        "Precision(평균)": s[:, 1].mean(), "Precision(표준편차)": s[:, 1].std(),
                        "Recall(평균)": s[:, 2].mean(), "Recall(표준편차)": s[:, 2].std(),
                        "F1(평균)": s[:, 3].mean(), "F1(표준편차)": s[:, 3].std()})
        print(f"  {name}: Accuracy {s[:,0].mean():.4f}±{s[:,0].std():.4f}, "
              f"Precision {s[:,1].mean():.4f}±{s[:,1].std():.4f}, "
              f"Recall {s[:,2].mean():.4f}±{s[:,2].std():.4f}, "
              f"F1 {s[:,3].mean():.4f}±{s[:,3].std():.4f}")
    pd.DataFrame(cv_rows).round(4).to_csv(
        os.path.join(OUT, "04_교차검증_결과.csv"), index=False, encoding="utf-8-sig")

    res_df = pd.DataFrame(results).round(4)
    res_df.to_csv(os.path.join(OUT, "03_모델_성능지표.csv"),
                  index=False, encoding="utf-8-sig")
    print("\n[단계 4] 모델별 성능지표")
    print(res_df.to_string(index=False))

    # 테스트 파일별 오류 분해 (어느 시험에서 틀리는지 확인)
    per_file = []
    for f_name in test_df["파일"].unique():
        m = (test_df["파일"] == f_name).values
        yt = y_test.values[m]
        for name, pred in [("교안 규칙(3σ OR)", rule_pred),
                           ("Random Forest(상대)", rf_pred), ("PCA 이상탐지", pca_test_pred)]:
            pp = pred[m]
            tn, fp, fn, tp = confusion_matrix(yt, pp, labels=[0, 1]).ravel()
            per_file.append({"파일": f_name, "모델": name, "행 수": int(m.sum()),
                             "실제 이상": int(yt.sum()), "예측 이상": int(pp.sum()),
                             "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
                             "Accuracy": round(accuracy_score(yt, pp), 4)})
    per_file_df = pd.DataFrame(per_file)
    per_file_df.to_csv(os.path.join(OUT, "03c_테스트_파일별_결과.csv"),
                       index=False, encoding="utf-8-sig")
    print("\n[참고] 테스트 파일별 결과")
    print(per_file_df.to_string(index=False))

    # Test07은 NG 팩이지만 라벨이 마지막 227시점(4.9%)에만 붙어 있어,
    # 나머지 구간은 '이상 패턴이 보이지만 정상으로 라벨된' 상태이다.
    # 라벨 정의 차이의 영향을 보기 위해 Test07을 제외한 지표도 함께 계산한다.
    ex = (test_df["파일"] != "Test07_NG_dchg").values
    ex_rows = [evaluate_classification_model("교안 규칙(3σ OR)", y_test[ex], rule_pred[ex],
                                             "테스트(Test07 제외)"),
               evaluate_classification_model("Random Forest(상대)", y_test[ex], rf_pred[ex],
                                             "테스트(Test07 제외)"),
               evaluate_classification_model("PCA 이상탐지", y_test[ex], pca_test_pred[ex],
                                             "테스트(Test07 제외)")]
    pd.DataFrame(ex_rows).round(4).to_csv(
        os.path.join(OUT, "03d_Test07제외_성능지표.csv"), index=False, encoding="utf-8-sig")

    # 변수 중요도 (교안 Ch40 : 의사결정나무의 장점 - 변수 중요도 확인)
    imp = pd.DataFrame({
        "Feature": REL_FEATURES,
        "Decision Tree": tree_model.feature_importances_,
        "Random Forest": forest_model.feature_importances_,
    }).sort_values("Random Forest", ascending=False).round(4)
    imp.to_csv(os.path.join(OUT, "05_변수중요도.csv"),
               index=False, encoding="utf-8-sig")
    print("\n변수 중요도 상위 5개")
    print(imp.head(5).to_string(index=False))

    # ---------------- 단계 5 : 이상 셀 순위 + 구간 ----------------
    # 셀 단위 이상 점수 = 이상 구간에서의 평균 z-score
    cell_rank_all = []
    for prefix, (z, label, cv_cols) in cell_z_store.items():
        if label.sum() == 0:
            continue
        score = z[label == 1].mean(axis=0)
        top = np.argsort(score)[::-1][:5]
        for rank, idx in enumerate(top, 1):
            cell_rank_all.append({"파일": prefix, "순위": rank,
                                  "셀": cv_cols[idx],
                                  "이상점수": round(float(score[idx]), 3)})
    cell_rank = pd.DataFrame(cell_rank_all)
    cell_rank.to_csv(os.path.join(OUT, "06_이상셀_순위.csv"),
                     index=False, encoding="utf-8-sig")
    print("\n파일별 이상 점수 상위 셀 (상위 2개만 표시)")
    print(cell_rank[cell_rank["순위"] <= 2].to_string(index=False))

    # 예측 결과 저장 (Dash 연동용)
    pred_out = test_df[["파일", "충방전", "판정", "시점", "label"]].copy()
    pred_out["규칙_예측"] = rule_pred
    pred_out["RF_예측"] = rf_pred
    pred_out["PCA_예측"] = pca_test_pred
    pred_out["PCA_이상점수_Q"] = np.round(Q_test, 4)
    pred_out["PCA_이상점수_T2"] = np.round(T2_test, 4)
    pred_out.to_csv(os.path.join(OUT, "07_테스트_예측결과.csv"),
                    index=False, encoding="utf-8-sig")

    # ---------------- 단계 5-2 : 시험(파일) 단위 품질 판정 ----------------
    # 현장에서는 "이 팩을 NG로 볼 것인가"가 최종 판단이다.
    # 시점 단위 이상 예측 비율이 기준을 넘으면 NG로 판정한다.
    # 기준 비율은 학습 파일에서만 정한다.
    def file_level(pred_col, frame, ratio):
        g = frame.groupby("파일").agg(이상비율=(pred_col, "mean"),
                                    실제=("판정", "first"))
        g["예측"] = np.where(g["이상비율"] >= ratio, "NG", "OK")
        return g

    train_eval = train_df.copy()
    train_eval["PCA_예측"] = pca_train_pred
    ratios = np.arange(0.05, 0.85, 0.05)
    best_ratio, best_hit = 0.05, -1
    for r in ratios:
        g = file_level("PCA_예측", train_eval, r)
        hit = (g["예측"] == g["실제"]).sum()
        if hit > best_hit:
            best_hit, best_ratio = hit, r
    test_eval = test_df.copy()
    test_eval["PCA_예측"] = pca_test_pred
    test_eval["RF_예측"] = rf_pred
    file_rows = []
    for col, name in [("PCA_예측", "PCA 이상탐지"), ("RF_예측", "Random Forest(상대)")]:
        g = file_level(col, test_eval, best_ratio).reset_index()
        g["모델"] = name
        file_rows.append(g)
    file_df = pd.concat(file_rows)[["모델", "파일", "이상비율", "실제", "예측"]].round(3)
    file_df["정답여부"] = np.where(file_df["실제"] == file_df["예측"], "O", "X")
    file_df.to_csv(os.path.join(OUT, "08_시험단위_판정결과.csv"),
                   index=False, encoding="utf-8-sig")
    print(f"\n[단계 5] 시험(파일) 단위 품질 판정 - NG 기준 이상비율 {best_ratio:.2f}")
    print(file_df.to_string(index=False))
    for name in file_df["모델"].unique():
        sub = file_df[file_df["모델"] == name]
        print(f"  {name}: {int((sub['정답여부']=='O').sum())}/{len(sub)} 파일 정답")

    # ---------------- 단계 6 : 시각화 (교안 Part 3) ----------------
    make_plots(df, res_df, imp, pred_out, cell_z_store, cell_rank)

    print("\n산출물 저장 위치:", OUT)
    return res_df


# =============================================================================
# 시각화 (Plotly : 교안 Part 3 Ch20/21, Dash 연동 고려)
# =============================================================================

def save_fig(fig, name):
    fig.write_html(os.path.join(OUT, name + ".html"), include_plotlyjs="cdn")
    try:
        fig.write_image(os.path.join(OUT, name + ".png"), width=1200, height=650, scale=2)
    except Exception as e:
        print("  (PNG 저장 생략:", e, ")")


def make_plots(df, res_df, imp, pred_out, cell_z_store, cell_rank):
    # 1) 충방전별 셀 전압 편차 분포 (OK vs NG)
    fig = px.box(df, x="충방전", y="셀전압_편차", color="판정",
                 title="충방전 구분별 셀 전압 편차 분포 (OK vs NG)",
                 labels={"셀전압_편차": "셀 전압 편차 (V)", "충방전": "충방전 구분"})
    save_fig(fig, "fig1_충방전별_셀전압편차")

    # 2) 충방전별 모듈 온도 편차 분포
    fig = px.box(df, x="충방전", y="모듈온도_편차", color="판정",
                 title="충방전 구분별 모듈 온도 편차 분포 (OK vs NG)",
                 labels={"모듈온도_편차": "모듈 온도 편차 (℃)", "충방전": "충방전 구분"})
    save_fig(fig, "fig2_충방전별_모듈온도편차")

    # 3) 셀 전압 Heatmap + 이상 구간 (기획서 '배터리 품질' 화면 핵심 출력)
    target = "Test07_NG_dchg"
    z, label, cv_cols = cell_z_store[target]
    step = max(1, len(z) // 300)                 # 화면 표시용 다운샘플
    zz = z[::step]
    fig = go.Figure(go.Heatmap(z=zz.T, x=np.arange(len(z))[::step],
                               y=cv_cols, colorscale="YlOrRd", zmin=0, zmax=6,
                               colorbar=dict(title="이상 점수<br>(z-score)")))
    a = np.where(label == 1)[0]
    if len(a):
        fig.add_vrect(x0=a.min(), x1=a.max(), line_width=0,
                      fillcolor="blue", opacity=0.15,
                      annotation_text="실제 이상 구간", annotation_position="top left")
    fig.update_layout(title=f"{target} 셀 전압 이상 점수 Heatmap (176셀 x 시점)",
                      xaxis_title="시점", yaxis_title="셀",
                      yaxis=dict(showticklabels=False))
    save_fig(fig, "fig3_셀전압_Heatmap")

    # 4) 주요 센서 패턴 + 정상/이상 구간 표시
    sub = df[df["파일"] == target]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["시점"], y=sub["셀전압_최소"],
                             name="셀 전압 최소", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=sub["시점"], y=sub["셀전압_평균"],
                             name="셀 전압 평균", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=sub["시점"], y=sub["셀전압_편차"] + 3.5,
                             name="셀 전압 편차(+3.5 오프셋)", line=dict(width=1)))
    if len(a):
        fig.add_vrect(x0=a.min(), x1=a.max(), line_width=0,
                      fillcolor="red", opacity=0.15,
                      annotation_text="실제 이상 구간", annotation_position="top left")
    p = pred_out[pred_out["파일"] == target]
    for s, e in _to_sequences(p["PCA_예측"].values):
        fig.add_vrect(x0=s, x1=e, line_width=0, fillcolor="blue", opacity=0.12)
    fig.update_layout(title=f"{target} 주요 센서 패턴과 정상/이상 구간"
                            " (빨강=실제 이상, 파랑=PCA 탐지)",
                      xaxis_title="시점", yaxis_title="전압 (V)")
    save_fig(fig, "fig4_이상구간_센서패턴")

    # 5) 모델 성능 비교
    m = res_df[res_df["구분"] == "테스트"].melt(
        id_vars="모델", value_vars=["Accuracy", "Precision", "Recall", "F1-score"],
        var_name="지표", value_name="값")
    fig = px.bar(m, x="지표", y="값", color="모델", barmode="group", text_auto=".3f",
                 title="테스트 데이터 기준 모델 성능 비교")
    fig.add_hline(y=0.7, line_dash="dash",
                  annotation_text="기획서 목표: Precision 0.7")
    fig.update_yaxes(range=[0, 1.05])
    save_fig(fig, "fig5_모델성능비교")

    # 6) 변수 중요도
    fig = px.bar(imp.head(10).melt(id_vars="Feature", var_name="모델", value_name="중요도"),
                 x="중요도", y="Feature", color="모델", barmode="group",
                 orientation="h", title="변수 중요도 상위 10개")
    fig.update_yaxes(autorange="reversed")
    save_fig(fig, "fig6_변수중요도")


def _to_sequences(binary):
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


if __name__ == "__main__":
    main()
