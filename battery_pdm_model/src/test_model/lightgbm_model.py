"""
배터리 레이저 용접 공정 이상 탐지 - LightGBM 회귀 + 잔차 Z-score

흐름
  1) 전처리 : 컬럼 공백 제거 → 결측/중복 확인 → 단일값 컬럼 제거(SetFrequency, SetDuty)
             → IQR(4배) 이상치 확인 → 학습 데이터의 미출력(RealPower=0) 행 제거
  2) EDA    : 공정 조건(SetPower)별 RealPower 분포, 월별 출력 추이(드리프트) 저장
  3) 학습   : 공정 조건(PageNo, Speed, Length, SetPower, GateOnTime) → RealPower 를 LGBMRegressor 로 예측
             (발주량 예측 모델과 같은 방식: 시간 순서 분할, regression_l1, early stopping → 전체 재학습)
  4) 이상판정: 잔차 = 실제값 - 예측값, 정상 학습데이터 out-of-fold 잔차의 SetPower 별 중앙값/robust 표준편차로 Z-score 계산
             |Z| >= 4.0 이면 이상 (가이드북 threshold 기준)
  5) 평가   : 회귀(MAE, RMSE, R2) / 분류(Accuracy, Precision, Recall, F1, 혼동행렬)
  6) 저장   : 대시보드용 CSV(행별 결과, 이상 구간 점검 목록, 조건별 요약, 지표), 그래프 PNG
"""
import sys
import warnings
from pathlib import Path

import importlib
import subprocess


# 필요한 패키지가 '지금 실행 중인 파이썬'에 없으면 그 파이썬에 바로 설치
# (VS Code ▶ 실행 파이썬과 터미널 pip 의 파이썬이 달라 ModuleNotFoundError 가 나는 문제 방지)
def ensure_package(import_name, pip_name=None):
    try:
        importlib.import_module(import_name)
    except ImportError:
        pip_name = pip_name or import_name
        print(f"[설치] '{pip_name}' 패키지가 없어 설치합니다 → {sys.executable}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
        importlib.invalidate_caches()


for _import_name, _pip_name in [
    ("numpy", None),
    ("pandas", None),
    ("matplotlib", None),
    ("sklearn", "scikit-learn"),
    ("lightgbm", None),
]:
    ensure_package(_import_name, _pip_name)

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")
import matplotlib

matplotlib.use("Agg")  # 창을 띄우지 않고 파일로만 저장
import matplotlib.pyplot as plt
import lightgbm as lgb
from lightgbm import LGBMRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ---------------------------------------------------------------
# 경로 (src/test_model/lightgbm.py 기준 → 배터리 예지보전 모델 폴더)
# ---------------------------------------------------------------
ROOT_PATH = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_PATH / "data"
TRAIN_DATA_PATH = DATA_PATH / "raw_data" / "train" / "Training_Data.csv"
TEST_FILES = {
    "WeldingTest_01_OK": (DATA_PATH / "raw_data" / "test" / "WeldingTest_01_OK.csv", None),
    "WeldingTest_02_OK": (DATA_PATH / "raw_data" / "test" / "WeldingTest_02_OK.csv", None),
    "WeldingTest_03_NG": (
        DATA_PATH / "raw_data" / "test" / "WeldingTest_03_NG.csv",
        DATA_PATH / "preprocessed" / "test" / "WeldingTest_03_NG_Label.csv",
    ),
    "WeldingTest_04_NG": (
        DATA_PATH / "raw_data" / "test" / "WeldingTest_04_NG.csv",
        DATA_PATH / "preprocessed" / "test" / "WeldingTest_04_NG_Label.csv",
    ),
}
OUTPUT_PATH = ROOT_PATH / "output" / "lightgbm"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------
# 컬럼 정의 / 설정값
# ---------------------------------------------------------------
target_column = "RealPower"  # 실제 용접 출력(W)
feature_columns = ["PageNo", "Speed", "Length", "SetPower", "GateOnTime"]  # 공정 조건
time_column = "WorkingTime"
condition_column = "SetPower"  # 공정 조건(설비 출력 설정) 필터 기준
Z_THRESHOLD = 4.0  # 가이드북 이상 판정 기준
VALID_RATIO = 0.2  # 학습 데이터 뒤쪽 20% (시간 순) 를 검증 구간으로 사용
N_FOLDS = 5  # 정상 잔차 기준을 만들 시간 블록 fold 수
SEED = 42


##################################
# 데이터 정제 및 전처리 함수
def preprocessing(df, name):
    df = df.rename(columns=lambda x: x.strip())  # 속성명 공백 처리
    print(f"\n[{name}] shape={df.shape}")
    print(f"  결측치 수: {int(df.isna().sum().sum())}, 중복 행 수: {int(df.duplicated().sum())}")

    df = df.dropna().drop_duplicates()
    df[time_column] = pd.to_datetime(df[time_column])
    df = df.sort_values(time_column, kind="stable").reset_index(drop=True)  # 시간 순 정렬

    # 값이 하나뿐인 컬럼은 정보가 없으므로 제거 (가이드북 removeConstant)
    constant_cols = [c for c in ["SetFrequency", "SetDuty"] if c in df.columns and df[c].nunique() <= 1]
    df = df.drop(columns=constant_cols)
    print(f"  단일값 컬럼 제거: {constant_cols}")
    return df


# 이상치 판별 함수 (IQR c배 범위를 벗어나는 행)
def identify_outliers(df, c):
    constant = float(c)
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    IQR_outliers = df[((df.lt(Q1 - constant * IQR)) | (df.gt(Q3 + constant * IQR))).any(axis=1)]
    return pd.DataFrame(IQR_outliers)


# 라벨 붙이는 함수 (OK 파일 = 전체 0, NG 파일 = 라벨 파일을 행 순서로 붙임)
def attach_label(df, label_path=None):
    df = df.reset_index(drop=True)
    if label_path is None:
        df["label"] = 0
        return df
    label_df = pd.read_csv(label_path).reset_index(drop=True)
    label_col = "label" if "label" in label_df.columns else label_df.columns[-1]
    n = min(len(df), len(label_df))
    df = df.iloc[:n].copy()
    df["label"] = label_df[label_col].iloc[:n].astype(int).values
    return df


# 공정 조건별 잔차 기준 계산 (robust: 중앙값, 1.4826*MAD)
# 평균/표준편차를 쓰면 드물게 나오는 용접 레시피(Speed 100~175) 예측 오차 몇 백 행 때문에
# 82/83 조건의 표준편차가 65W 까지 커져 이상을 놓치게 됨 → 이상값에 강한 중앙값/MAD 사용
def residual_stats(residual, condition):
    stats = pd.DataFrame({"residual": residual, condition_column: condition})
    stats = stats.groupby(condition_column)["residual"].agg(
        mean="median",
        std=lambda x: 1.4826 * np.median(np.abs(x - np.median(x))),
    )
    overall = (float(np.median(residual)), float(1.4826 * np.median(np.abs(residual - np.median(residual)))))
    return stats, overall


# 잔차 → Z-score
# 학습 때 없던 공정 조건(예: SetPower=35)은 가장 가까운 SetPower 의 기준을 사용
def calc_z_score(residual, condition, stats, overall):
    known = stats.index.to_numpy()
    nearest = np.array([known[np.argmin(np.abs(known - c))] for c in condition])
    mu = stats.loc[nearest, "mean"].to_numpy()
    sigma = stats.loc[nearest, "std"].to_numpy()
    sigma = np.where(sigma > 1e-9, sigma, 1e-9)
    return (residual - mu) / sigma


# 점검 사유 (설비 담당자가 바로 이해할 수 있는 문장)
def inspection_reason(row, known_conditions):
    if row["anomaly_pred"] == 0:
        return ""
    if row[target_column] <= 0:
        reason = "미출력(0W) - 레이저 발진/게이트 점검"
    elif row["z_score"] < 0:
        reason = "출력 저하 - 광학계/렌즈 오염, 레이저 소스 점검"
    else:
        reason = "출력 과다 - 출력 제어부/설정값 점검"
    if row[condition_column] not in known_conditions:
        reason += " + 학습에 없는 출력 설정값(SetPower) 확인"
    return reason


# 연속된 이상 행을 하나의 '이상 구간'으로 묶기 (대시보드 점검 목록)
def build_anomaly_segments(result_df):
    rows = []
    for file_name, g in result_df.groupby("file", sort=False):
        g = g.reset_index(drop=True)
        flag = g["anomaly_pred"].to_numpy()
        seg_id = np.cumsum(np.r_[1, np.diff(flag) != 0])
        for _, s in g[flag == 1].groupby(seg_id[flag == 1]):
            n_rows = len(s)
            main_reason = s["점검사유"].mode().iloc[0]
            if n_rows >= 39 or "미출력" in main_reason:
                priority = "긴급"  # 모듈 1개(39포인트) 이상 연속 or 미출력
            elif n_rows >= 3:
                priority = "높음"
            else:
                priority = "보통"  # 단발성
            rows.append({
                "file": file_name,
                "_pages": set(s["PageNo"].unique()),
                "시작시간": s[time_column].min(),
                "종료시간": s[time_column].max(),
                "이상행수": n_rows,
                "PageNo": ",".join(map(str, sorted(s["PageNo"].unique()))),
                "SetPower": ",".join(map(str, sorted(s[condition_column].unique()))),
                "평균_실제출력": round(s[target_column].mean(), 1),
                "평균_예측출력": round(s["predicted"].mean(), 1),
                "최대_|Z|": round(s["z_score"].abs().max(), 1),
                "점검사유": main_reason,
                "점검우선순위": priority,
                "실제_불량행수": int(s["label"].sum()),
            })
    seg = pd.DataFrame(rows)
    if seg.empty:
        return seg
    # 같은 용접 포인트(PageNo)에서 단발성 이상이 반복되면 → 해당 포인트 지그/위치/설정 점검 필요
    single = seg[seg["이상행수"] < 3]
    page_repeat = single.explode("_pages").groupby(["file", "_pages"]).size()
    seg["동일포인트_반복횟수"] = [
        max(page_repeat.get((f, p), 0) for p in pages) if n < 3 else 0
        for f, pages, n in zip(seg["file"], seg["_pages"], seg["이상행수"])
    ]
    seg.loc[(seg["점검우선순위"] == "보통") & (seg["동일포인트_반복횟수"] >= 3), "점검우선순위"] = "높음(동일 포인트 반복)"
    return seg.drop(columns="_pages")


def regression_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan,
    }


def classification_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "오경보율(FPR)": fp / (fp + tn) if (fp + tn) else 0.0,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
##################################


# ===============================================================
# 1. 데이터 불러오기 및 전처리
# ===============================================================
train_df = preprocessing(pd.read_csv(TRAIN_DATA_PATH), "Training_Data")

test_list = []
for file_name, (data_path, label_path) in TEST_FILES.items():
    df = preprocessing(pd.read_csv(data_path), file_name)
    df = attach_label(df, label_path)
    df["file"] = file_name
    test_list.append(df)
test_df = pd.concat(test_list, ignore_index=True)

# IQR 이상치 확인 (가이드북 c=4) - 공정 조건이 섞여 있어 전체 IQR로는 잘 안 잡힘을 확인
print(f"\nIQR(4배) 이상치 행 수(학습): {len(identify_outliers(train_df[feature_columns + [target_column]], 4.0))}")

# 학습 데이터에도 RealPower=0 인 모듈(39행)이 섞여 있음 → 정상 패턴 학습을 위해 제거
zero_rows = train_df[train_df[target_column] <= 0]
print(f"학습 데이터 미출력(RealPower=0) 행 제거: {len(zero_rows)}행, 발생일 {zero_rows[time_column].dt.date.unique()}")
train_df = train_df[train_df[target_column] > 0].reset_index(drop=True)


# ===============================================================
# 2. EDA - 공정 조건별 출력, 월별 출력 추이
# ===============================================================
condition_eda = train_df.groupby(condition_column)[target_column].agg(["count", "mean", "std", "min", "max"]).round(2)
print("\n[EDA] SetPower 별 RealPower\n", condition_eda)
condition_eda.to_csv(OUTPUT_PATH / "eda_condition_power.csv", encoding="utf-8-sig")

monthly_trend = (
    train_df.assign(month=train_df[time_column].dt.to_period("M").astype(str))
    .groupby(["month", condition_column])[target_column].agg(["mean", "std"]).round(2).reset_index()
)
print("\n[EDA] 월별 RealPower 추이 (출력 드리프트 확인)\n", monthly_trend.pivot(index="month", columns=condition_column, values="mean"))
monthly_trend.to_csv(OUTPUT_PATH / "eda_monthly_power_trend.csv", index=False, encoding="utf-8-sig")

recipe_eda = train_df.groupby(["Speed", "Length", "GateOnTime", condition_column]).size().rename("행수").reset_index()
print("\n[EDA] 용접 레시피(Speed/Length/GateOnTime/SetPower) 조합별 행 수\n", recipe_eda.to_string(index=False))
recipe_eda.to_csv(OUTPUT_PATH / "eda_recipe_count.csv", index=False, encoding="utf-8-sig")

corr = train_df[feature_columns + [target_column]].corr().round(3)
corr.to_csv(OUTPUT_PATH / "eda_correlation.csv", encoding="utf-8-sig")


# ===============================================================
# 3. LightGBM 학습 (발주량 예측 모델 방식: 시간 순서 분할 + early stopping → 전체 재학습)
# ===============================================================
def make_lightgbm(n_estimators):
    return LGBMRegressor(
        objective="regression_l1",  # MAE 기준 → 이상값 영향에 덜 민감
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=5.0,
        random_state=SEED,
        n_jobs=4,
        verbosity=-1,
    )


# 3-1. 트리 개수 결정: 앞 80% 학습 / 뒤 20% 검증 (시간 순서)
split_idx = int(len(train_df) * (1 - VALID_RATIO))
fit_df = train_df.iloc[:split_idx]
valid_df = train_df.iloc[split_idx:]
print(f"\n학습 구간: {fit_df[time_column].min()} ~ {fit_df[time_column].max()} ({len(fit_df)}행)")
print(f"검증 구간: {valid_df[time_column].min()} ~ {valid_df[time_column].max()} ({len(valid_df)}행)")

tuning_model = make_lightgbm(2000)
tuning_model.fit(
    fit_df[feature_columns],
    fit_df[target_column],
    eval_set=[(valid_df[feature_columns], valid_df[target_column])],
    eval_metric="mae",
    callbacks=[lgb.early_stopping(60, verbose=False)],
)
best_iteration = int(tuning_model.best_iteration_ or tuning_model.n_estimators)
valid_pred = tuning_model.predict(valid_df[feature_columns])
print(f"early stopping 결과 best_iteration={best_iteration}")

# 3-2. 정상 잔차 기준 만들기: 시간 블록 K-fold 로 out-of-fold 예측
#      (학습에 쓰지 않은 기간을 예측한 잔차 → 월별 출력 드리프트까지 포함한 '정상 변동폭')
oof_pred = np.zeros(len(train_df))
fold_id = np.arange(len(train_df)) * N_FOLDS // len(train_df)  # 시간 순으로 연속된 블록
for k in range(N_FOLDS):
    tr_mask, te_mask = fold_id != k, fold_id == k
    fold_model = make_lightgbm(best_iteration)
    fold_model.fit(train_df.loc[tr_mask, feature_columns], train_df.loc[tr_mask, target_column])
    oof_pred[te_mask] = fold_model.predict(train_df.loc[te_mask, feature_columns])

# 3-3. 최종 모델: 전체 학습 데이터로 재학습
lightgbm_model = make_lightgbm(best_iteration)
lightgbm_model.fit(train_df[feature_columns], train_df[target_column])
train_pred = lightgbm_model.predict(train_df[feature_columns])
test_pred = lightgbm_model.predict(test_df[feature_columns])
print("LightGBM 최종 학습 완료")


# ===============================================================
# 4. 잔차 → Z-score → 이상 판정
#    기준(mu, sigma)은 테스트가 아니라 '정상 학습 데이터 out-of-fold 잔차'로 고정 (데이터 누수 방지)
# ===============================================================
oof_residual = train_df[target_column].to_numpy() - oof_pred
stats, overall = residual_stats(oof_residual, train_df[condition_column].to_numpy())
print("\n[학습 데이터 OOF 기준] SetPower 별 잔차 중앙값/robust 표준편차\n", stats.round(3))
stats.to_csv(OUTPUT_PATH / "residual_baseline_by_condition.csv", encoding="utf-8-sig")

known_conditions = set(train_df[condition_column].unique())
result_df = test_df[["file", time_column] + feature_columns + [target_column, "label"]].copy()
result_df["predicted"] = test_pred
result_df["residual"] = result_df[target_column] - result_df["predicted"]
result_df["z_score"] = calc_z_score(result_df["residual"].to_numpy(), result_df[condition_column].to_numpy(), stats, overall)
result_df["anomaly_pred"] = (result_df["z_score"].abs() >= Z_THRESHOLD).astype(int)
# 대시보드 표시용 이상 점수(0~1): |Z| = 임계값(4) 일 때 0.5, |Z| 가 커질수록 1 에 가까워짐 (로지스틱 변환)
result_df["anomaly_prob"] = 1.0 / (1.0 + np.exp(-2.0 * (result_df["z_score"].abs() - Z_THRESHOLD)))
result_df["점검사유"] = result_df.apply(inspection_reason, axis=1, known_conditions=known_conditions)
result_df["점검필요"] = np.where(result_df["anomaly_pred"] == 1, "Y", "N")


# ===============================================================
# 5. 평가
# ===============================================================
reg_rows = [{"구간": "train(최종모델)", **regression_metrics(train_df[target_column], train_pred)},
            {"구간": "validation(시간분할)", **regression_metrics(valid_df[target_column], valid_pred)},
            {"구간": "train OOF(시간블록 5-fold)", **regression_metrics(train_df[target_column], oof_pred)}]
cls_rows = []
for file_name, g in result_df.groupby("file", sort=False):
    normal = g[g["label"] == 0]
    reg_rows.append({"구간": f"{file_name} (정상행)", **regression_metrics(normal[target_column], normal["predicted"])})
    reg_rows.append({"구간": f"{file_name} (전체)", **regression_metrics(g[target_column], g["predicted"])})
    cls_rows.append({"구간": file_name, **classification_metrics(g["label"], g["anomaly_pred"])})
cls_rows.append({"구간": "전체 테스트", **classification_metrics(result_df["label"], result_df["anomaly_pred"])})

reg_metrics_df = pd.DataFrame(reg_rows).round(4)
cls_metrics_df = pd.DataFrame(cls_rows).round(4)
print("\n회귀 성능 (RealPower 예측)\n", reg_metrics_df.to_string(index=False))
print("\n이상탐지 성능 (|Z| >= 4)\n", cls_metrics_df.to_string(index=False))
print("\n전체 테스트 혼동행렬 [[TN, FP], [FN, TP]]\n", confusion_matrix(result_df["label"], result_df["anomaly_pred"], labels=[0, 1]))

# 공정 조건별 요약 (대시보드: 설비 조건 필터)
condition_summary = (
    result_df.groupby(["file", condition_column])
    .agg(행수=("label", "size"), 평균_실제출력=(target_column, "mean"), 평균_예측출력=("predicted", "mean"),
         평균_잔차=("residual", "mean"), 이상판정수=("anomaly_pred", "sum"), 실제불량수=("label", "sum"))
    .round(2).reset_index()
)

# 변수 중요도
importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance_gain": lightgbm_model.booster_.feature_importance(importance_type="gain"),
    "importance_split": lightgbm_model.booster_.feature_importance(importance_type="split"),
}).sort_values("importance_gain", ascending=False)
print("\n변수 중요도\n", importance_df.to_string(index=False))

segments_df = build_anomaly_segments(result_df)
print(f"\n이상 구간(점검 목록) {len(segments_df)}건\n", segments_df.to_string(index=False) if len(segments_df) else "")


# ===============================================================
# 6. 결과 저장 (대시보드 연동용)
# ===============================================================
result_df.to_csv(OUTPUT_PATH / "lightgbm_predictions.csv", index=False, encoding="utf-8-sig")
segments_df.to_csv(OUTPUT_PATH / "lightgbm_anomaly_segments.csv", index=False, encoding="utf-8-sig")
condition_summary.to_csv(OUTPUT_PATH / "lightgbm_condition_summary.csv", index=False, encoding="utf-8-sig")
reg_metrics_df.to_csv(OUTPUT_PATH / "lightgbm_regression_metrics.csv", index=False, encoding="utf-8-sig")
cls_metrics_df.to_csv(OUTPUT_PATH / "lightgbm_classification_metrics.csv", index=False, encoding="utf-8-sig")
importance_df.to_csv(OUTPUT_PATH / "lightgbm_feature_importance.csv", index=False, encoding="utf-8-sig")
lightgbm_model.booster_.save_model(str(OUTPUT_PATH / "lightgbm_model.txt"))

# 그래프: 파일별 실제값/예측값 + Z-score
plt.rcParams["axes.unicode_minus"] = False
for file_name, g in result_df.groupby("file", sort=False):
    g = g.reset_index(drop=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(g.index, g[target_column], label="Actual RealPower", lw=1)
    axes[0].plot(g.index, g["predicted"], label="LightGBM Predicted", lw=1, alpha=0.8)
    axes[0].scatter(g.index[g["label"] == 1], g.loc[g["label"] == 1, target_column], s=25, c="red", label="True anomaly")
    axes[0].set_ylabel("RealPower (W)")
    axes[0].legend(loc="lower left")
    axes[0].set_title(f"{file_name} - Actual vs Predicted")
    axes[1].plot(g.index, g["z_score"].clip(-50, 50), lw=1, label="Z-score (clipped ±50)")
    axes[1].axhline(Z_THRESHOLD, c="red", ls="--", lw=1)
    axes[1].axhline(-Z_THRESHOLD, c="red", ls="--", lw=1, label=f"Threshold ±{Z_THRESHOLD}")
    axes[1].scatter(g.index[g["anomaly_pred"] == 1], g.loc[g["anomaly_pred"] == 1, "z_score"].clip(-50, 50), s=15, c="orange", label="Predicted anomaly")
    axes[1].set_xlabel("Row index (time order)")
    axes[1].set_ylabel("Z-score")
    axes[1].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH / f"lightgbm_{file_name}.png", dpi=120)
    plt.close(fig)

print(f"\n결과 저장 완료: {OUTPUT_PATH}")
