import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

# isolation forest + 이상탐지
# 학습용 데이터
train_df = pd.read_csv("data/pdm/raw_data/train/Training_Data.csv")

# 테스트용 데이터
test_df1 = pd.read_csv("data/pdm/raw_data/test/WeldingTest_01_OK.csv")
test_df2 = pd.read_csv("data/pdm/raw_data/test/WeldingTest_02_OK.csv")
test_df3 = pd.read_csv("data/pdm/raw_data/test/WeldingTest_03_NG.csv")
test_df4 = pd.read_csv("data/pdm/raw_data/test/WeldingTest_04_NG.csv")

# 라벨 데이터
label_df3 = pd.read_csv("data/pdm/test/WeldingTest_03_NG_Label.csv")
label_df4 = pd.read_csv("data/pdm/test/WeldingTest_04_NG_Label.csv")

# 컬럼정의
# [주의] 회귀 때와 다르게 target_column이 없다. RealPower도 "예측 대상"이 아니라
# Speed, Length 등과 똑같은 "입력값(Feature)"으로 같이 넣는다.
feature_columns = ["Speed", "Length", "SetPower", "GateOnTime", "RealPower"]
time_column = "WorkingTime"  # 정렬/그래프 x용


##################################
# 데이터 정제 및 전처리 함수
def preprocessing(df):

    # 데이터 종류 및 개수 확인
    print(df.head())
    print(df.columns)
    print(df.info())
    print(df.dtypes)
    print(df.describe())

    # 결측치 확인
    # 중복값 확인
    # 값의 범위와 단위 확인
    # 데이터 전처리
    df = df.rename(columns=lambda x: x.strip())  # 속성명 공백처리
    print(df.duplicated().sum())  # 중복값 확인
    print(df.isna().sum())  # 결측치 확인
    df = df.dropna()  # 결측치 제거
    # df["WorkingTime"] = pd.to_datetime(df["WorkingTime"]) #날짜 변환

    return df


# 라벨 붙이는 함수
# OK 파일은 라벨 파일이 없으므로 전체를 label=0으로 채우고,
# NG 파일은 라벨 파일을 '행 순서'로 맞춰 붙인다 (PageNo는 39개 단위로 반복되므로 merge 불가)
def attach_label(df, label_df=None):
    df = df.reset_index(drop=True)
    if label_df is None:
        df["label"] = 0
    else:
        label_df = label_df.reset_index(drop=True)
        label_col = "label" if "label" in label_df.columns else label_df.columns[-1]
        n = min(len(df), len(label_df))
        df = df.iloc[:n].copy()
        df["label"] = label_df[label_col].iloc[:n].astype(int).values
    return df


# 이상치 판별 함수
def identify_outliers(df, c):
    constant = float(c)
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    IQR_outliers = df[
        ((df.lt(Q1 - constant * IQR)) | (df.gt(Q3 + constant * IQR))).any(axis=1)
    ]  # lt = 보다 작은값, gt = 보다 큰 값
    IQR_outliers = pd.DataFrame(IQR_outliers)  # 이상치인 행
    return IQR_outliers


# 이상치 제거 함수
def remove_outliers(df, c):
    df = pd.DataFrame(df)
    outliers = identify_outliers(df, c)
    df_out = pd.DataFrame(outliers)
    df.drop(df_out.index, inplace=True)
    return df


# 결측치 제거 함수
##################################


train_df = preprocessing(train_df)
test_df1 = preprocessing(test_df1)
test_df2 = preprocessing(test_df2)
test_df3 = preprocessing(test_df3)
test_df4 = preprocessing(test_df4)

# 라벨 붙이기 (전처리 직후, concat 하기 전에 해야 함 - PageNo가 파일마다 반복되므로)
test_df1 = attach_label(test_df1)
test_df2 = attach_label(test_df2)
test_df3 = attach_label(test_df3, label_df3)
test_df4 = attach_label(test_df4, label_df4)

test_df = pd.concat(
    [test_df1, test_df2, test_df3, test_df4],
    ignore_index=True,
)

# Isolation Forest (이상 범위가 있는지 확인하기 위해 사용 - 회귀와 달리 target 없음)
train_model_df = train_df[feature_columns].dropna()
test_model_df = test_df[feature_columns + ["label"]].dropna()

X_train = train_model_df[feature_columns]
X_test = test_model_df[feature_columns]
y_true_label = test_model_df["label"].values  # 실제 양품/불량 (채점용 정답지)

isolation_forest_model = IsolationForest(n_estimators=100, contamination=0.001, random_state=42)
isolation_forest_model.fit(X_train)  # [주의] fit(X_train)만 있음. y_train 자체가 없음 (RealPower도, label도 안 줌)

train_raw_pred = isolation_forest_model.predict(X_train)  # 1=정상, -1=이상
test_raw_pred = isolation_forest_model.predict(X_test)
print(train_raw_pred[:20])
print(test_raw_pred[:20])

# 우리 표기(0=정상, 1=이상)로 변환
train_pred = (train_raw_pred == -1).astype(int)
anomaly_pred = (test_raw_pred == -1).astype(int)

# ---------------------------------------------------------------
# 평가 (회귀+잔차+Z-score 때와 동일한 방식 - anomaly_pred vs 실제 label 비교)
# 여기서는 잔차/Z-score 단계가 통째로 없다. Isolation Forest가 이미 -1/1을 바로 내놓기 때문.
# ---------------------------------------------------------------
accuracy = accuracy_score(y_true_label, anomaly_pred)
precision = precision_score(y_true_label, anomaly_pred, zero_division=0)
recall = recall_score(y_true_label, anomaly_pred, zero_division=0)
f1 = f1_score(y_true_label, anomaly_pred, zero_division=0)
cm = confusion_matrix(y_true_label, anomaly_pred, labels=[0, 1])

print("\n이상탐지 성능 (Isolation Forest)")
print(f"accuracy : {accuracy}")
print(f"precision : {precision}")
print(f"recall : {recall}")
print(f"f1 : {f1}")
print(f"cm : {cm}")