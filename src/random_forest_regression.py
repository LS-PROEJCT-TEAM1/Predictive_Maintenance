import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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

# random forest + 회귀 -> 잔차 -> Z-score -> anomaly_pred
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
target_column = "RealPower"
feature_columns = ["Speed", "Length", "SetPower", "GateOnTime"]
time_column = "WorkingTime"  # 정렬/그래프 x용
Z_THRESHOLD = 4.0


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

##################################


train_df = preprocessing(train_df)
test_df1 = preprocessing(test_df1)
test_df2 = preprocessing(test_df2)
test_df3 = preprocessing(test_df3)
test_df4 = preprocessing(test_df4)

test_df1 = attach_label(test_df1)
test_df2 = attach_label(test_df2)
test_df3 = attach_label(test_df3, label_df3)
test_df4 = attach_label(test_df4, label_df4)

test_df = pd.concat(
    [test_df1, test_df2, test_df3, test_df4],
    ignore_index=True,
)

# Random Forest Tree (회귀 = 이상 범위가 있는지 확인하기 위해 사용)
train_model_df = train_df[feature_columns + [target_column]].dropna()
test_model_df = test_df[feature_columns + [target_column, "label"]].dropna()

X_train, y_train = train_model_df[feature_columns], train_model_df[target_column]
X_test, y_test = test_model_df[feature_columns], test_model_df[target_column]
y_true_label = test_model_df["label"].values

decision_tree_model = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
decision_tree_model.fit(X_train, y_train)
test_pred = decision_tree_model.predict(X_test)
train_pred = decision_tree_model.predict(X_train)
print(test_pred[:20])
print(train_pred[:20])

# 실제값 / 예측값 / 오차 / 절대오차 비교
result_df = pd.DataFrame({"실제값": y_test.values, "예측값": test_pred})
result_df["오차"] = result_df["실제값"] - result_df["예측값"]
result_df["절대오차"] = result_df["오차"].abs()
print(result_df)

# 회귀
train_mae = mean_absolute_error(y_train, train_pred)
train_rmse = mean_squared_error(y_train, train_pred) ** 0.5
train_r2 = r2_score(y_train, train_pred)

test_mae = mean_absolute_error(y_test, test_pred)
test_rmse = mean_squared_error(y_test, test_pred) ** 0.5
test_r2 = r2_score(y_test, test_pred)

print("학습 성능")
print(f"mae : {train_mae}")
print(f"rmse : {train_rmse}")
print(f"r2 : {train_r2}")

print("테스트 성능")
print(f"mae : {test_mae}")
print(f"rmse : {test_rmse}")
print(f"r2 : {test_r2}")


# ---------------------------------------------------------------
train_residual = train_pred - y_train.values
mu_fixed, sigma_fixed = train_residual.mean(), train_residual.std()
print(f"\n[학습 데이터 기준] 잔차 평균={mu_fixed:.3f}, 표준편차={sigma_fixed:.3f}")
 
test_residual = test_pred - y_test.values
z_score = (test_residual - mu_fixed) / (sigma_fixed if sigma_fixed > 1e-9 else 1e-9)
anomaly_pred = (np.abs(z_score) > Z_THRESHOLD).astype(int)
 
accuracy = accuracy_score(y_true_label, anomaly_pred)
precision = precision_score(y_true_label, anomaly_pred, zero_division=0)
recall = recall_score(y_true_label, anomaly_pred, zero_division=0)
f1 = f1_score(y_true_label, anomaly_pred, zero_division=0)
cm = confusion_matrix(y_true_label, anomaly_pred, labels=[0, 1])
 
print("\n이상탐지 성능 (Z-score 기반)")
print(f"accuracy : {accuracy}")
print(f"precision : {precision}")
print(f"recall : {recall}")
print(f"f1 : {f1}")
print(f"cm : {cm}")