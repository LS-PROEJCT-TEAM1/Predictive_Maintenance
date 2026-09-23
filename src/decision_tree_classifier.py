import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
# decision + 분류
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
target_column = "label"  # 양품 0, 불량 1
feature_columns = ["Speed", "Length", "SetPower", "GateOnTime", "RealPower"]


# 데이터 정제 및 전처리
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


# 라벨 파일이 없는 파일(Training_Data, OK 파일들)은 전체를 label=0으로 채움
def load_all_normal(df):
    df["label"] = 0
    return df


# 라벨 파일이 있는 NG 파일은 '행 순서'로 라벨을 맞춰 붙임
# PageNo가 39씩 사이클을 돌기 때문에 PageNo로 merge 할 수 없음
def load_with_positional_label(df, label_df):
    df = df.reset_index(drop=True)
    label_df = label_df.reset_index(drop=True)
    label_col = "label" if "label" in label_df.columns else label_df.columns[-1]

    if len(df) != len(label_df):
        print(
            f"[경고] 행 개수가 다릅니다: 데이터 {len(df)}행 vs 라벨 {len(label_df)}행. 짧은 쪽에 맞춥니다."
        )

    n = min(len(df), len(label_df))
    df = df.iloc[:n].copy()
    df["label"] = label_df[label_col].iloc[:n].astype(int).values

    return df


"""
reference_df(Training_Data) 기준으로 고정값 컬럼(SetFrequency, SetDuty 등)을 찾아 모두 제거.
단, label(target)은 Training_Data에서 전부 0이라 '고정값'처럼 보이지만 실제로는
다른 파일(NG3/NG4)에서 1도 나오는 target 컬럼이므로 절대 제거 대상에서 제외한다.
"""
def drop_constant_columns(df_list, reference_df):
    protected_cols = {target_column}
    constant_cols = [
        c
        for c in reference_df.columns
        if c not in protected_cols and reference_df[c].nunique() <= 1
    ]
    if constant_cols:
        print(f"[정보] 값이 고정되어 제거되는 컬럼: {constant_cols}")
    return [df.drop(columns=constant_cols, errors="ignore") for df in df_list]

"""
# 회귀 모델
def evaluate_classification_model(model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(
        f"[{model_name}] Accuracy={accuracy:.4f}  Precision={precision:.4f}  "
        f"Recall={recall:.4f}  F1={f1:.4f}"
    )
    print("Confusion Matrix ([[TN, FP], [FN, TP]])")
    print(cm)
    return {
        "모델": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0],
        "TP": cm[1, 1],
    }
"""

train_df = load_all_normal(preprocessing(train_df))
test_df1 = load_all_normal(preprocessing(test_df1))
test_df2 = load_all_normal(preprocessing(test_df2))
test_df3 = load_with_positional_label(preprocessing(test_df3), preprocessing(label_df3))
test_df4 = load_with_positional_label(preprocessing(test_df4), preprocessing(label_df4))


train_df, test_df1, test_df2, test_df3, test_df4 = drop_constant_columns(
    [train_df, test_df1, test_df2, test_df3, test_df4], reference_df=train_df
)

for name, df in [
    ("Training_Data", train_df),
    ("WeldingTest_01_OK", test_df1),
    ("WeldingTest_02_OK", test_df2),
    ("WeldingTest_03_NG", test_df3),
    ("WeldingTest_04_NG", test_df4),
]:
    print(f"  {name}: {len(df)}행, 불량(label=1) {int(df['label'].sum())}개")


train_model_df = pd.concat([train_df, test_df3], ignore_index=True)[
    feature_columns + [target_column]
].dropna()
test_model_df = pd.concat([test_df1, test_df2, test_df4], ignore_index=True)[
    feature_columns + [target_column]
].dropna()

X_train, y_train = train_model_df[feature_columns], train_model_df[target_column]
X_test, y_test = test_model_df[feature_columns], test_model_df[target_column]

decision_model = DecisionTreeClassifier(
    max_depth=4, random_state=42, class_weight="balanced"
)
decision_model.fit(X_train, y_train)
train_pred = decision_model.predict(X_train)
test_pred = decision_model.predict(X_test)
print(train_pred[:20])
print(test_pred[:20])

# 정확성, 정밀도, 재현율
train_accuracy = accuracy_score(y_train, train_pred)
train_precision = precision_score(y_train, train_pred, zero_division=0)
train_recall = recall_score(y_train, train_pred, zero_division=0)
train_f1 = f1_score(y_train, train_pred, zero_division=0)
train_cm = confusion_matrix(y_train, train_pred, labels=[0, 1])

test_accuracy = accuracy_score(y_test, test_pred)
test_precision = precision_score(y_test, test_pred, zero_division=0)
test_recall = recall_score(y_test, test_pred, zero_division=0)
test_f1 = f1_score(y_test, test_pred, zero_division=0)
test_cm = confusion_matrix(y_test, test_pred, labels=[0, 1])

print("학습 성능")
print(f"accuracy : {train_accuracy}")
print(f"precision : {train_precision}")
print(f"recall : {train_recall}")
print(f"f1 : {train_f1}")
print(f"cm : {train_cm}")
print()
print("테스트 성능")
print(f"accuracy : {test_accuracy}")
print(f"precision : {test_precision}")
print(f"recall : {test_recall}")
print(f"f1 : {test_f1}")
print(f"cm : {test_cm}")
