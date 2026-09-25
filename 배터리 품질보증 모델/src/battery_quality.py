import glob
import pandas as pd
import numpy as np
import math

file_paths = sorted(glob.glob("data/*.csv"))
print("찾은 파일 개수:", len(file_paths))
print()

# 1.파일별 미리보기
df_list = []
for path in file_paths:
    df = pd.read_csv(path)
    print(f"=== {path} ===")
    print(df.head())
    print()
    df["파일명"] = path.split("/")[-1]
    df_list.append(df)

# 2.전체 파일을 하나로 합치기
all_df = pd.concat(df_list, ignore_index=True)

print("=" * 50)
print("전체 데이터 합친 결과")
print("=" * 50)
print()

# 3.데이터 구조 확인
print("전체 데이터 크기 (행, 열)")
print(all_df.shape)
print()

# 4.열 이름 확인
print("전체 열 이름")
print(all_df.columns.tolist())
print()

# 5.자료형 확인
print("자료형별 열 개수")
print(all_df.dtypes.value_counts())
print()
print("자료형 상세 (info)")
print(all_df.info())
print()

# 6.결측치 확인
print("전체 결측치 개수")
print(all_df.isna().sum().sum())
print()
print("열별 결측치 개수 (결측치 있는 열만)")
missing_by_col = all_df.isna().sum()
print(missing_by_col[missing_by_col > 0])
print()

# 7.중복값 확인
print("완전히 중복된 행 개수")
print(all_df.duplicated().sum())
print()

#null 값을 포함하고 있는 행 표시
null_data = all_df[all_df.isnull().any(axis=1)]
print(null_data)

#속성의 값이 오직 하나인 속성 제거
def removeCanstant(df,n):
    df = df[[col for col in df if df [col].nunique()>n]]
    return df
data1 = removeCanstant(all_df,1)
print(data1.shape)

#결측치 처리
def handleMissingValue(data):
    df = data.copy()
    numeric_df = df.select_dtypes(include='number').columns
    categorical_df = df.select_dtypes(include=['object', 'string', 'category']).columns

    for i in categorical_df:
        df[i] = df[i].fillna(df[i].mode()[0])

    df_flag_null = df.isnull()
    i, c = np.where(df_flag_null)

    for j in range(len(i)):
        if i[j] == 0:
            s = df.iloc[:, c[j]]
            id_s = s.notna().idxmax()
            val = df.iat[id_s, c[j]]
            df.iat[i[j], c[j]] = val

        elif i[j] == (len(df) - 1):
            s = df.iloc[:, c[j]]
            id_s = s.notna()[::-1].idxmax()
            val = df.iat[id_s, c[j]]
            df.iat[i[j], c[j]] = val

        else:
            low = df.iat[i[j] - 1, c[j]]
            high = df.iat[i[j] + 1, c[j]]

            if math.isnan(high):
                val = low
            else:
                val = (low + high) / 2
            df.iat[i[j], c[j]] = val   

    return df 

data2 = handleMissingValue(data1)
print("처리 후 남은 결측치 총 개수:", data2.isna().sum().sum())
print()
print("혹시 남은 게 있다면 어느 열인지:")
remaining = data2.isna().sum()
print(remaining[remaining > 0])

print("처리 전 결측치:", data1.isna().sum().sum())
data2 = handleMissingValue(data1)
print("처리 후 결측치:", data2.isna().sum().sum())