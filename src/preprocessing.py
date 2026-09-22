import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 전처리
test_df1 = pd.read_csv('data/raw_data/test/WeldingTest_01_OK.csv')
test_df2 = pd.read_csv('data/raw_data/test/WeldingTest_02_OK.csv')
test_df3 = pd.read_csv('data/raw_data/test/WeldingTest_03_NG.csv')
test_df4 = pd.read_csv('data/raw_data/test/WeldingTest_04_NG.csv')

print(test_df1.head())
