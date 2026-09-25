# 트랙 B 데이터 품질 및 분할 보고서 v2

## 원본 파일 검사

| file | rows | missing_cells | duplicate_rows | nonpositive_time_diffs | complete_cycles | positive_rows | realpower_zero_rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Training_Data | 135759 | 0 | 0 | 6 | 3481 | 0 | 39 |
| WeldingTest_01_OK | 1950 | 0 | 0 | 0 | 50 | 0 | 0 |
| WeldingTest_02_OK | 1872 | 0 | 0 | 0 | 48 | 0 | 0 |
| WeldingTest_03_NG | 1053 | 0 | 0 | 0 | 27 | 19 | 0 |
| WeldingTest_04_NG | 351 | 0 | 0 | 0 | 9 | 273 | 39 |

## 처리 규칙

- 필수 9개 열과 WorkingTime 날짜시간 변환을 검사했다.
- PageNo 1~39가 순서대로 존재하는 완전한 39행 용접 사이클만 허용했다.
- 결측·중복·음수값을 임의 보간하거나 삭제하지 않고 발견 수를 기록했다.
- NG04의 RealPower=0인 39행은 라벨이 있는 실제 이상 구간이므로 제거하지 않았다.
- 개발·검증은 행을 섞기 전에 cycle을 시간순으로 배정했다.
- 잠금 시험은 개발과 다른 파일 전체를 사용했다.

## 정상 이력 분할

- 완전 사이클: 3481
- 정상 기준 학습: 2436 cycles / 95004 rows
- 정상 임계값 보정: 1045 cycles / 40755 rows

## 지도 개발 클래스 비율

| split | cycles | rows | positive_rows | positive_rate |
| --- | --- | --- | --- | --- |
| development_train | 53 | 2067 | 16 | 0.0077 |
| development_validation | 24 | 936 | 3 | 0.0032 |

지도 모델은 희소한 이상 행을 보완하기 위해 class_weight='balanced' 또는 class_weight='balanced_subsample'을 사용했다.