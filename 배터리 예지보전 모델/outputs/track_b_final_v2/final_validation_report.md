# 트랙 B 지도·비지도 최종 검증 보고서 v2

검증 세트에서 선택한 최종 모델: **RobustPhaseZ**

## 평가 설계

- 개발 파일: WeldingTest_01_OK, WeldingTest_03_NG.
- 독립 잠금 시험 파일: WeldingTest_02_OK, WeldingTest_04_NG.
- 잠금 시험 파일은 학습, 임계값 결정, 모델 선택에 사용하지 않았다.
- 개발 파일 내부도 39행 용접 사이클을 유지하고 시간순으로 분할했다.
- 비지도 모델은 Training_Data 정상 구간만 학습하고 정상 보정 구간으로 임계값을 고정했다.
- 지도 모델은 희소한 이상 행에 balanced class weight를 적용했다.
- 이벤트는 원본 파일의 연속 이상 구간으로 계산하여 NG04의 한 이벤트를 여러 cycle 이벤트로 부풀리지 않았다.

## 개발 validation 결과

| model | family | precision | recall | f1 | false_positive_rate | event_recall | false_alarm_events |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LogisticCurrent | supervised | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0 |
| RandomForestCurrent | supervised | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0 |
| LogisticHistoryOnly | supervised | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0 |
| RobustPhaseZ | unsupervised | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0 |
| IsolationForestNormal | unsupervised | 0.0000 | 0.0000 | 0.0000 | 0.0118 | 0.0000 | 11 |

## 독립 파일 잠금 시험 결과

| model | family | accuracy | precision | recall | f1 | tn | fp | fn | tp | event_recall | false_alarm_events | mean_detection_delay_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LogisticCurrent | supervised | 0.9991 | 0.9927 | 1.0000 | 0.9964 | 1948 | 2 | 0 | 273 | 1.0000 | 2 | 0.0000 |
| RandomForestCurrent | supervised | 0.9523 | 0.9941 | 0.6154 | 0.7602 | 1949 | 1 | 105 | 168 | 1.0000 | 1 | 0.0000 |
| LogisticHistoryOnly | supervised | 0.8920 | 0.9714 | 0.1245 | 0.2208 | 1949 | 1 | 239 | 34 | 1.0000 | 1 | 12.0000 |
| RobustPhaseZ | unsupervised | 0.9964 | 0.9715 | 1.0000 | 0.9856 | 1942 | 8 | 0 | 273 | 1.0000 | 7 | 0.0000 |
| IsolationForestNormal | unsupervised | 0.9316 | 0.8667 | 0.5238 | 0.6530 | 1928 | 22 | 130 | 143 | 1.0000 | 22 | 0.0000 |

## 최종 모델의 파일별 성능

| source_file | rows | precision | recall | f1 | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WeldingTest_02_OK | 1872 | 0.0000 | 0.0000 | 0.0000 | 1872 | 0 | 0 | 0 |
| WeldingTest_04_NG | 351 | 0.9715 | 1.0000 | 0.9856 | 70 | 8 | 0 | 273 |

## 반대 방향 파일 스트레스 테스트

이 표는 02_OK+04_NG로 개발하고 01_OK+03_NG 전체를 시험한 민감도 분석이다. NG04에 독립된 두 번째 이상 이벤트가 없어 내부 validation이 같은 장기 이벤트를 나누므로 최종 모델 선택에는 사용하지 않았다.

| model | family | precision | recall | f1 | fn | fp | event_recall | false_alarm_events |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LogisticCurrent | supervised | 0.0000 | 0.0000 | 0.0000 | 19 | 0 | 0.0000 | 0 |
| RandomForestCurrent | supervised | 0.0000 | 0.0000 | 0.0000 | 19 | 0 | 0.0000 | 0 |
| LogisticHistoryOnly | supervised | 0.0000 | 0.0000 | 0.0000 | 19 | 0 | 0.0000 | 0 |
| RobustPhaseZ | unsupervised | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 1.0000 | 0 |
| IsolationForestNormal | unsupervised | 0.0000 | 0.0000 | 0.0000 | 19 | 44 | 0.0000 | 44 |

## 성공 기준 판정

- 불량 행 Recall >= 0.90: PASS (1.0000)
- 물리 이벤트 Recall = 1.00: PASS (1.0000)
- 정상 행 FPR <= 0.01: PASS (0.0041)
- 독립 시험 FN: 0, FP: 8

## 높은 성능에 대한 점검

- 잠금 시험은 학습과 다른 파일 및 다른 NG 유형 전체를 사용하므로 이전 cycle 혼합 평가보다 강하다.
- Current 모델은 현재 RealPower를 사용하므로 현재 이상 감지 성능이다. 미래 고장 예측 성능으로 해석하지 않는다.
- HistoryOnly 결과를 함께 제시하여 현재 측정값을 제거했을 때의 성능 저하를 확인한다.
- 비지도 RobustPhaseZ가 높은 성능을 보이면 정상 PageNo별 RealPower 범위와 NG04의 분리가 매우 크다는 데이터 특성 때문이다.

## 주요 특징

- LogisticCurrent: RelativePowerError (0.279), PhaseSignedZ (0.276), PhaseAbsZ (0.276), RealPowerDelta (0.062), TimeGapSeconds (0.021)
- LogisticHistoryOnly: PageSin (0.370), SetPower (0.287), TimeGapSeconds (0.117), PageNo (0.095), GateOnTime (0.085)
- RandomForestCurrent: RealPower (0.210), RelativePowerError (0.187), PhaseSignedZ (0.169), PhaseAbsZ (0.146), SetPower (0.104)

## 데이터 규모

- 정상 기준 학습: 2436 cycles / 95004 rows
- 정상 임계값 보정: 1045 cycles / 40755 rows
- 지도 개발 train: 53 cycles
- 지도 개발 validation: 24 cycles
- 잠금 시험: WeldingTest_02_OK 48 cycles + WeldingTest_04_NG 9 cycles.

## 한계

- 독립 시험 NG 파일이 하나이며 NG04에는 장기 이상 이벤트가 한 건뿐이다.
- NG03과 NG04 외의 새로운 고장 모드에 대한 외부 타당성은 확인되지 않았다.
- 현 데이터에는 고장 이전을 명시하는 horizon 라벨이 없어 사전 예측 시간을 직접 평가할 수 없다.
- 신규 시점·신규 설비 파일을 추가 확보해 파일 단위 반복 검증을 해야 한다.