# 예지보전 재검증 보고서 v3

선택 기준모델: **RobustPhaseZ**. 운영 지도/비지도 조합은 validation에서 각각 선택 후 OR로 결합한다.
평가 성격: **재사용 파일 회고 평가**. 이전 실험에서 본 자료이며 새로운 외부 검증이 아니다.
현재 이상 탐지이며 미래 고장 시간·RUL을 예측하지 않는다.

## 데이터 처리

사이클 내부 시간 오류 1개와 0 출력 사이클 1개를 격리했다.
원본 ID를 유지하며 사이클 시작 시각으로 정렬했다. 정상 fit/calibration 경계가 분리됨을 검사했다.
주 평가의 정상 학습 자료는 02_OK보다 미래 자료를 포함한다. 전체를 미래 시점 검증으로 부르지 않는다.
별도 normal_only_forward_time 결과만 모든 시험 이전 자료로 적합·보정했다.

## 행 단위 회고 시험

| model | precision | recall | f1 | fp | fn | event_total | event_recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LogisticCurrent | 0.9927 | 1.0000 | 0.9964 | 2 | 0 | 1 | 1.0000 |
| RandomForestCurrent | 0.9941 | 0.6154 | 0.7602 | 1 | 105 | 1 | 1.0000 |
| LogisticHistoryOnly | 0.0000 | 0.0000 | 0.0000 | 50 | 273 | 1 | 0.0000 |
| RobustPhaseZ | 0.9891 | 1.0000 | 0.9945 | 3 | 0 | 1 | 1.0000 |
| LightGBMResidual | 0.9579 | 1.0000 | 0.9785 | 12 | 0 | 1 | 1.0000 |
| IsolationForestNormal | 0.6250 | 0.0366 | 0.0692 | 6 | 263 | 1 | 1.0000 |

## 사이클 및 운영 경보

| model | cycle_precision | cycle_recall | cycle_f1 | cycle_fpr | false_alarm_count |
| --- | --- | --- | --- | --- | --- |
| LogisticCurrent | 0.8750 | 1.0000 | 0.9333 | 0.0200 | 2 |
| RandomForestCurrent | 0.8750 | 1.0000 | 0.9333 | 0.0200 | 1 |
| LogisticHistoryOnly | 0.0000 | 0.0000 | 0.0000 | 0.9400 | 49 |
| RobustPhaseZ | 0.8750 | 1.0000 | 0.9333 | 0.0200 | 2 |
| LightGBMResidual | 0.8750 | 1.0000 | 0.9333 | 0.0200 | 2 |
| IsolationForestNormal | 0.5000 | 0.8571 | 0.6316 | 0.1200 | 6 |
| LogisticCurrent OR RobustPhaseZ | 0.8750 | 1.0000 | 0.9333 | 0.0200 | 2 |
| LogisticCurrent OR LightGBMResidual | 0.8750 | 1.0000 | 0.9333 | 0.0200 | 2 |
| LogisticCurrent OR IsolationForestNormal | 0.5000 | 1.0000 | 0.6667 | 0.1400 | 8 |
| RandomForestCurrent OR RobustPhaseZ | 0.8750 | 1.0000 | 0.9333 | 0.0200 | 2 |
| RandomForestCurrent OR LightGBMResidual | 0.8750 | 1.0000 | 0.9333 | 0.0200 | 2 |
| RandomForestCurrent OR IsolationForestNormal | 0.5000 | 1.0000 | 0.6667 | 0.1400 | 7 |
| LogisticHistoryOnly OR RobustPhaseZ | 0.1296 | 1.0000 | 0.2295 | 0.9400 | 50 |
| LogisticHistoryOnly OR LightGBMResidual | 0.1296 | 1.0000 | 0.2295 | 0.9400 | 50 |
| LogisticHistoryOnly OR IsolationForestNormal | 0.1132 | 0.8571 | 0.2000 | 0.9400 | 55 |

사이클은 한 행 이상 이상이면 양성이다. 경보는 첫 양성 행에서 시작하고 정상 행 또는 120초 초과 수집 공백에서 종료한다.
관측시간당 오경보는 0~120초의 유효한 인접 간격만 사용한 보조 지표이며 실제 가동시간이 아니다.
사이클 FPR의 현장 승인 한도는 미정이다. 행 FPR 1%와 같은 지표로 비교하지 않는다.

## 추가 시간 검증: 정상 전용 기준모델

| source_file | precision | recall | f1 | fp | fn |
| --- | --- | --- | --- | --- | --- |
| WeldingTest_01_OK | 0.0000 | 0.0000 | 0.0000 | 1950 | 0 |
| WeldingTest_02_OK | 0.0000 | 0.0000 | 0.0000 | 1872 | 0 |
| WeldingTest_03_NG | 0.0180 | 1.0000 | 0.0354 | 1034 | 0 |
| WeldingTest_04_NG | 0.9286 | 1.0000 | 0.9630 | 21 | 0 |

별도 기준모델의 시간 검증이며 주 모델 선택에는 사용하지 않았다. 고장 종류와 설비 일반화의 외부 검증을 대체하지 않는다.

## 한계와 선택

지도 validation 이상은 3건, 회고 시험 이상은 1개의 연속 사건이다. 행/사이클 수를 독립 사건 수로 해석하지 않는다.
동률은 사전에 고정한 단순성 순서를 사용했다. 측정 학습시간이나 역방향 시험 성적을 선택에 사용하지 않았다.
개발 분할·정상 분위수 민감도는 sensitivity_metrics.csv, 반대 파일 방향은 reverse_file_stress_metrics.csv에 보존했다.
반대 방향 개발은 NG04의 단일 사건을 나누므로 사건 독립 검증이 아니다.
외부 신규 사건·현장 SOP·단위 및 점검 비용 기준은 추가 확보가 필요하다.

## 재현성

run_manifest.json에 계획·코드·입력·모델 해시, 환경 버전, 선택 시각을 기록했다.
selection.json은 회고 시험 모델 추론 전에 저장했다. 학습 산출물 경로는 기존 연동을 위해 track_b_final_v2를 유지하며 모델 버전은 track-b-audited-v3다.
