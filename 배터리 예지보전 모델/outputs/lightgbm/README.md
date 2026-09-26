# LightGBM 정상 출력 회귀 기반 용접 이상 탐지

> 코드: `src/lightgbm_model.py`
> 원본: `src/final_three_model_comparison.py`의 3개 모델(RobustZ / LightGBM / N-HiTS) 공통 비교 중 **LightGBM 부분을 추출**하고, 대시보드는 **LightGBM 단일 모델**로 구성
> 결과 저장 위치: `outputs/lightgbm/`

---

## 1. 추출 범위와 원본 대비 변경 사항

| 원본 위치 | 추출한 내용 |
|---|---|
| `final_three_model_comparison.py` | `split_all_cycles`(사이클 70/30 분할), `calc_binary_metrics`, `regions`, `event_summary`, `main`의 LightGBM 학습·평가 흐름 |
| `compare_models.py` | `read_signal`, `add_cycle_id`, `quantile_threshold`, `phase_location_scale`, `regression_features`, `LightGBMDetector` |

- 원본은 `compare_models.py`와 `reproduce_guidebook.py`를 import하고, 여기에 torch·N-HiTS 의존성이 있다. 추출본은 필요한 함수를 파일 안에 모두 넣어서 **lightgbm만 있으면 단독으로 실행**된다.
- 분할 방식, Feature, 임계값 규칙은 원본과 같다.

**원본 대비 변경 사항**

| 항목 | 원본 | 현재 | 이유 |
|---|---|---|---|
| huber `alpha` | 기본값 0.9 | **50.0** | 기본값은 출력(W) 단위에 비해 너무 작아 회귀가 학습되지 않았다 (정상 행 MAE 약 454W → **6.4W**) |
| 임계값 (보정 99.9 백분위) | 5.5642 | **3.6564** | 모델이 바뀌면서 이상 점수 분포가 달라짐 |
| 전체 TN / FP / FN / TP | 4926 / 8 / 0 / 292 | **4921 / 13 / 0 / 292** | 7장 참고 |
| 이벤트 탐지 | 20 / 20 | **20 / 20** | 같음 |

**추가한 부분**

- 필요한 패키지가 없을 때 실행 중인 파이썬에 자동으로 설치하는 코드
- 대시보드 연동용 산출물 12종과 점검 이력 템플릿 생성 (8장)

---

## 2. 데이터

| 파일 | 행 수 | 용도 | 라벨 |
|---|---|---|---|
| `raw_data/train/Training_Data.csv` | 135,759 (3,481개 사이클) | 학습 + 정상 보정 | 없음 (정상 운전 가정) |
| `raw_data/test/WeldingTest_01_OK.csv` | 1,950 | 테스트 | 전체 0 |
| `raw_data/test/WeldingTest_02_OK.csv` | 1,872 | 테스트 | 전체 0 |
| `raw_data/test/WeldingTest_03_NG.csv` | 1,053 | 테스트 (단발성 이상) | `preprocessed/test/WeldingTest_03_NG_Label.csv` (이상 19행 = 이벤트 19개) |
| `raw_data/test/WeldingTest_04_NG.csv` | 351 | 테스트 (연속 구간 이상) | `preprocessed/test/WeldingTest_04_NG_Label.csv` (이상 273행 = 이벤트 1개) |

1개 사이클은 배터리모듈 1개에 해당하며, 39개 용접 포인트(PageNo 1~39)로 이루어진다.

---

## 3. 전처리 방법

| 순서 | 처리 | 근거 |
|---|---|---|
| 1 | 컬럼명 앞뒤 공백 제거 (`' RealPower'` → `'RealPower'`) | 원본 CSV 컬럼명에 공백이 있음 |
| 2 | `WorkingTime`을 datetime으로 변환 (`errors="raise"`) | 시간 순서 분할과 TimeGap 계산에 필요. 형식이 틀리면 즉시 오류 |
| 3 | 사이클 ID 부여: PageNo가 1로 돌아올 때마다 새 사이클 | 용접 공정이 39포인트 단위로 반복되기 때문 |
| 4 | 사이클 완전성 검사: 39행, PageNo 1~39 순서인지 확인. 하나라도 어긋나면 오류로 중단 | 학습 데이터 3,481개 사이클이 모두 완전함 |
| 5 | **행을 제거하지 않음** | 결측·중복 0건, IQR 4배 검사 0건. RealPower=0 사이클(2022-03-18) 1개도 그대로 포함 |
| 6 | 시간 순 사이클 분할: 앞 70%는 학습, 뒤 30%는 정상 보정 | 미래 데이터로 과거를 학습하지 않도록 시간 순서로 분할 (데이터 누수 방지) |

**분할 결과**

| 구분 | 기간 | 사이클 | 행 |
|---|---|---|---|
| 학습 | 2022-01-08 ~ 2022-06-10 | 2,436 | 95,004 |
| 정상 보정 | 2022-06-10 ~ 2022-07-29 | 1,045 | 40,755 |

학습 구간에서 앞 85%로 트리를 학습하고, 뒤 15%로 early stopping(30) 검증을 한다.

> 가이드북의 `RealPower < 1300 → +930.5` 변환은 N-HiTS 전용이라 LightGBM에는 적용하지 않는다.

---

## 4. EDA

이 스크립트에는 EDA 코드가 없다. 모델 설계의 근거가 된 데이터 특성은 다음과 같다 (`Training_Data.csv` 기준).

| 확인 항목 | 결과 | 모델 설계에 반영한 방식 |
|---|---|---|
| 단일값 컬럼 | `SetFrequency`=1000, `SetDuty`=100 | Feature에서 제외 |
| SetPower별 RealPower | 38% ≈ 688W, 82% ≈ 1690W, 83% ≈ 1711W (3개 그룹) | 전체 분포 대신 **PageNo별 기준**으로 점수 표준화 |
| PageNo와 조건의 관계 | 같은 PageNo는 항상 같은 SetPower·Speed·Length 레시피 | PageNo별 중앙값·scale 사용이 타당 |
| 상관관계 | RealPower와 SetPower 1.00, Speed·Length 0.997, GateOnTime 0.885 | 공정 조건으로 출력을 예측하는 회귀 접근 |
| 월별 출력 추이 | 2022-01 → 07에 38%는 약 -9W, 83%는 약 -22W 하락 | 시간 순 분할에서 보정 구간(6~7월)이 최근 수준을 반영 |
| RealPower=0 | 1개 사이클(39행) | 정상 데이터에 섞인 이상이지만 원본 규칙에 따라 유지 |
| 테스트 이상 패턴 | 03_NG는 PageNo 14에서 SetPower 35%, 출력 약 630W인 단발성 이상. 04_NG는 16:35 이후 0W 또는 약 1200W인 연속 이상 | 이벤트 단위 탐지 지표를 함께 평가 |

---

## 5. 모델 구성 – Feature / Target / 회귀·분류

### Feature (9개, `regression_features`)

| Feature | 설명 |
|---|---|
| `PageNo` | 용접 포인트 순번 (1~39) |
| `Speed` | 용접 속도 설정 (mm/s) |
| `Length` | 용접 길이 설정 (mm) |
| `SetPower` | 용접 출력 설정 (%) |
| `GateOnTime` | 용접 게이트 오픈 시간 |
| `PreviousPower` | **직전 행의 RealPower**. 사이클 첫 행(PageNo=1)은 학습 데이터 PageNo 39의 중앙값으로 대체 |
| `TimeGap` | 직전 행과의 시간 간격(초), 0~120으로 제한 |
| `PageSin`, `PageCos` | PageNo를 원형으로 표현: sin, cos(2π·(PageNo−1)/39) |

### Target

| 구분 | 컬럼 | 설명 |
|---|---|---|
| **회귀 Target** | `RealPower` | 실제 용접 출력 (W) |
| **분류 결과** | `anomaly_pred` | 0 = 정상, 1 = 이상 (이상 점수 > 임계값) |
| 평가용 정답 | `label` | NG 라벨 파일. **학습과 임계값 결정에는 사용하지 않음** |

### 회귀 모델: `LGBMRegressor`

| 파라미터 | 값 |
|---|---|
| objective | `huber` |
| n_estimators | 500 |
| learning_rate | 0.04 |
| num_leaves | 31 |
| min_child_samples | 40 |
| subsample / colsample_bytree | 0.90 / 0.90 |
| reg_lambda | 1.0 |
| **alpha (huber 전환 기준)** | **50.0** (원본 기본값 0.9) |
| random_state | 42 |
| early stopping | 30 rounds |

### 이상 점수와 분류(이상 판정)

```
1) 잔차       r = | 실제 RealPower − LightGBM 예측값 |
2) 기준       학습 데이터 r의 PageNo별 중앙값 m(PageNo)과
              scale s(PageNo) = max(1.4826 × MAD, 0.25 × 표준편차, 1.0)
3) 이상 점수  score = | r − m(PageNo) | / s(PageNo)
4) 임계값     정상 보정 데이터 score의 99.9 백분위 = 3.6564
5) 판정       score > 3.6564 → 이상(1), 아니면 정상(0)
```

### 대시보드용 파생 값

| 값 | 계산 방법 |
|---|---|
| `risk_ratio` / `risk_level` | score ÷ 임계값. 0.8 미만이면 정상, 0.8~1.0이면 관찰, 1.0 이상이면 이상 |
| `page_median_power` | 학습 데이터의 PageNo별 RealPower 중앙값 (참고 기준선) |
| `inspection_reason` (점검 사유) | 출력 0W → 미출력 (레이저 발진·게이트 점검) · 출력이 다른 SetPower 그룹 수준 → 설정-출력 불일치 · 출력 < PageNo 중앙값 → 출력 저하 (광학계·렌즈·레이저 소스) · 출력 > PageNo 중앙값 → 출력 과다 (출력 제어부·설정값) · 학습에 없는 SetPower면 끝에 "+ 출력 설정값(레시피) 확인" 추가 <br> ※ 점검 사유는 가이드북에 없는 내용으로, 레이저 용접 설비의 일반 지식을 바탕으로 프로젝트에서 임의로 정한 추정 규칙이다. |
| `priority` (점검 우선순위) | 긴급: 39행 이상 연속 또는 0W 포함 · 높음(동일 포인트 반복): 같은 PageNo에서 단발성 이상이 3회 이상 · 높음: 3행 이상 연속 · 보통: 그 외 |
| `judgement` (모듈 판정) | 이상 포인트 0개 → 정상 · 1~2개 → 재검사 권고 · 3개 이상 또는 0W 포함 → 불량 의심 |

---

## 6. 결과

### 회귀 성능 (RealPower 예측)

| 구간 | MAE (W) | RMSE (W) | R² |
|---|---|---|---|
| 01_OK | 5.58 | 6.95 | 0.9998 |
| 02_OK | 6.08 | 7.71 | 0.9998 |
| 03_NG 정상 행 | 6.51 | 8.11 | 0.9997 |
| 04_NG 정상 행 | 32.04 | 165.61 | 0.8882 |
| **전체 테스트 정상 행** | **6.38** | 22.11 | 0.9979 |

가이드북의 정확도 기준은 20W(2000W의 1%), 허용 한도는 60~80W다. 정상 행 MAE 6.4W는 두 기준 모두 안에 들어온다. 04_NG 정상 행의 오차가 큰 것은 설정과 출력이 한 행씩 어긋난 2행 때문이다.

### 이상 탐지 – 전체 테스트 (5,226행)

| Accuracy | Precision | Recall | F1 | TN | FP | FN | TP | 이벤트 탐지 | 정상 오경보(이벤트) | Macro F1(03·04) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.9975 | 0.9574 | **1.0000** | 0.9782 | 4921 | 13 | 0 | 292 | **20 / 20** | 5 | 0.9773 |

### 파일별

| 파일 | Precision | Recall | F1 | TN / FP / FN / TP | 이벤트 탐지 | 정상 오경보(이벤트) |
|---|---|---|---|---|---|---|
| 01_OK | – | – | – | 1950 / 0 / 0 / 0 | – | 0 |
| 02_OK | – | – | – | 1871 / 1 / 0 / 0 | – | 1 |
| 03_NG | 0.9500 | 1.0000 | 0.9744 | 1033 / 1 / 0 / 19 | 19 / 19 | 1 |
| 04_NG | 0.9613 | 1.0000 | 0.9803 | 67 / 11 / 0 / 273 | 1 / 1 | 3 |

**오경보 13건**

| 파일 | 건수 | 내용 |
|---|---|---|
| 04_NG | 9 | 첫 모듈 PageNo 2~11. 출력이 평소보다 높은 수준(1월과 비슷)이라 경계값 근처에서 걸림 (점수 3.7~6.0) |
| 04_NG | 2 | 설정은 83%인데 출력 693W, 설정은 38%인데 출력 1717W. 설정과 출력이 한 행씩 어긋난 경우로, 라벨은 정상이지만 실제로 확인이 필요한 불일치 (점수 128, 296) |
| 02_OK | 1 | PageNo 19 (38% → 83% 전환 지점). 예측이 1616W로 낮게 나옴 (점수 8.6) |
| 03_NG | 1 | PageNo 16, 682W. 점수 3.68로 임계값 바로 위 |

### 변수 중요도 (gain 순)

Speed > Length > SetPower > PreviousPower > TimeGap > GateOnTime > PageNo > PageCos > PageSin

원본에서는 6개 변수의 중요도가 0이었지만, 수정 후에는 9개 변수가 모두 쓰인다.

### 원본(alpha 기본값)과 비교

| 모델 | 정상 행 MAE | F1 | FP | 이벤트 탐지 | 정상 오경보(이벤트) |
|---|---|---|---|---|---|
| 원본 (alpha 0.9) | 453.6W | 0.9865 | 8 | 20/20 | 7 |
| **현재 (alpha 50)** | **6.4W** | 0.9782 | 13 | 20/20 | **5** |

- 원본은 예측값이 거의 두 값(1296W, 1332W)뿐이었다. 그래서 이상 점수가 사실상 PageNo별 robust Z와 같았고, RobustZ와 결과가 똑같았다.
- 수정 후에는 실제 출력을 예측하므로 "실제 vs 예측" 그래프와 회귀 지표를 대시보드에서 의미 있게 쓸 수 있다.
- 오경보 행 수(FP)는 5건 늘었지만, 오경보 이벤트 수는 7건에서 5건으로 줄었다.

---

## 7. 핵심 결론 · 한계 · 개선 제안

**핵심 결론**

- LightGBM 단일 모델로 단발성 이상(03_NG) 19개와 연속 이상(04_NG) 1개, 총 **20개 이벤트를 모두 탐지**했다 (Recall 1.0).
- 정상 행 예측 오차는 6.4W로, 가이드북 정확도 기준 20W 안에 들어온다.
- 임계값을 정상 보정 데이터로만 정했기 때문에 테스트 라벨이 새어 들어가지 않은 결과다.
- 학습 기간 동안 출력이 월별로 떨어졌다: 82%·83% 조건에서 1월 대비 7월에 약 -21~-22W. 대시보드에서 레이저 소스·광학계 점검 근거로 보여준다.

**한계**

1. 정상 파일(02_OK)에서도 오경보가 1건 발생했다. 레시피가 바뀌는 지점(PageNo 19)에서 예측이 흔들린 것이다.
2. `PreviousPower`(직전 출력)를 Feature로 쓰기 때문에, 이상이 길게 이어지면 예측이 이상값을 따라갈 수 있다. 04_NG는 탐지했지만, 더 긴 이상 구간에서는 확인이 필요하다.
3. 이상 라벨이 있는 파일이 2개뿐이라 임계값(99.9 백분위)이 일반화된다고 보장하기 어렵다.
4. AASX 설비 정보(명판·식별)는 원본 파일 값이 비어 있어('-') 모두 '미등록'으로 표시된다.

**개선 제안**

1. `PreviousPower`를 뺀 모델과 비교해서 연속 이상에 얼마나 강한지 검증한다.
2. 레시피 전환 지점(PageNo 13, 19, 31)의 오경보를 줄이도록 전환 여부 Feature를 추가하거나, 전환 지점만 따로 기준을 둔다.
3. 최근 N개 모듈 기준으로 기준값을 갱신하는 이동 기준선을 적용해 출력 드리프트에 대응한다.
4. `inspection_log.csv`에 쌓인 '이상없음(오경보)' 기록으로 임계값을 다시 보정한다.

---

## 8. 실행 방법 및 산출물

```powershell
cd "battery_pdm_model\src"
python lightgbm_model.py
```

- numpy, pandas, scikit-learn, lightgbm이 없으면 실행 중인 파이썬에 자동으로 설치한다. torch나 pytorch-forecasting은 필요 없다.
- 경로는 `Path(__file__)` 기준 상대 경로라서 프로젝트 폴더 이름(예: `battery_pdm_model`)이 바뀌어도 실행된다. 데이터 폴더 이름(`Dataset_전자부품(배터리팩) 예지보전 AI 데이터셋`)은 바꾸지 않는다.
- 실행 시간은 약 10초 이내다.

### 대시보드 연동 산출물 (`outputs/lightgbm/`)

| # | 파일 | 단위 / 행 수 | 주요 컬럼 | 쓰이는 화면 (요구사항 ID) |
|---|---|---|---|---|
| 1 | `data_quality_summary.csv` | 파일 × 검증 항목 / 65 | file, role, check_item, result(정상/경고/오류), count, description | 데이터 업로드·검증 (DAT-002, 003) |
| 2 | `predictions.csv` | 행 / 5,226 | file, row_index, WorkingTime, cycle_id, PageNo, Speed, Length, SetPower, GateOnTime, RealPower, predicted_power, page_median_power, abs_residual, anomaly_score, threshold, risk_ratio, risk_level, anomaly_pred, unseen_setpower, label, inspection_reason | 필터, KPI, 실제 vs 예측 그래프, 이상 점수 그래프, 히트맵, 조회 테이블, 상세 조회 (DAT-004, DSH-001·002, EDA-003, MDL-001~003, ANM-004, INS-005) |
| 3 | `anomaly_segments.csv` | 이상 구간 / 25 | segment_id, file, start_time, end_time, start_row, end_row, anomaly_rows, cycle_ids, page_nos, set_powers, mean_real_power, mean_predicted_power, mean_page_median_power, max_anomaly_score, max_risk_ratio, inspection_reason, contains_zero_power, true_anomaly_rows, same_page_repeat, priority | 점검 목록, 우선순위, 점검 사유 (INS-001, 003, 004) |
| 4 | `module_judgement.csv` | 모듈(39행) / 134 | file, cycle_id, start_time, end_time, points, anomaly_points, max_anomaly_score, zero_power_points, true_anomaly_points, anomaly_page_nos, judgement(정상/재검사 권고/불량 의심) | 모듈 판정, KPI (INS-002) |
| 5 | `metrics_by_file.csv` | 파일 + 전체 / 5 | accuracy, precision, recall, f1, tn, fp, fn, tp, fpr, event_total, event_detected, missed_events, normal_false_alarm_events, 탐지 지연, macro_f1_03_04, threshold, inference_ms_per_1000_rows | 성능 지표, 혼동행렬, 이벤트 지표 (ANM-002, 003) |
| 6 | `regression_metrics.csv` | 파일 × (정상 행/전체 행) / 10 | MAE, RMSE, R2, accuracy_standard_w(20), tolerance_limit_w(60), retrain_warning | 회귀 품질 지표 (MDL-004) |
| 7 | `threshold_sweep.csv` | 보정 분위수 / 10 | calibration_quantile(99.0~99.99), threshold, precision, recall, f1, fp, event_detected, normal_false_alarm_events, is_default | 임계값 조정 슬라이더 (ANM-001) |
| 8 | `model_baseline_by_pageno.csv` | PageNo / 39 | page_median_power, residual_median, residual_scale, threshold, calibration_quantile, train_period | 새 CSV 탐지 기준값, 재학습 (MDL-001, 006) |
| 9 | `eda_condition_summary.csv` | 데이터 × 그룹 / 243 | dataset, group_type(SetPower/Recipe/PageNo), group_value, count, mean, std, min, median, max, anomaly_count, label_anomaly_count | 조건별 분포, 공정 조건별 요약 (EDA-001, DSH-004) |
| 10 | `eda_monthly_trend.csv` | 월 × SetPower / 21 | month, SetPower, count, mean, std, change_from_first_w, change_from_first_pct, drift_status | 월별 출력 추이, KPI 드리프트 (EDA-002) |
| 11 | `eda_correlation.csv` | 변수 × 변수 / 6 | variable + 6개 변수 상관계수 | 상관관계 히트맵 (EDA-004) |
| 12 | `equipment_info.csv` | 항목 / 64 | category, item, value('-'이면 미등록), source | 설비 정보 패널 (DSH-003) |
| 13 | `inspection_log.csv` | 점검 기록 / 빈 템플릿 | segment_id, file, status, inspector, memo, action_time, updated_at | 점검 이력 (INS-006). **이미 있으면 덮어쓰지 않음** |

**그 외 파일**

| 파일 | 내용 |
|---|---|
| `lightgbm_model.txt` | 학습된 LightGBM 모델 (재학습 없이 예측할 때 사용) |
| `lightgbm_results.json` | 분할 정보, 임계값, best_iteration, 학습 시간, 추론 속도, 전체 결과, 산출물 목록 |
| `lightgbm_feature_importance.csv` | 변수 중요도 (gain / split) |
