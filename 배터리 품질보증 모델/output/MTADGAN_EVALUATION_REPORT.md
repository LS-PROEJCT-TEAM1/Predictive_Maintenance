# MTadGAN 실제 학습·검증 결과

## 결론

MTadGAN은 이 데이터셋의 최종 모델로 채택하지 않는다. 동일한 정상 학습 범위와 동일한 잠금 시험에서 PCA(T²·SPE)의 F1은 0.9866, MTadGAN의 F1은 0.4275였다. MTadGAN은 복잡도와 학습 비용이 더 크면서 정상 구간 오탐과 불량 구간 미탐이 모두 많았다.

## 검증 설계

- 정상 학습: `1000~1004_chg/dchg.csv` 10개
- 개발/운영점 선택: `Test05_NG_chg`, `Test09_NG_dchg`
- 잠금 최종시험: `Test03_OK_chg`, `Test04_OK_dchg`, `Test06_NG_chg`, `Test07_NG_dchg`, `Test08_NG_chg`
- MTadGAN 입력: 208개 전압·온도 변수, 학습 데이터에만 적합한 PCA 3성분과 MinMaxScaler, 길이 10 윈도우
- 학습: 충전·방전 모델 분리, 각 30 epoch, batch 64, critic 5회, latent 20, seed 42
- 임계값 후보: 정상 학습 점수의 50/75/80/85/90/95/97.5/99% 분위수
- 선택 기준: 개발 파일을 합친 pooled F1
- 선택 결과: 충전 50% 분위수(2.016779), 방전 50% 분위수(1.700158), padding 100
- 최종시험에서는 선택값을 변경하지 않았다.

## 잠금 최종시험 결과

| 파일 | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Test03_OK_chg | 0 | 505 | 0 | 915 | 0.0000 | 0.0000 | 0.0000 |
| Test04_OK_dchg | 0 | 1,342 | 0 | 162 | 0.0000 | 0.0000 | 0.0000 |
| Test06_NG_chg | 565 | 0 | 288 | 0 | 1.0000 | 0.6624 | 0.7969 |
| Test07_NG_dchg | 226 | 936 | 0 | 3,431 | 0.1945 | 1.0000 | 0.3256 |
| Test08_NG_chg | 910 | 0 | 1,485 | 0 | 1.0000 | 0.3800 | 0.5507 |
| **합계/pooled** | **1,701** | **2,783** | **1,773** | **4,508** | **0.3793** | **0.4896** | **0.4275** |

전체 Accuracy는 0.5768이다. MTadGAN 윈도우 점수는 각 파일의 첫 경계 1시점을 제외하므로 총 평가 수는 10,765이며, PCA 결과의 10,770과 5시점 차이가 있다.

## 검증 과정에서 수정한 문제

기존 MTadGAN 스크립트는 테스트 파일마다 PCA와 MinMaxScaler를 새로 적합하고 있었다. 이는 테스트 분포를 전처리에 사용하는 누수이므로 학습 데이터에서 적합한 전처리기를 저장하고 예측 시 그대로 재사용하도록 수정했다.

또한 가이드북식 `mean + k×std`는 `k`만 고정하고 평균과 표준편차를 각 시험 파일에서 다시 계산한다. 이 결과(F1 0.4503)는 시험 분포에 적응한 참고값으로만 보관하고, 모델 선정에는 정상 학습 점수 후보를 개발 파일에서 선택해 잠근 결과(F1 0.4275)를 사용했다.

## 산출물

- `training_history_mtadgan_chg.csv`, `training_history_mtadgan_dchg.csv`: 30 epoch 손실 이력
- `mtadgan_normal_calibration.csv`: 정상 팩별 점수 통계
- `mtadgan_detection_config.json`: 정상 점수 분위수 후보
- `mtadgan_operating_point_search.csv`: 개발 운영점 탐색표
- `mtadgan_operating_point.json`: 잠근 충전·방전 임계값과 padding
- `metrics_mtadgan_normal_locked_final.csv`: 유효한 잠금 최종 결과
- `metrics_mtadgan_test_adaptive_final.csv`: 시험별 재보정 참고 결과(모델 선정에 사용하지 않음)
