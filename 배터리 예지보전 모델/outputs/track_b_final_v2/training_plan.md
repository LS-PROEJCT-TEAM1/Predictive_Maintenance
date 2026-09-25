# 트랙 B 최종 학습 계획 v2

## 평가 목표

- 지도 분류 모델 2종 이상과 정상 패턴 기반 비지도 모델 1종 이상을 동일한 독립 시험 세트에서 비교한다.
- 불량 행 Recall과 물리 이벤트 Recall을 우선하고, 정상 오경보율과 오경보 이벤트 수를 함께 제한한다.
- 높은 정확도가 동일 파일 분할이나 현재 라벨의 직접 누수에서 나오지 않도록 파일 역할을 먼저 잠근다.

## 데이터 역할

- 정상 기준 학습/보정: `Training_Data.csv`의 앞 70%/뒤 30% 완전 용접 사이클.
- 지도 개발: `WeldingTest_01_OK`와 `WeldingTest_03_NG`; 파일 내부는 39행 완전 사이클을 유지한 시간순 70%/30% train/validation.
- 잠금 시험: `WeldingTest_02_OK`와 `WeldingTest_04_NG` 전체 파일. 적합, 임계값 선택, 모델 선택에 사용하지 않는다.

## 후보 모델

- 지도: LogisticCurrent, RandomForestCurrent.
- 현재값 의존성 진단: LogisticHistoryOnly. 현재 RealPower/Speed/Length를 제외하여 사전 징후 성능을 확인한다.
- 비지도: RobustPhaseZ, IsolationForestNormal. 정상 이력만 학습하며 임계값은 정상 보정 구간 99.9 분위수로 고정한다.

## 선택과 검증

- 지도 임계값은 validation에서 이벤트 Recall 100%, 정상 FPR 1% 이하를 우선한다.
- 최종 모델은 validation에서 이벤트 누락, FN, 오경보 이벤트, F1, FP 순으로 선택한다.
- 모델 선택 뒤 잠금 시험을 한 번 평가한다. 모든 모델은 동일 행과 동일 라벨을 사용한다.
- 이벤트는 cycle별로 잘게 세지 않고 원본 파일에서 연속된 라벨 구간으로 정의한다.

## 해석 경계

- Current 모델은 현재 시점 이상 탐지기이며 미래 고장 예측기로 과장하지 않는다.
- HistoryOnly는 현재 측정값 없이 선행 탐지가 가능한지 확인하는 진단 모델이다.
- NG04는 하나의 장기 이벤트뿐이므로 이벤트 Recall 100%만으로 일반화를 단정하지 않는다.
