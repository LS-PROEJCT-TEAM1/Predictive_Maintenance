# 최종 5모델 expanding walk-forward 평가

- 5개 시간 구간을 순서대로 검증했으며 미래 데이터를 과거 학습에 사용하지 않았다.
- LSTM과 공정하게 비교하기 위해 각 fold의 학습 구간에 한 번도 없던 부품은 해당 fold의 모든 모델 평가에서 제외했다.
- 평가 대상 모델: XGBoost, LightGBM, CatBoost, LSTM, 3-day Moving Average
- pooled MAE 기준 1위: **3-day Moving Average** (39.5075)
- pooled 테스트 표본: 3,808건

단일 holdout 결과와 함께 보며, 약 50일의 짧은 수집 기간 때문에 장기 계절성 검증으로 해석하지 않는다.
