# Predictive Maintenance & Demand Forecasting

KAMP 제조 데이터를 활용한 예지보전 및 부품 발주량 예측 프로젝트 저장소입니다.

## 프로젝트 구성

| 디렉터리 | 내용 |
|---|---|
| `발주량 예측 모델/` | 117개 부품의 D+3 발주량 예측, 5개 모델 비교, Dash 화면, Firestore 준비 데이터 |
| `배터리 예지보전 모델/` | 배터리 모듈 용접 데이터 기반 이상·고장 예지보전 모델 |
| `data/pdm/` | 공용 예지보전 원본 및 전처리 데이터 |
| `src/` | 공용 전처리 코드 |

## 발주량 예측 모델

동일 부품의 연속 3일 실제 발주량과 D+3~D+5 계획량을 이용해 마지막 입력일 기준 D+3 실제 발주량을 예측합니다.

- 비교 방법: XGBoost, LightGBM, CatBoost, LSTM, 3일 이동평균
- 최종 holdout 전체 1위: 3일 이동평균, MAE 30.264
- 최종 holdout 학습형 모델 1위: CatBoost, MAE 34.068
- 5구간 walk-forward 학습형 모델 1위: XGBoost, MAE 42.070
- 운영안: CatBoost 주 모델, 3일 이동평균 fallback
- Dash: 부품별 예측 조회, 모델 성능, 계획 대비 확인 대상, 신규 CSV 추론
- Firestore: 컬렉션별 JSONL과 업로드 매니페스트 준비 완료. 자동 업로드는 하지 않음

자세한 설치 방법과 실행 순서는 [`발주량 예측 모델/README.md`](./발주량%20예측%20모델/README.md)를 확인하세요.

주요 결과:

- [`최종 분석 보고서`](./발주량%20예측%20모델/outputs/final_model_evaluation/FINAL_ANALYSIS_REPORT.md)
- [`Walk-forward 결과`](./발주량%20예측%20모델/outputs/final_walk_forward/metrics_pooled.csv)
- [`Firestore 데이터 구조`](./발주량%20예측%20모델/docs/FIRESTORE_SCHEMA.md)
- [`대시보드 기능 정의`](./발주량%20예측%20모델/docs/DASHBOARD_REQUIREMENTS.md)

## 발주량 예측 Dash 실행

```powershell
cd "발주량 예측 모델"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\app.py
```

브라우저에서 `http://127.0.0.1:8050`으로 접속합니다.

## 데이터 해석 주의

발주량 데이터 수집 기간은 약 50일입니다. 장기 계절성이나 연간 수요 안정성을 검증한 결과로 해석하지 않습니다. 또한 재고량, 단가, 조달 리드타임이 없으므로 현재 예측은 발주 판단 보조용이며 재고비용 최적화 결과가 아닙니다.
