# D+3 발주량 예측

117개 부품의 일별 발주 로그를 이용해 기준일로부터 3일 뒤 실제 발주량을 예측하는 프로젝트다. 현재 운영안은 CatBoost를 주 ML 모델로 사용하고, 신규 부품 또는 입력 부족 상황에서는 3일 이동평균을 보조 모델로 사용한다.

## 현재 모델 결론

- 전체 최종 holdout 성능 1위: 3일 이동평균
- 학습형 모델 성능 1위: CatBoost
- 공통 비교 모델: XGBoost, LightGBM, CatBoost, LSTM, 3일 이동평균
- 보조 기준: 원본 D+3 계획량
- 예측 입력: 동일 부품의 연속 3일 actual_d, plan_d3, plan_d4, plan_d5
- 예측 출력: 마지막 입력일 기준 D+3 실제 발주 수량

## 설치

Windows PowerShell 기준이다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 실행 순서

기존 최종 모델을 그대로 사용하는 경우 3번부터 실행할 수 있다.

```powershell
# 1. 단일 holdout 최종 학습
.\.venv\Scripts\python.exe .\src\train_evaluate_final_models.py

# 2. 최종 5모델 expanding walk-forward 검증
.\.venv\Scripts\python.exe .\src\evaluate_walk_forward_final_models.py

# 3. 부품별 분석, 그림, Dashboard/Firestore 파일 생성
.\.venv\Scripts\python.exe .\src\build_dashboard_assets.py

# 4. Dash 실행
.\.venv\Scripts\python.exe .\app.py

# 5. 추론·Firestore 파일 검사
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Dash 주소는 `http://127.0.0.1:8050`이다.

## CSV 추론

입력 파일은 동일 부품의 연속 3일 자료이며 다음 열을 가져야 한다.

`part_number,date,actual_d,plan_d3,plan_d4,plan_d5`

```powershell
.\.venv\Scripts\python.exe .\src\inference.py .\outputs\dashboard_data\inference_input_template.csv
```

학습에 없던 부품은 CatBoost 결과 대신 3일 이동평균을 반환한다. 계획량과 실제량의 0은 정상값으로 유지한다.

## 주요 산출물

- `outputs/final_model_evaluation`: 최종 holdout 평가와 시각화
- `outputs/final_walk_forward`: 5구간 반복검증
- `outputs/dashboard_data/csv`: Dash가 읽는 평면 파일
- `outputs/dashboard_data/firestore`: Firestore 컬렉션별 JSONL
- `models/final_v3`: 저장된 최종 모델

Firestore에는 자동 업로드하지 않는다. 업로드 전에 `outputs/dashboard_data/firestore/manifest.json`과 `docs/FIRESTORE_SCHEMA.md`를 확인한다.

## 데이터 제한

원본 수집 기간은 약 50일이다. 따라서 연간 계절성이나 장기 수요 변화는 검증할 수 없다. 재고량, 단가, 조달 리드타임 자료도 없으므로 현재 예측은 발주 판단을 보조하지만 재고비용 최적화 결과는 아니다.

자세한 오류 해결 방법은 `docs/EXECUTION_AND_ERROR_GUIDE.md`를 참고한다.
