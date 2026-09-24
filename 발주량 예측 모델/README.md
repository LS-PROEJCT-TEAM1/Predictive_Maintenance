# D+3 발주량 예측

117개 부품의 일별 발주 로그를 이용해 기준일로부터 3일 뒤 실제 발주량을 예측하는 프로젝트다. D+3 예측 시점에 맞춰 학습·검증·시험 사이에 엄격한 시간 간격을 둔 재평가 결과, 현재 운영 예측기는 3일 이동평균이고 학습형 모델 후보는 XGBoost다.

## 현재 모델 결론

- 누수 제거 3-fold 반복검증 전체 1위: 3일 이동평균
- 누수 제거 3-fold 반복검증 학습형 모델 1위: XGBoost
- 독립 최종 holdout 1위: CatBoost(모델 선정에는 사용하지 않음)
- 공통 비교 모델: XGBoost, LightGBM, CatBoost, LSTM, 3일 이동평균
- 보조 기준: 원본 D+3 계획량
- 예측 입력: 동일 부품의 연속 3일 actual_d, plan_d3, plan_d4, plan_d5
- 예측 출력: 마지막 입력일 기준 D+3 실제 발주 수량
- 수량 정의: 원본 Total과 시간대별 합계를 별도 평가했으며 두 경우 모두 같은 모델 순위

## 설치

Windows PowerShell 기준이다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 실행 순서

현재의 공식 비교 결과를 다시 만들려면 1번을 실행한다. Dashboard와 CSV 추론은 원본 Total 시나리오의 3일 이동평균 기본 예측과 XGBoost 보조 예측에 연결되어 있다.

```powershell
# 1. 누수 제거 반복검증 + 독립 holdout + 두 수량 정의 재평가
.\.venv\Scripts\python.exe .\src\train_evaluate_purged_models.py

# 2. 재평가 결과 자동 검사
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

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

권장 예측은 반복검증 1위인 3일 이동평균이다. 학습에 포함된 부품은 XGBoost 보조 예측도 함께 반환한다. 계획량과 실제량의 0은 정상값으로 유지한다.

## 주요 산출물

- `outputs/purged_evaluation`: 누수 제거 반복검증, 독립 holdout, 수량 정의별 결과
- `outputs/final_model_evaluation`: 이전 단일 holdout 평가(참고용)
- `outputs/final_walk_forward`: 이전 비-purged 반복검증(참고용, 모델 선정에 사용하지 않음)
- `outputs/dashboard_data/csv`: Dash가 읽는 평면 파일
- `outputs/dashboard_data/firestore`: Firestore 컬렉션별 JSONL
- `models/purged_v1`: 누수 제거 최종 학습 모델
- `models/final_v3`: 이전 모델(기존 dashboard 호환용)

Firestore에는 자동 업로드하지 않는다. 업로드 전에 `outputs/dashboard_data/firestore/manifest.json`과 `docs/FIRESTORE_SCHEMA.md`를 확인한다.

## 데이터 제한

원본 수집 기간은 약 50일이다. 따라서 연간 계절성이나 장기 수요 변화는 검증할 수 없다. 재고량, 단가, 조달 리드타임 자료도 없으므로 현재 예측은 발주 판단을 보조하지만 재고비용 최적화 결과는 아니다.

자세한 오류 해결 방법은 `docs/EXECUTION_AND_ERROR_GUIDE.md`를 참고한다.
