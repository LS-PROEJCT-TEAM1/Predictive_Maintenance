# 실행 및 오류 해결

## 실행 전 확인

1. Python 3.12 가상환경을 사용한다.
2. `requirements.txt`를 설치한다.
3. 원본 `data.xls`가 `Dataset_공급망 최적화 AI 데이터셋/data`에 있는지 확인한다.
4. 기존 모델을 사용할 때는 `models/final_v3`에 CatBoost 모델과 metadata.json이 있는지 확인한다.

## 자주 발생하는 오류

### ModuleNotFoundError

프로젝트 전용 Python이 아닌 다른 Python을 실행한 경우가 많다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Dashboard data is missing

대시보드 파일을 먼저 만든다.

```powershell
.\.venv\Scripts\python.exe .\src\build_dashboard_assets.py
```

### Missing columns 또는 exactly three daily rows

추론 CSV는 정확히 3행이어야 하며 다음 열을 가져야 한다.

`part_number,date,actual_d,plan_d3,plan_d4,plan_d5`

세 날짜는 연속이어야 하고 모든 행의 part_number가 같아야 한다.

### part_not_seen_in_training

학습에 없던 부품이다. 오류로 종료하지 않고 3일 이동평균을 권장 예측으로 사용한다. 해당 부품 데이터가 충분히 쌓이면 모델 재학습 대상에 포함한다.

### CatBoost 모델 로딩 오류

모델 경로 문제를 줄이기 위해 추론 모듈이 임시 영문 경로로 복사해 한 번 더 로딩한다. 계속 실패하면 `models/final_v3/catboost.cbm`이 Git LFS 포인터 파일이 아닌 실제 모델인지 확인한다.

### Walk-forward 실행 시간이 김

5개 fold에서 XGBoost, LightGBM, CatBoost, LSTM을 매번 다시 학습하므로 단일 holdout보다 오래 걸린다. 완료 후 `outputs/final_walk_forward/metrics_pooled.csv`가 생성됐는지 확인한다.

## 결과 검증

- 모든 모델의 최종 holdout N과 Parts 값이 같은지 확인한다.
- Firestore `manifest.json`의 document_count와 각 JSONL 줄 수가 같은지 확인한다.
- JSONL에 NaN 또는 Infinity 문자열이 없는지 확인한다.
- 추론 템플릿으로 실행했을 때 target_date가 마지막 입력 날짜보다 3일 뒤인지 확인한다.
- Dash에서 부품 선택 시 그래프와 부품별 성능표가 함께 바뀌는지 확인한다.

## 데이터 해석 주의

약 50일 자료만 있으므로 장기 계절성이나 연간 안정성을 주장하지 않는다. `alerts`는 계획 대비 예측 차이이며 실제 재고 부족 확률이 아니다.
