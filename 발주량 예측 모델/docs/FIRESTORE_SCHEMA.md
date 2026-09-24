# Firestore 준비 데이터 구조

`outputs/dashboard_data/firestore`의 JSONL 파일은 한 줄이 Firestore 문서 한 개다. 아직 Firestore에는 업로드하지 않는다. 각 문서의 `document_id`를 Firestore 문서 ID로 사용한다.

## 컬렉션

| 컬렉션 | 용도 | 주요 조회 키 |
|---|---|---|
| dashboard_config | 현재 모델, 예측 기간, 화면 설정 | document_id=current |
| parts | 부품 목록, 데이터 범위, 권장 모델 | part_number |
| daily_history | 부품별 일별 실제·계획 수량 | part_number, date |
| forecasts | 최종 테스트 실제값과 모델별 예측 | part_number, target_date |
| model_metrics | 전체 모델 성능 | Model |
| part_metrics | 부품별 모델 성능 | part_number, Model |
| daily_summary | 날짜별 전체 수량과 예측 합계 | target_date |
| alerts | 계획량과 권장 예측의 차이가 큰 최신 부품 | alert_type, target_date |
| data_quality | 데이터 품질 및 전처리 건수 | metric |
| walk_forward_metrics | fold별 반복검증 결과 | Fold, Model |

## 권장 인덱스

- `daily_history`: part_number 오름차순 + date 오름차순
- `forecasts`: part_number 오름차순 + target_date 오름차순
- `part_metrics`: part_number 오름차순 + MAE 오름차순
- `alerts`: alert_type 오름차순 + target_date 내림차순
- `walk_forward_metrics`: Model 오름차순 + Fold 오름차순

## 값 규칙

- 날짜와 시각은 UTC ISO 8601 문자열로 저장한다.
- NaN과 무한대는 JSON `null`로 변환한다.
- Part Number는 숫자로 변환하지 않고 문자열로 유지한다.
- 실제 수량과 계획 수량의 0은 정상값으로 유지한다.
- 예측 수량은 0 이상으로 제한한다.
- `alerts`는 재고 부족 확정 경보가 아니다. 재고량과 조달 리드타임이 없으므로 계획 대비 예측 차이를 보여주는 확인 대상이다.

## 화면과 컬렉션 연결

- 상단 KPI: forecasts, model_metrics
- 부품 선택 목록: parts
- 실제·예측 추이: forecasts 또는 daily_summary
- 부품별 성능표: part_metrics
- 계획 대비 확인 대상: alerts
- 데이터 품질 화면: data_quality
- 모델 안정성 화면: walk_forward_metrics
- 과거 입력 조회: daily_history

## 추론 결과 저장안

실제 Firestore 연결 시 `inference_results` 컬렉션을 추가한다. 문서에는 `part_number`, `origin_date`, `target_date`, `xgboost_prediction`, `moving_average_3d`, `recommended_model`, `recommended_forecast`, `model_version`, `created_at`을 저장한다. 현재 로컬 Dash는 결과를 DB에 기록하지 않는다.
