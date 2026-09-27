# 공식 로컬 시드 v3

2026-09-27 재학습 결과를 취합한 `2026-09-27.v3`, schemaVersion 3, 총 269개 문서입니다. **로컬 생성만 완료했고 Firebase 업로드는 하지 않았습니다.** 원격과의 동일성을 주장하지 않습니다.

| 파일 | 문서 수 | 내용 |
|---|---:|---|
| core.jsonl | 7 | 프로젝트, 통합 요약, 검토 대상 |
| demand.jsonl | 128 | 발주 설정·요약, 부품 117개, 비교 모델 9개 |
| maintenance.jsonl | 99 | 기존 용접 이상 탐지 자료 |
| quality.jsonl | 35 | 기존 품질 보증 자료 |

공식 업로드 입력은 `firestore/seed/manifest.json`과 그 파일에 기재된 JSONL 4개입니다. 발주 연구 폴더의 별도 평면 JSONL을 추가로 올리지 않습니다. `manufacturingAi/manufacturing-ai` 아래에 저장하는 기존 경로를 유지합니다. `supply_*`, `pdm_*`, `battery_*`는 기준 데이터가 아닙니다.

발주 운영 정책은 전체 시간순 CV로 선정한 **7-day Moving Average 기본 / LSTM 보조**입니다. `demandConfig/current`의 `primaryModel`, `auxiliaryModel`, `walkForwardMetrics`, `modelVersion=audited_v2`를 읽습니다. 최근값·3일 평균·7일 평균·원본 계획·XGBoost·LightGBM·CatBoost·LSTM·Related LightGBM을 비교합니다.

`Part 21`, `Part 26`은 같은 부품·시각의 서로 다른 원본 수량을 해소하지 못했으므로 `quarantined=true`, 예측·일별 이력 배열은 비워 둡니다. 원본은 연구 폴더에 보존합니다. 117개 등록 부품이 모두 같은 날짜에 예측되는 것은 아닙니다. 최신일 예측 수는 `forecastPartCount`를 사용합니다.

마지막 7개 목표일은 기존 실험에서 이미 본 기간입니다. `evaluationLabel=retrospective_previously_observed_period`, `newUnseenValidationAvailable=false`이며 `modelMetrics`는 **회고 평가**입니다. 독립 미관측 시험 성능이라고 표현하지 않습니다. 모델·부품별 정책 선택에는 `walkForwardMetrics`를 사용했습니다.

`featureContract`는 일별 최종 로그 이후의 달력 D+3 예측입니다. 예측 대상은 ERP 일별 최종 발주 계획량이고 실측 소비량이 아닙니다. 최근 연속 3일을 필수로, 최근 8일 이력을 권장합니다. 관측되지 않은 날짜는 0으로 채우지 않습니다. 미학습 신규 부품은 최근값으로 대체하고 이를 응답에 표시합니다.

`uncertainty`는 CV 잔차로 계산한 참고 상한입니다. 안전재고·최적 발주량·서비스 수준 보장이 아닙니다. 재고·납기·단가·현장 SOP는 연결되지 않았습니다.

각 파일의 SHA-256, 크기, 문서 수와 생성에 사용한 원본 산출물 SHA-256은 매니페스트에 기록합니다. 이전 공식 버전은 `versions/seed-v1`(259개) 및 `versions/seed-2026-09-26.v2`(266개)에 보존했습니다. 이번 작업에서 인증 계정·업무 기록·개인 Copilot 대화는 변경하지 않았습니다.
