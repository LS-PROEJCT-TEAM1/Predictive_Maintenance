# 공식 시드 v2

버전: `2026-09-26.v2` / schemaVersion 2

## 기준과 보존

- 현재 공식 로컬 데이터: `firestore/seed` (266개)
- 이전 공식 데이터: `firestore/versions/seed-v1` (259개, 기존 해시 보존)
- 공통 7개, 발주량 125개, 예지보전 99개, 품질 35개.
- 원격 Firebase는 아직 이전 상태다. 이번 작업은 로컬 생성·검증만 수행했다.
- `supply_*`, `pdm_*`, `battery_*`는 별도·이전 데이터로 유지한다.
- 모든 문서에 schemaVersion/dataVersion을 기록한다. manifest의 sourceArtifacts는 생성 입력의 경로와 SHA-256을 기록한다.

## 발주량

기존 부품별 시험 MAE 기반 권장 모델을 운영 선택에 사용하지 않는다.
검증에서 선정한 `3-day Moving Average`를 운영 기본, XGBoost를 보조로 명시한다.
부품별 latestAlert와 통합 집계는 동일한 운영 예측을 사용한다.
검토 기준은 `abs(예측-계획) >= max(10, 계획*0.20)`이다.
저장된 최종 시험 성능은 비교 근거로 보존하며 모델 선정 근거로 사용하지 않는다.

등록 부품은 117개이고 예측 이력이 있는 부품은 111개다. 최신 목표일
2021-11-01에 예측이 있는 부품은 109개이므로 해당 날짜의 forecastPartCount는 109다.
최신 합계는 10,745.33개, 계획은 9,160개, 검토 대상은 56개다.
다른 날짜의 마지막 관측을 해당 날짜 합계에 섞지 않는다.

## 예지보전

현재 track_b_final_v2에서 생성한 트랙 시드를 재사용한다.
미확인 이벤트 수는 실제 선택 시험 이벤트 수로 집계한다.
현시점 이상 탐지·과거 시험 재생 데이터이며 미래 고장 시점 예측이 아니다.

## 품질

이전 `output/quality_ml` 대신 `output/models`의 최신 PCA 및 모델 비교 결과를 사용한다.
MTadGAN 잠금 평가도 포함하며, 윈도우 정렬에 따른 평가 행 수 차이를 명시한다.
데이터 품질 문서는 최신 10개 정상 기준 파일과 7개 개발·시험 파일 요약이다.

- primaryModel: PCA (T²·SPE); secondaryModel: Random Forest
- ruleThresholds는 pcaThresholds로 대체한다.
- pcaQ는 최신 PCA의 SPE, pcaT2는 Hotelling_T2, pcaPrediction은 저장된 예측이다.
- 임계값은 학습 전용 탐색 결과의 반올림된 저장값이다. 저장된 예측을 이 반올림 값으로 다시 생성하지 않는다.
- abnormalPointCount, abnormalSegmentCount, anomalySegments는 전체 시점에서 계산한다.
- series는 최대 300개 표시점이며 정밀 재추론·전체 시계열 내보내기의 원본으로 사용하지 않는다.
- 최신 Test07 결과는 이상 145시점, 연속 이상 1구간이다.
- decision은 모든 시험에서 pending이다. actualResult/actualLabel은 평가 정답,
  aiStatus는 예측 상태, decision은 작업자 판정으로 구분한다.
- suspectedCells는 예측된 이상 구간의 평균 절대 셀 z-score로 산정한다.
  정답 라벨을 사용하지 않으므로 기존 평가용 셀 순위와 값이 다를 수 있다.
- cellHeatmap과 temperatureHeatmap은 대표 이상 근거 시점의 정제된 측정값이다.
- 실제 현장 SOP가 없으며 이 데이터는 승인된 현장 절차를 대체하지 않는다.

## 소비 앱과 원격 전환

기존 Dash 앱은 별도 CSV·트랙 시드를 읽으므로 이 갱신만으로 화면 데이터가 바뀌지는 않는다.
예정된 공통 FastAPI가 이 계약을 읽도록 연결해야 한다.
품질의 구형 rulePrediction/ruleThresholds/자동 출하판정에 의존하는 소비자는 마이그레이션해야 한다.
신규 직원 프로필과 처리 이력은 `manufacturingAccess`와 `manufacturingOperations` 같은
별도 루트에 둬 기준 시드 배포와 분리한다. 실제 루트 생성은 백엔드 구현 단계에 수행한다.
기존 전체/루트 교체 배포 도구로 운영 기록을 지우지 않도록 전환 절차를 검증한 뒤 원격에 반영한다.

## 검증

```powershell
& '.\.venv\Scripts\python.exe' -B -m unittest discover -s firestore/tests -v
```

5개 검사: v1·v2 해시/출처/경로/크기, 발주량 합계·검토 정책,
품질 전체 시점과 저장 지표 일치, 정답 라벨 변경에 대한 운영 근거 불변,
통합 요약과 도메인별 문서 일치.
