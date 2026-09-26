# 트랙 B 모델 카드

## 목적

배터리모듈 레이저 용접 공정에서 정상 패턴과 다른 행·용접 사이클을 조기에 탐지한다. 불량 Recall과 이상 이벤트 누락 최소화를 우선하고, 같은 조건에서는 오경보를 최소화한다.

## 최종 평가 프로토콜

- 개발 파일: WeldingTest_01_OK, WeldingTest_03_NG
- 독립 잠금 시험 파일: WeldingTest_02_OK, WeldingTest_04_NG
- 개발 내부 분할 단위: 39행 용접 사이클, 시간순 train/validation
- 지도학습: Logistic Regression, Random Forest, 현재 RealPower 제외 진단 Logistic
- 비지도학습: PageNo별 Robust Z-score, Isolation Forest
- 지도모델 불균형 처리: `class_weight="balanced"` 또는 `balanced_subsample`
- 모델 선택: validation의 이벤트 누락, FN, 오경보 이벤트, F1, FP, 학습시간 순
- 최종 보고: 모델 선택에 사용하지 않은 전체 잠금 시험 파일
- 모든 모델은 동일 test 행·라벨로 평가

세부 수치와 혼동행렬은 `outputs/track_b_final_v2/final_validation_report.md`, `metrics.csv`, `classification_reports.json`에 있다.

## 검증 결과 요약

- 최종 선택: RobustPhaseZ
- 독립 파일 시험: Precision 0.9715, Recall 1.0000, F1 0.9856, FN 0, FP 8
- 지도 LogisticCurrent: Precision 0.9927, Recall 1.0000, F1 0.9964
- 반대 파일 방향 스트레스 테스트에서 지도 모델은 NG03 19개 이벤트를 놓쳤고 RobustPhaseZ는 19개를 모두 탐지했다. 이 고장 유형 강건성 때문에 validation 동률에서 단순하고 계산비용이 낮은 RobustPhaseZ를 선택했다.

## 배포 모델

모델 파일은 `outputs/track_b_final_v2/models`에 저장된다. 현재 선택 모델은 `run_manifest.json`의 `winner_selected_on_validation`에서 확인한다. 입력 데이터와 파이프라인의 SHA-256도 같은 파일에 기록된다.

## 제한사항

- 현재 네 시험 파일은 과거 실험에서 이미 사용됐다. 파일 수준 분리는 이전 평가보다 강하지만 완전히 새로운 외부 데이터는 아니다.
- 잠금 시험의 NG04에는 연속 이상 이벤트가 한 건뿐이므로 이벤트 Recall 1.0이 새로운 고장 유형까지 보장하지 않는다.
- 03 파일의 고립 이상과 04 파일의 연속 이상 외 고장 모드는 포함되지 않는다.
- 현재값 모델은 현시점 이상 감지기다. 미래 고장 예측 성능으로 표현하면 안 된다.
- GateOnTime 단위와 현장 허용범위는 설비 담당자의 확인이 필요하다.

## 현장 적용 전 승인 조건

1. 신규 시점·신규 파일의 OK/NG 외부 검증
2. 점검 비용과 생산중단 위험을 반영한 임계값 승인
3. 데이터 스키마와 센서 단위 확인
4. 오경보·미탐 모니터링 및 재학습 기준 확정
