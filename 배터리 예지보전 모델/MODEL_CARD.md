# 트랙 B 모델 카드 — track-b-audited-v3

## 목적

배터리모듈 레이저 용접 공정에서 정상 패턴과 다른 행·용접 사이클을 조기에 탐지한다. 불량 Recall과 이상 이벤트 누락 최소화를 우선하고, 같은 조건에서는 오경보를 최소화한다.

## 최종 평가 프로토콜

- 개발 파일: WeldingTest_01_OK, WeldingTest_03_NG
- 재사용 파일 회고 시험: WeldingTest_02_OK, WeldingTest_04_NG. 신규 외부 시험이 아니다.
- 개발 내부 분할 단위: 39행 용접 사이클, 시간순 train/validation
- 지도학습: Logistic Regression, Random Forest, 현재 RealPower 제외 진단 Logistic
- 비지도학습: PageNo별 Robust Z-score, LightGBM 정상 출력 회귀·잔차 탐지, Isolation Forest
- 정상 기준 정제: 0 출력이 있는 사이클 1개와 사이클 내부 시간 역전 1개 격리. 원본 ID 유지 후 사이클 시작 시각으로 정렬.
- 지도모델 불균형 처리: `class_weight="balanced"` 또는 `balanced_subsample`
- 모델 선택: validation의 이벤트 누락, FN, 오경보 이벤트, F1, FP 순. 동률은 학습 전 계획의 단순성 순서로 결정한다. 학습시간은 선택에 쓰지 않는다.
- 최종 보고: 모델 선택에 사용하지 않은 전체 잠금 시험 파일
- 모든 모델은 동일 test 행·라벨로 평가

세부 수치와 혼동행렬은 `outputs/track_b_final_v2/final_validation_report.md`, `metrics.csv`, `classification_reports.json`에 있다.

## 검증 결과 요약

- 최종 선택: RobustPhaseZ
- 회고 파일 시험: Precision 0.9891, Recall 1.0000, F1 0.9945, FN 0, FP 3
- 지도 LogisticCurrent: Precision 0.9927, Recall 1.0000, F1 0.9964
- 비지도 LightGBMResidual: Precision 0.9579, Recall 1.0000, F1 0.9785, FN 0, FP 12
- RobustPhaseZ는 validation 동률에서 사전 정의 단순성 순서로 선택했다. 반대 파일 방향 스트레스와 추가 민감도 결과는 선택에 사용하지 않았다.
- 기본 운영 조합 LogisticCurrent OR RobustPhaseZ의 사이클 Precision 87.5%, Recall 100%, F1 0.9333, 정상 사이클 FPR 2%. 사이클은 한 행 이상 경보로 정의한다.
- 모든 시험보다 과거인 정상 자료로 적합·보정한 별도 RobustPhaseZ는 01_OK·02_OK 정상 행 전부를 오경보 처리했다. 시간 변화에 대한 일반화 실패이며 현장 적용 검증은 미완료다.

## 입력과 경보 계약

필수 열·유한한 숫자·공정 번호·날짜·완전한 39행 사이클·시간 증가를 검사한다. 오류 입력은 ValueError로 거부하고 정상 예측 파일을 만들지 않는다. 센서 단위와 물리 범위의 현장 승인은 별도다.

경보는 첫 양성 행에서 시작하고 정상 행 또는 120초 초과 수집 공백에서 끝난다. 사이클 경계만으로 분리하지 않으며 미래 행을 이용한 평활화를 하지 않는다. 모델/정책 버전이 바뀌면 업무 확인 기록의 대상 ID도 달라진다.

## 배포 모델

모델 파일은 기존 연동 경로인 `outputs/track_b_final_v2/models`에 저장된다. 실제 버전은 track-b-audited-v3다. 현재 선택 모델, 정책, 입력·코드·계획·모델 SHA-256, 라이브러리 버전은 run_manifest.json에 기록된다. selection.json은 회고 시험 추론 전에 저장된다.

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
