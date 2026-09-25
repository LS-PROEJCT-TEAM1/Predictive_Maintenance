# 배터리팩 레이저 용접 예지보전 - 트랙 B

KAMP 전자부품(배터리팩) 예지보전 데이터셋으로 지도 분류와 정상 패턴 기반 비지도 이상탐지를 같은 잠금 시험 세트에서 비교한다.

## 최종 파이프라인

비교 모델:

1. Logistic Regression 현재 이상 지도 분류
2. Random Forest 현재 이상 지도 분류
3. 현재 RealPower 제외 Logistic Regression 진단 모델
4. PageNo별 Robust Z-score 비지도 탐지
5. Isolation Forest 정상 패턴 비지도 탐지

지도 개발에는 `01_OK + 03_NG`, 독립 잠금 시험에는 파일 전체인 `02_OK + 04_NG`를 사용한다. 개발 내부도 39행 용접 사이클을 유지해 시간순으로 나눈다. 지도모델의 임계값과 최종 모델 선택은 validation에서만 수행하고 잠금 시험 파일은 최종 보고에만 사용한다.

## 환경 구성

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
```

`py` 명령이 없으면 설치된 Python 3.12 실행파일의 절대경로로 첫 명령을 실행한다.

## 학습·검증 실행

```powershell
& '.\.venv\Scripts\python.exe' '.\src\track_b_final_v2.py'
```

주요 결과는 `outputs/track_b_final_v2`에 생성된다.

- `final_validation_report.md`: 최종 평가 보고서
- `data_quality_report.md`: 파일별 품질과 처리 전후 기록
- `metrics.csv`: validation/test 지표와 혼동행렬 값
- `classification_reports.json`: 모델별 classification report
- `predictions.csv`: 행별 점수·임계값·예측
- `split_manifest.csv`: 개발 용접 사이클별 train/validation 배정
- `reverse_file_stress_metrics.csv`: 반대 파일 방향 일반화 민감도 분석
- `feature_importance.csv`: 지도모델 설명 결과
- `models/*.joblib`: 재사용 가능한 모델
- `run_manifest.json`: 데이터·코드 해시, 모델 버전, 최종 모델
- `plots`: 혼동행렬, 특징 중요도, 이상점수·임계값 시각화

## Dash 실행

학습·검증 실행 후:

```powershell
& '.\.venv\Scripts\python.exe' '.\app.py'
```

브라우저에서 `http://127.0.0.1:8050`을 연다. 지도학습과 비지도학습 모델을 별도로 선택할 수 있으며 다음 화면을 제공한다.

- `설비 현황`: 종합 상태, 이상 위치, 이벤트 수, 모델 합의, 점검 우선순위
- `용접 위치 지도`: cycle × PageNo 1~39 히트맵과 위치별 이상 집중도
- `이벤트·점검`: 고립형·연속형 이상 구분, 이벤트 상세, 현장 확인 항목, CSV 다운로드
- `모델 검증실`: 지도·비지도 모델을 분리한 지표와 혼동행렬
- `데이터 품질`: 파일별 품질 검사, 모델·데이터 버전, 적용 한계

현재 화면은 네 시험 파일을 운영 상황처럼 재생하는 모드다. 실제 설비 연결 시에는 같은 화면 구조에 실시간 수집 데이터와 작업자 조치 이력을 연결한다.
원본 KAMP 시험 파일이 없는 배포 환경에서는 `firestore/seed/measurements.jsonl`을 자동으로 읽어 같은 재생 화면을 구성한다.

## Firestore 데이터

Dash 운영 재생 데이터를 Firestore 문서 단위 JSONL로 분리한 파일은 `firestore/seed`에 있다. 측정값에는 지도·비지도 전체 모델의 점수와 판정이 포함되고, 이벤트에는 기본 모델 조합의 점검 권고가 포함된다.

```powershell
& '.\.venv\Scripts\python.exe' '.\scripts\build_firestore_seed.py'
& '.\.venv\Scripts\python.exe' '.\scripts\import_firestore_seed.py'
```

첫 명령은 시드를 다시 만들고 두 번째 명령은 실제 접속 없이 문서 경로·중복·크기 제한을 검증한다. 실제 Firebase 적재 방법과 컬렉션 구조는 `firestore/README.md`를 참고한다. 서비스 계정 키는 저장소에 포함하지 않는다.

## 기존 가이드북 재현 실험

기존 N-HiTS와 2구간 Z-score 재현 결과는 `outputs/guidebook_reproduction`, `outputs/fair_four_model_comparison`, `outputs/final_three_model_comparison`에 보존했다. 가이드북 N-HiTS는 현재 최종 모델 선정 프로토콜과 분리된 legacy 실험이다.

`requirements-guidebook.txt`는 N-HiTS 재현용 별도 환경이다. 최종 트랙 B 파이프라인과 Dash는 `requirements.txt`만 사용한다.

## 오류 대응

- 출력 파일이 없다는 Dash 오류: 먼저 `src/track_b_final_v2.py`를 실행한다.
- 열 누락 오류: 원본 CSV의 9개 필수 열과 `DATA_DICTIONARY.md`를 확인한다.
- 라벨 길이 오류: NG 원본 CSV와 Label CSV의 행 수가 같은지 확인한다.
- 모델 재학습 확인: `run_manifest.json`의 데이터·파이프라인 SHA-256과 실행 결과를 비교한다.
- 잠금 test 변경 금지: `split_manifest.csv`의 test 그룹을 모델 선택이나 임계값 조정에 사용하지 않는다.

## 문서

- `DATA_SOURCES.md`: 출처·이용조건·원본 파일
- `DATA_DICTIONARY.md`: 변수와 파생 특징
- `MODEL_CARD.md`: 목적, 검증, 한계, 배포 승인 조건

## 저장 모델로 새 파일 예측

```powershell
& '.\.venv\Scripts\python.exe' '.\src\predict_track_b.py' `
  '.\Dataset_전자부품(배터리팩) 예지보전 AI 데이터셋\data\raw_data\test\WeldingTest_01_OK.csv' `
  '.\outputs\track_b_final_v2\inference_example.csv'
```

`--model LogisticCurrent`처럼 모델을 지정할 수 있다. 생략하면 validation에서 선정된 최종 모델을 사용한다.

## 자동 검증

```powershell
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
```
