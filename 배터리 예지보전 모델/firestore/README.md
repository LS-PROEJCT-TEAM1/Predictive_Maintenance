# Firestore 데이터 시드

이 디렉터리는 Dash 운영 재생 화면의 데이터를 Firestore 문서 구조로 옮긴 결과다. 원본 KAMP 파일 전체를 복제하지 않고, 화면에 필요한 측정값·모델 점수·이벤트·평가 메타데이터를 저장한다.

## 구조

```text
projects/{projectId}
  modelEvaluations/{model}_{split}
  dataQuality/{sourceFile}
  weldingRuns/{sourceFile}
    measurements/{sourceRow}
    events/{eventId}
```

`seed/*.jsonl`의 각 줄은 다음 형태다.

```json
{"path":"projects/track-b-final-v2/...","data":{"field":"value"}}
```

- `projects.jsonl`: 프로젝트와 기본 모델 정보
- `model_evaluations.jsonl`: validation/독립 시험 모델 평가
- `data_quality.jsonl`: 파일별 데이터 품질 검사
- `welding_runs.jsonl`: 시험 파일 단위 요약
- `measurements.jsonl`: 행별 공정값과 지도·비지도 모델 점수
- `anomaly_events.jsonl`: 기본 지도·비지도 모델 조합의 이벤트와 권장 점검
- `manifest.json`: 문서 수, 파일 크기, SHA-256

## 다시 생성

프로젝트 루트에서 학습 산출물과 로컬 KAMP 데이터가 준비된 상태로 실행한다.

```powershell
& '.\.venv\Scripts\python.exe' '.\scripts\build_firestore_seed.py'
```

## 검증 및 적재

검증은 Firebase 접속이나 자격증명 없이 수행된다.

```powershell
& '.\.venv\Scripts\python.exe' '.\scripts\import_firestore_seed.py'
```

실제 적재는 서비스 계정 JSON을 저장소 밖에 두고 `GOOGLE_APPLICATION_CREDENTIALS`로 지정한 후 실행한다. 같은 경로의 문서는 덮어쓰므로 반복 실행할 수 있다.

```powershell
& '.\.venv\Scripts\python.exe' -m pip install -r '.\requirements-firestore.txt'
$env:GOOGLE_APPLICATION_CREDENTIALS = 'C:\secure\firebase-service-account.json'
& '.\.venv\Scripts\python.exe' '.\scripts\import_firestore_seed.py' --apply --project-id 'YOUR_FIREBASE_PROJECT_ID'
```

관리자 SDK는 Firestore 보안 규칙을 우회하므로, 운영 적재 전 프로젝트 ID와 자격증명 대상을 반드시 확인한다. 서비스 계정 키, `.env`, 실제 Firebase 프로젝트 ID는 Git에 커밋하지 않는다.
