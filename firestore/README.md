# 통합 Firestore 배포

이 디렉터리는 발주량 예측, 배터리 용접 예지보전, 배터리 품질보증, 통합 현황 화면에 필요한 데이터를 하나의 Firestore 구조로 생성하고 배포한다.

현재 공식 시드는 **`2026-09-27.v3` / schemaVersion 3 / 269개 문서**다.
이전 259개 문서는 `versions/seed-v1`에 보존한다. 원격 Firebase는 이번 갱신에서
변경하지 않았으므로 로컬과의 동기화 여부는 `DEPLOYMENT_STATUS.md`를 확인한다.
변경 내역과 필드 계약은 `SEED_V3.md`에 정리했다.

## 데이터 구조

```text
manufacturingAi/{workspaceId}
  overview/current
  actions/{actionId}
  demandOverview/current
  demandConfig/current
  demandParts/{partId}
  demandModels/{modelId}
  maintenanceOverview/current
  maintenanceConfig/current
  maintenanceRuns/{runId}
    events/{eventId}
    measurementChunks/{chunkId}
  maintenanceModels/{modelId}
  maintenanceDataQuality/{fileId}
  qualityOverview/current
  qualityConfig/current
  qualityTests/{testId}
  qualityModels/{modelId}
  qualityDataQuality/{fileId}
```

원본 데이터는 서로 병합하지 않는다. `overview/current`와 `actions`만 세 트랙의 상태 및 조치 정보를 공통 형식으로 요약한다.

Spark 요금제의 문서 쓰기 수를 줄이기 위해 발주 이력은 부품 문서의 배열로, 용접 측정값은 100개 단위 청크로, 품질 시계열은 시험별 최대 300개 표시점으로 저장한다.

## 1. 시드 생성

```powershell
& 'C:\Users\USER\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  '.\firestore\build_unified_seed.py'
```

## 2. 관리자 인증

서비스 계정 JSON은 저장소 밖에 보관한다. JSON 내용을 채팅이나 Git에 올리지 않는다.

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS = 'C:\secure\firebase-service-account.json'
```

서비스 계정에는 최소 `Cloud Datastore User (roles/datastore.user)` 권한이 필요하다.

## 3. 읽기 전용 사전 점검

다음 명령은 현재 컬렉션과 문서 수, 새 시드 수만 표시하며 변경하지 않는다.

```powershell
& '.\.venv\Scripts\python.exe' '.\firestore\deploy_firestore.py' `
  --project-id 'YOUR_PROJECT_ID' `
  --database-id '(default)'
```

## 4. 통합 데이터만 교체 (권장)

다음 명령은 시드에 포함된 최상위 컬렉션만 교체한다. 현재 시드에서는
`manufacturingAi`만 삭제 후 다시 적재하며, `supply_*`, `pdm_*`, `battery_*` 같은
다른 최상위 컬렉션은 건드리지 않는다.

```powershell
& '.\.venv\Scripts\python.exe' '.\firestore\deploy_firestore.py' `
  --project-id 'YOUR_PROJECT_ID' `
  --database-id '(default)' `
  --apply `
  --replace-seed-roots `
  --confirm-project-id 'YOUR_PROJECT_ID'
```

## 5. 데이터베이스 전체 교체 (비권장)

대상 프로젝트 ID를 두 번 일치시켜야 실행된다.
시드와 무관한 컬렉션이 있으면 기본적으로 중단되며, 정말 전체 삭제할 때만
`--allow-delete-unrelated-collections`를 추가한다.

```powershell
& '.\.venv\Scripts\python.exe' '.\firestore\deploy_firestore.py' `
  --project-id 'YOUR_PROJECT_ID' `
  --database-id '(default)' `
  --apply `
  --replace-all `
  --allow-delete-unrelated-collections `
  --confirm-project-id 'YOUR_PROJECT_ID'
```

관리자 SDK는 Firestore Security Rules를 우회하고 IAM으로 인증한다. 따라서 현재 규칙이 `allow read, write: if false;`여도 관리자 적재는 가능하다. 브라우저가 Firestore를 직접 읽는 구조라면 별도로 인증·보안 규칙을 설계해야 한다.

전체 삭제는 되돌릴 수 없다. 배포 도구는 먼저 대상 프로젝트, 데이터베이스 ID,
현재 문서 수, 시드와 무관한 컬렉션을 출력한다. 프로젝트 ID가 이중 확인되고
관련 없는 컬렉션 삭제까지 명시적으로 허용된 경우에만 전체 삭제한다.
