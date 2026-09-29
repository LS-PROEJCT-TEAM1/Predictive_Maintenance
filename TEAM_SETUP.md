# BatteryFlow AI 팀원 실행 안내

## GitHub clone 후 실행 (Windows)

1. Python **3.12 (64비트)**와 Git을 설치합니다.
2. 아래 명령으로 **팀원 테스트 브랜치**를 clone하고 해당 폴더를 엽니다.
3. **setup_local.cmd**를 실행합니다. 최초 설치에는 인터넷이 필요합니다.
4. 전달받은 **.env**와 **Firebase 관리자 SDK JSON 한 개**를 `run_local.py`와 같은 루트 폴더에 넣습니다.
5. **start_local.cmd**를 실행하고 http://127.0.0.1:8070 을 엽니다. 등록된 직원 계정으로 로그인합니다.
6. 파일이 없거나 설정이 부족하면 누락 항목을 실행 창에 알리고 **로컬 체험**으로 시작합니다. 체험에서는 실제 저장과 Gemini가 비활성화됩니다.

```powershell
git clone --depth 1 --branch codex/team-local-preview https://github.com/LS-PROEJCT-TEAM1/Predictive_Maintenance.git BatteryFlow-team
cd BatteryFlow-team
.\setup_local.cmd
.\start_local.cmd
```

현재 테스트 배포는 `codex/team-local-preview` 브랜치입니다.
대시보드 최소 실행 묶음은 약 9.55MiB입니다.
브랜치에는 이 실행 묶음과 별도로 기존 연구 자료 및 발주량 재학습 코드·결과도 포함합니다.
`--depth 1`은 과거 커밋 이력 다운로드를 줄이며 현재 브랜치의 연구 자료는 포함합니다.

이미 8070 포트가 사용 중이면 `start_local.cmd --port 8071`로 실행하세요.
종료는 실행 창에서 Ctrl+C입니다. `.venv`는 팀원 PC에서 생성하며 Git에 넣지 않습니다.
GitHub 용량과 별개로 Python 패키지 설치에는 추가 디스크 공간과 시간이 필요합니다.

체험 범위: 네 화면, 공유 필터, 모델 검증, 표 검색·정렬, 상세 보기,
CSV 내보내기, 재학습한 LSTM과 7일 이동평균을 이용한 발주 CSV 추론.
**Firebase 로그인, 업무 기록 저장·조회, Gemini Copilot은 체험 모드에서 비활성화**됩니다.
가짜 저장 성공이나 가짜 AI 답변은 표시하지 않습니다. 공식 시드는 변경하지 않습니다.
체험 서버는 127.0.0.1에서만 실행하고 외부에 배포하지 마세요.

### 현재 작업 디렉터리의 기능과 입력 방식

| 영역 | 구현된 기능 | 입력 방식 |
|---|---|---|
| 통합 현황 | 세 트랙 요약, 검토 목록에서 상세 업무로 이동 | 연결 모드는 Firestore, 체험 모드는 로컬 시드 |
| 공급망 예측 | D+3 예측·계획 비교, 부품 검토·기록, CSV 재예측 | 공식 분석 + 업로드 추론 |
| 예지보전 | 고정 시점 3D 지점 검사, 사이클·이벤트 확인, CSV 검사·분석 | 공식 용접 시험 + 업로드 분석 |
| 품질 보증 | 고정 시점 3D 셀 검사, 원본·보정 비교, 판정·이력, CSV 검사·분석 | 공식 품질 시험 + 업로드 분석 |
| Firebase·Copilot | 직원 인증, 공유 업무 기록, 본인 전용 대화, Gemini·RAG·FAISS | 실제 연결 설정 필요; 체험 모드에서는 비활성화 |

발표용 검증·결론 자료는 운영 화면에서 분리하여 `archives/ui-20260928-before-workflow-3d`에 보관했습니다. 실제 설비의 실시간 수집·제어는 포함하지 않습니다.

공식 버전은 `2026-09-27.v4`, 275문서이며 Firestore 반영과 일치 검증이 완료되었습니다. 연결 모드 분석은 Firestore 캐시를 읽고, 품질 원본과 추론 모델은 동일 버전의 로컬 runtime을 사용합니다. 최신 코드의 GitHub 게시 여부는 별도로 확인해야 합니다. [현재 연결 검증 결과](CONNECTED_WORKFLOW_VERIFICATION.md).

## 실제 Firebase·Gemini 연결

`start_local.cmd`는 기본 `auto` 모드입니다. 팀원이 별도로 전달받을 파일은 루트 `.env`와 `*firebase-adminsdk*.json` 한 개입니다. SDK 파일명은 그대로 두세요. SDK가 여러 개면 `.env`의 `FIREBASE_SERVICE_ACCOUNT_FILE`에 사용할 파일명을 지정합니다. 경로는 프로젝트 루트 기준입니다.

- `.env`: Firebase 프로젝트 ID, Web API 키, Gemini 키·모델, 실행 모드를 담습니다. `.env.example`은 빈 예시이며 실제 키가 없습니다.
- 설정과 SDK가 갖춰지면 실제 Firebase 로그인·Firestore·Gemini를 사용합니다. 등록 계정에는 `manufacturingRole=employee` 또는 `admin`이 필요합니다. 파일 배포나 실행은 계정을 만들거나 권한을 자동 부여하지 않습니다.
- SDK 형식 오류, 프로젝트 불일치, 중복 SDK는 오류로 중단합니다. 연결 후 잘못된 키·권한·할당량·통신 장애도 오류로 표시하며 데모로 바꾸지 않습니다.
- `start_local.cmd --connected`는 누락 시에도 중단합니다. `start_demo.cmd`는 설정과 무관하게 체험을 강제합니다. 모드는 시작할 때만 선택하므로 파일을 넣거나 수정한 뒤 재시작하세요.
- 루트 `.env`가 없는 기존 PC는 `.local/settings.json` 방식도 지원합니다. 루트 `.env` 또는 SDK가 있으면 기존 외부 설정을 섞지 않습니다. 운영체제 환경변수는 `.env`보다 우선합니다.
- 최초 1회 `setup_local.cmd`로 의존성을 설치해야 합니다. 첫 Copilot 사용 때 임베딩 모델 다운로드와 로컬 검색 인덱스 생성에 시간이 걸릴 수 있습니다.
- 모든 팀원은 같은 Firebase 프로젝트와 Gemini 무료 할당량을 공유합니다. 업무 기록은 공유되고, 앱의 대화 조회는 로그인한 작성자 기준입니다. 관리자 SDK 보유자는 앱 밖에서 서비스 계정 권한으로 접근할 수 있으므로 신뢰하는 개발 팀원에게만 비공개로 전달하세요.
- `.env`와 SDK는 Git 및 실행 묶음에서 제외됩니다. 이름을 `.env.txt`로 저장하지 마세요. JSON 키를 브라우저 정적 파일 폴더에 넣지 마세요.

이미 clone한 팀원은 현재 테스트 브랜치에서 `git pull --ff-only` 후 파일 두 개를 루트에 넣고 `start_local.cmd`를 실행하면 됩니다. 다른 브랜치에 있다면 `codex/team-local-preview`로 먼저 전환하세요.

## 실행 데이터

- `firestore/seed`: 공식 로컬 v4 275문서. 파일 크기와 해시는 manifest에서 확인합니다.
- `runtime/quality`: 잠금 시험 5개 CSV의 **무손실 gzip**, 품질 셀·진행률·원본 표 조회용.
- `runtime/demand`: 재학습 LSTM·XGBoost·메타데이터·CSV 템플릿. 학습 과정은 포함하지 않습니다.
- `runtime/validation`: 화면에 필요한 평가 보조표 8개(예지보전 운영 조합·민감도·시간 검증·역방향 평가 포함).
- `runtime/manifest.json`: 원본 경로·원본 SHA-256·배포 파일 SHA-256·시드 버전.
- `runtime` 자료 합계: 약 2.78MiB. 품질 PCA와 예지보전 기본 조합의 고정 실행 자료 및 예제를 포함합니다. 원본 파일을 수정하거나 다운샘플링하지 않았습니다.

시작 시 공식 시드와 실행 자료의 해시·버전을 검사합니다.
원본 연구 폴더의 가상환경이나 학습 데이터 전체 없이 실행됩니다.
공식 시드를 갱신한 관리자는 원본 자료가 있는 PC에서
`.venv\Scripts\python.exe scripts/package_runtime.py`로 실행 자료도 갱신합니다.

## GitHub에 포함할 파일 확인

`.venv\Scripts\python.exe scripts/team_bundle.py --list`는 실행에 필요한 파일 목록과 총 크기를 출력합니다.
`--copy-to .local/team-clone-check`는 그 파일만 새 폴더에 복사하여 누락 검증을 돕습니다.
기존 폴더는 덮어쓰지 않습니다. 복사본에는 개인 설정, 키, 환경 파일, 브라우저 임시 자료가 없습니다.
관리자는 업로드 브랜치에 이 목록의 파일이 모두 포함됐는지 확인해야 합니다.
원래 저장소의 연구 자료·커밋 이력은 clone 크기에 추가되며, 위 용량은 실행용 파일 기준입니다.

## 검증

```powershell
.\.venv\Scripts\python.exe run_local.py --demo --check
.\.venv\Scripts\python.exe -m unittest discover -s backend/tests -v
```

Firestore 저장 테스트는 대역을 사용하며 실제 클라우드 저장 검증을 대체하지 않습니다.
Gemini 품질 평가와 원격 Firestore 시드 배포는 이번 팀원 실행 준비 범위에 포함하지 않습니다.
