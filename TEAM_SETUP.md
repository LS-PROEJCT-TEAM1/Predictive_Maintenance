# BatteryFlow AI 팀원 실행 안내

## GitHub clone 후 로컬 체험 (Windows)

1. Python **3.12 (64비트)**와 Git을 설치합니다.
2. 아래 명령으로 **팀원 테스트 브랜치**를 clone하고 해당 폴더를 엽니다.
3. **setup_local.cmd**를 실행합니다. 최초 설치에는 인터넷이 필요합니다.
4. **start_demo.cmd**를 실행하고 http://127.0.0.1:8070 을 엽니다.
5. **체험 시작**을 누릅니다. 이메일·비밀번호·API 키가 필요하지 않습니다.

```powershell
git clone --depth 1 --branch codex/team-local-preview https://github.com/LS-PROEJCT-TEAM1/Predictive_Maintenance.git BatteryFlow-team
cd BatteryFlow-team
.\setup_local.cmd
.\start_demo.cmd
```

현재 테스트 배포는 `codex/team-local-preview` 브랜치입니다.
기존 연구 자료를 포함한 해당 브랜치의 파일 합계는 약 178.66MiB이며,
대시보드 최소 실행 묶음은 약 10.49MiB입니다. 가장 큰 파일은 약 46.37MiB입니다.
`--depth 1`은 과거 커밋 이력 다운로드를 줄이며 현재 브랜치의 연구 자료는 포함합니다.

이미 8070 포트가 사용 중이면 `start_demo.cmd --port 8071`로 실행하세요.
종료는 실행 창에서 Ctrl+C입니다. `.venv`는 팀원 PC에서 생성하며 Git에 넣지 않습니다.
GitHub 용량과 별개로 Python 패키지 설치에는 추가 디스크 공간과 시간이 필요합니다.

체험 범위: 네 화면, 공유 필터, 모델 검증, 표 검색·정렬, 상세 보기,
CSV 내보내기, 기존 이동평균·XGBoost를 이용한 발주 CSV 추론.
**Firebase 로그인, 업무 기록 저장·조회, Gemini Copilot은 체험 모드에서 비활성화**됩니다.
가짜 저장 성공이나 가짜 AI 답변은 표시하지 않습니다. 공식 시드는 변경하지 않습니다.
체험 서버는 127.0.0.1에서만 실행하고 외부에 배포하지 마세요.

## 실제 Firebase·Gemini 연결

`start_local.cmd`는 기존 Firebase 인증 모드입니다. 체험 모드로 자동 전환하지 않습니다.
관리자는 LOCAL_RUN.md에 따라 각 PC의 `.local/settings.json`과 외부 비밀 파일 경로를 설정합니다.
Firebase 관리자 키, Gemini 키, 직원 비밀번호는 Git에 올리지 않습니다.
단순 화면 테스트를 하는 팀원에게 Firebase 관리자 키를 배포할 필요가 없습니다.
실제 연결 기능을 여러 팀원이 함께 사용할 단계에는 관리자만 키를 보유한 공용 백엔드가 적절합니다.

## 실행 데이터

- `firestore/seed`: 공식 v2 266문서, 약 6.46MiB.
- `runtime/quality`: 잠금 시험 5개 CSV의 **무손실 gzip**, 품질 셀·진행률·원본 표 조회용.
- `runtime/demand`: 기존 XGBoost·전처리기·메타데이터. 학습 과정은 포함하지 않습니다.
- `runtime/validation`: 화면에 필요한 평가 보조표 4개.
- `runtime/manifest.json`: 원본 경로·원본 SHA-256·배포 파일 SHA-256·시드 버전.
- `runtime` 자료 합계: 약 3.21MiB. 원본 파일을 수정하거나 다운샘플링하지 않았습니다.

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
