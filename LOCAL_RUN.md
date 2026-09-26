# BatteryFlow AI 운영센터 · 로컬 실행

## 실행

Windows에서 `start_local.cmd`를 더블클릭하고 http://127.0.0.1:8070 을 여세요.
터미널에서 실행하려면 프로젝트 루트에서:

```powershell
.\.venv\Scripts\python.exe run_local.py
```

종료는 실행 창에서 Ctrl+C. 기본 포트는 8070입니다. 충돌 시 `run_local.py --port 8071`로 변경하세요.
FastAPI API 문서: http://127.0.0.1:8070/docs

## 처음 설치하는 환경

Python 3.12를 기준으로 합니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
```

팀원은 **setup_local.cmd → start_demo.cmd**로 외부 연결 없이 분석 화면을 확인할 수 있습니다.
자세한 절차는 [TEAM_SETUP.md](TEAM_SETUP.md)를 참고하세요.
CSV 재예측은 앱과 같은 Python 환경 및 `runtime/demand`의 기존 모델을 이용합니다.
품질 셀 신호는 `runtime/quality`에 포함한 무손실 압축 CSV를 읽습니다.
기존 연구 폴더의 가상환경이나 원본 학습 자료 전체를 설치할 필요가 없습니다.

## 직접 확인할 흐름

1. **통합 현황**: 세 영역의 요약, 우선 검토 목록, 프로젝트 개요, 데이터 품질, 결과 조회.
2. **공급망 예측**: 날짜·부품·모델 선택 → 차트 → 검토 목록 검색 → 행 선택 → 확인 처리.
   CSV로 재예측에서 템플릿을 다운로드하고 업로드하면 이동평균·XGBoost 결과가 나옵니다.
   템플릿은 추론 시험용 예시 수량이며 공식 데이터가 아닙니다.
3. **예지보전**: 시험 파일·지도/비지도 모델 변경 → 출력·점수·위험 분포 → 이벤트 상세와 확인 처리.
4. **품질 보증**: 시험·셀·진행률 변경 → 이상 근거·위치 지도 → 판정·조치에서 메모와 판정 확인.
5. 각 영역 **모델 검증**에서 원 평가 지표·독립 시험·비교 근거를 확인합니다.
6. **CSV 내보내기** → **CSV 파일 다운로드**로 현재 표의 필터와 정렬을 반영한 파일을 받습니다.
   링크는 15분 유효하며 서버 메모리에 최근 20개까지만 임시 보관됩니다.

## 데이터와 저장 범위

- 공식 로컬 데이터: `firestore/seed`, 버전 `2026-09-26.v2`, 266문서. 시작 시 파일 해시 검사.
- FastAPI가 데이터를 제공하고 Dash 콜백이 HTTP로 API를 호출합니다. 같은 8070 포트에서 실행합니다.
- 품질 선택 셀의 신호와 모델 검증 보조표는 `runtime`의 원본 압축본/평가 파일을 읽습니다.
- 예지보전·품질은 기존 시험 결과 탐색입니다. 신규 CSV 추론은 공급망에서 제공합니다.
- 업무 기록은 Firestore에 저장하며 최근 이력을 상단 **업무 기록**에서 조회합니다. 동시 수정 충돌은 거절하고, 같은 요청 재전송은 중복 저장하지 않습니다.
- 공급망 확인 상태는 부품·목표일·모델, 예지보전은 시험·모델 조합·이벤트, 품질은 시험별로 구분됩니다.
- Firebase 이메일 로그인, 공유 이력, Gemini + 로컬 다국어 임베딩 + FAISS를 연결했습니다. 앱 회원가입은 없습니다.
- Copilot은 질문 당시 화면과 허용된 프로젝트 문서만 검색합니다. 출처를 펼쳐 원문을 볼 수 있고, 대화는 작성자 본인만 조회합니다. 앱 관리자도 다른 직원의 대화를 조회할 API가 없습니다.
- **답변으로 검토 초안 만들기**는 저장 전 확인 화면을 엽니다. 단일 부품/품질 시험을 선택해 질문하거나, 예지보전의 이벤트를 선택하세요. 품질 초안의 기본 판정은 재시험 요청입니다. 다른 판정·수정 메모는 품질 판정 화면에서 직접 작성하세요. 확정한 초안의 메모는 직원 공용 업무 기록이 됩니다.
- 재학습, 원격 공식 시드 갱신, 실제 발주·출하·설비 제어는 수행하지 않습니다.
- 기본 실행은 127.0.0.1 전용입니다. 외부 배포용 HTTPS/프록시 설정은 포함하지 않습니다.

## 연결 설정과 직원 관리

이 PC에는 `.local/settings.json`으로 외부 파일 경로를 연결했습니다. 해당 폴더와 비밀 파일은 Git에서 제외합니다.
다른 PC에서는 다음 형식의 설정을 만들고 `MANUFACTURING_SETTINGS`로 경로를 지정하거나 `.local/settings.json`에 저장하세요.

```json
{
  "project_id": "ls-proejct-team1",
  "service_account_file": "C:/private/firebase-admin.json",
  "connections_file": "C:/private/connections.env",
  "firebase_web_key": "Firebase 웹 앱의 공개 API 키",
  "gemini_model": "gemini-2.5-flash"
}
```

`connections.env`에는 `GEMINI_API_KEY=...`를 넣습니다. 키를 프런트엔드나 저장소에 넣지 마세요.
비밀번호는 설정 파일에 저장하지 않습니다. 서비스 계정은 FastAPI만 사용합니다.

1. Firebase Console Authentication에서 이메일/비밀번호 제공자를 켜고 직원 계정을 추가합니다.
2. 관리자가 로컬에서 다음 명령으로 앱 역할을 부여합니다. Custom Claims는 Firebase Console 기본 사용자 화면에서 직접 편집할 수 없으므로 이 도구를 사용합니다.

```powershell
.\.venv\Scripts\python.exe -m backend.bootstrap --email employee@example.com --name "직원 이름" --role employee
```

관리자는 `--role admin`, 앱 권한 회수는 `--inactive`를 사용합니다. 이 도구는 비밀번호를 만들거나 변경하지 않으며 다른 Custom Claims를 보존합니다.
계정 비활성화·삭제·이메일/비밀번호 관리는 Firebase Console에서 수행합니다. 현재 역할을 서버가 다시 확인하며 읽기 접근 캐시는 최대 10초입니다. 업무 저장은 캐시 없이 확인합니다.
사용자가 제공한 두 계정에는 역할을 부여했습니다. 초기에 별도 `manufacturingEmployees` 프로필 2개도 생성했으나 최종 인증 기준은 **Firebase Auth의 manufacturingRole**이며 해당 프로필은 인증에 사용하지 않습니다.

세션은 8시간의 HttpOnly 쿠키입니다. 변경 API는 Origin과 CSRF 토큰을 검사하고 Dash 콜백도 같은 출처와 인증을 확인합니다.
현재 Firestore 규칙은 클라이언트 접근 전부 거절이며 그대로 유지했습니다. Admin SDK를 사용하는 서버가 사용자 권한과 대화 소유자를 검사합니다.
Firebase 프로젝트/서비스 계정 관리자는 콘솔을 통해 DB 자체에 접근할 수 있으므로, 앱 내 대화 접근 정책과 콘솔 관리 권한을 구분하세요.

신규 저장 경로:

- `manufacturingRecords/{recordId}`: 수정 전 상태와 작성자, 시각, 데이터 버전을 포함한 추가 전용 업무 이력.
- `manufacturingReviewState/{targetHash}`: 대상별 최신 상태와 충돌 방지 revision.
- `manufacturingConversations/{uid}/threads/{threadId}`: 사용자별 대화, 화면 스냅샷, 답변 및 인용 출처.

업무 기록은 최대 15초 캐시하며 저장 성공 시 무효화합니다. Copilot은 최근 30개 대화, 대화당 최대 20회 질문을 지원합니다.
로컬 FAISS 인덱스는 `.local/rag`에 저장합니다. 처음 질문할 때 공개 다국어 임베딩 모델을 다운로드하며, 이번 PC는 이미 준비했습니다. 모델 재학습은 없습니다.
문서 목록은 `backend/copilot.py`의 SOURCES로 제한합니다. 문서 변경 시 내용 해시를 비교해 인덱스를 다시 만듭니다. 대화·환경 파일·개인정보를 검색 인덱스에 넣지 않습니다.

## 현재 연결 검증 상태와 할당량

2026-09-27 검증에서 Firebase Auth 계정 로그인과 Gemini 2.5 Flash 실제 RAG 응답 생성을 확인했습니다.
Firestore는 `429 Quota exceeded`를 반환하여 **업무 이력 및 대화의 실제 저장/재조회 검증은 할당량 초기화 후 남아 있습니다.**
이때 분석 화면은 열리며 기록 상태는 조회 불가로 표시합니다. 저장 실패를 성공으로 표시하거나 로컬에 몰래 대체 저장하지 않습니다.
Copilot은 기록 저장소가 사용 불가능하면 Gemini를 호출하기 전에 중단해 무료 요청량을 낭비하지 않습니다.
무료 할당량은 태평양 시간 자정 무렵 초기화됩니다. 한국 시간 9월 기준 대략 오후 4시입니다.
Firebase Console의 Firestore 사용량을 확인한 뒤 다시 로그인하고 아래 순서로 검증하세요.

1. 품질 판정·조치 → 메모 작성 → 확인 후 저장 → 새로고침 → 업무 기록에 작성자와 판정 유지.
2. Copilot 질문 → 출처 확인 → 새로고침 → 내 대화 이력에서 같은 대화 조회.
3. 다른 계정 로그인 → 공용 업무 기록 조회 가능, 이전 계정의 대화는 목록/직접 주소에서 접근 불가.
4. 같은 대상의 확인 화면을 두 곳에서 열고 차례로 저장 → 두 번째는 충돌 안내.

Gemini 429는 무료 사용량 안내로 표시하며 모델 변경이나 유료 전환을 자동 수행하지 않습니다.

## 검증 및 문제 해결

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s backend/tests -v
```

원본 연구 산출물과 `firestore/versions`를 보유한 관리자 PC에서는 추가로
`python -m unittest discover -s firestore/tests -v`를 실행합니다.
팀원용 최소 묶음은 과거 시드와 연구 원본을 포함하지 않으며, 대신 백엔드 테스트에서
현재 시드·runtime 해시와 압축 원본의 동일성을 검사합니다.

- 실행 불가: 위 의존성을 설치하고 Python 버전을 확인하세요.
- 포트 사용 중: 이미 열린 서버를 이용하거나 포트를 변경하세요.
- 시드 해시 불일치: 시드 파일의 부분 수정을 되돌리거나 `firestore/build_unified_seed.py`로 완전한 시드를 다시 생성하세요.
- CSV 실패: UTF-8, 필수 열 6개, 동일 부품의 연속 3일, 0 이상 유한한 수량, 1MB 이하를 확인하세요.
- 기존 모델 환경 누락: 해당 트랙의 README와 requirements를 따라 환경을 복원하세요. 모델을 자동 재학습하지 않습니다.

구성: `backend/` API·데이터 / `frontend/` 화면·차트·스타일 / `run_local.py` 실행 진입점.
