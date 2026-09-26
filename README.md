# BatteryFlow AI 운영센터

## 팀원용 빠른 시작

Python 3.12 설치 → Git clone → **setup_local.cmd** → **start_demo.cmd** → http://127.0.0.1:8070

현재 팀원 테스트 배포 브랜치는 **codex/team-local-preview**입니다.
clone 명령은 아래 팀원 설치 안내를 참고하세요.

계정이나 API 키 없이 네 화면·필터·CSV 내보내기·발주량 추론을 체험할 수 있습니다.
체험 모드에서는 Firebase 기록 저장과 Gemini Copilot을 사용하지 않습니다.

- [팀원 설치·데이터 구성](TEAM_SETUP.md)
- [실제 Firebase·Gemini 연결](LOCAL_RUN.md)
- 공식 DB 원본: `firestore/seed` (v2, 266문서)
- FastAPI + Dash, 실행에 필요한 압축 데이터·기존 모델은 `runtime/`에 포함

아래는 개별 연구 트랙의 설명입니다. 통합 화면은 위 8070 주소로 실행하세요.

KAMP 제조 데이터를 활용한 예지보전 및 부품 발주량 예측 프로젝트 저장소입니다.

## 프로젝트 구성

| 디렉터리 | 내용 |
|---|---|
| `발주량 예측 모델/` | 117개 부품의 D+3 발주량 예측, 5개 모델 비교, Dash 화면, Firestore 준비 데이터 |
| `배터리 예지보전 모델/` | 배터리 모듈 용접 데이터 기반 이상·고장 예지보전 모델 |
| `data/pdm/` | 공용 예지보전 원본 및 전처리 데이터 |
| `src/` | 공용 전처리 코드 |

## 발주량 예측 모델

동일 부품의 연속 3일 실제 발주량과 D+3~D+5 계획량을 이용해 마지막 입력일 기준 D+3 실제 발주량을 예측합니다.

- 비교 방법: XGBoost, LightGBM, CatBoost, LSTM, 3일 이동평균
- 누수 제거 3-fold walk-forward 전체 1위: 3일 이동평균, MAE 36.952
- 누수 제거 3-fold walk-forward 학습형 모델 1위: XGBoost, MAE 48.390
- 독립 최종 holdout 1위: CatBoost, MAE 25.315(모델 선정에는 사용하지 않음)
- 운영안: 3일 이동평균 기본 예측, XGBoost 학습형 보조 예측
- Dash: 부품별 예측 조회, 모델 성능, 계획 대비 확인 대상, 신규 CSV 추론
- Firestore: 컬렉션별 JSONL과 업로드 매니페스트 준비 완료. 자동 업로드는 하지 않음

자세한 설치 방법과 실행 순서는 [`발주량 예측 모델/README.md`](./발주량%20예측%20모델/README.md)를 확인하세요.

주요 결과:

- [`누수 제거 최종 분석 보고서`](./발주량%20예측%20모델/outputs/purged_evaluation/PURGED_EVALUATION_REPORT.md)
- [`Walk-forward 결과`](./발주량%20예측%20모델/outputs/purged_evaluation/source_total/cv_metrics_pooled.csv)
- [`Firestore 데이터 구조`](./발주량%20예측%20모델/docs/FIRESTORE_SCHEMA.md)
- [`대시보드 기능 정의`](./발주량%20예측%20모델/docs/DASHBOARD_REQUIREMENTS.md)

## 발주량 예측 Dash 실행

```powershell
cd "발주량 예측 모델"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\app.py
```

브라우저에서 `http://127.0.0.1:8050`으로 접속합니다.

## 데이터 해석 주의

발주량 데이터 수집 기간은 약 50일입니다. 장기 계절성이나 연간 수요 안정성을 검증한 결과로 해석하지 않습니다. 또한 재고량, 단가, 조달 리드타임이 없으므로 현재 예측은 발주 판단 보조용이며 재고비용 최적화 결과가 아닙니다.
