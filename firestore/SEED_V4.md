# 공식 로컬 시드 2026-09-27.v4

예지보전 track-b-audited-v3 재학습 결과와 버전이 있는 경보 정책을 반영한다. DB 스키마는 3을 유지한다.

- 원본 데이터는 보존하고 정상 학습의 0 출력 1사이클·내부 시간 역전 1사이클을 격리했다.
- 6개 저장 모델로 시험 5,226행을 다시 추론했다. 이전 측정 시드를 복사하여 점수를 재사용하지 않는다.
- 회고 평가, 입력 계약, 정상 전용 시간 검증 실패, 사이클/조합별 평가를 화면에서 확인할 수 있다.
- 120초 초과 수집 공백에서 경보를 분리하여 문서 수는 275개다. 실제 이상 사건 수와 운영 경보 수를 구분한다.
- 수요·품질 모델은 재학습하지 않았다. 통합 데이터 버전 표기만 함께 갱신된다.
- 생성 당시에는 로컬 전용이었으며, **2026-09-28 Firestore 반영과 275개 전체 일치 검증을 완료**했다. 최신 기록은 [DEPLOYMENT_STATUS.md](DEPLOYMENT_STATUS.md)를 참고한다.

재생성 순서(저장소 루트):

```powershell
& '배터리 예지보전 모델/.venv/Scripts/python.exe' '배터리 예지보전 모델/src/track_b_final_v2.py'
& '배터리 예지보전 모델/.venv/Scripts/python.exe' '배터리 예지보전 모델/scripts/build_firestore_seed.py'
& '.venv/Scripts/python.exe' firestore/build_unified_seed.py
& '.venv/Scripts/python.exe' scripts/package_runtime.py
& '.venv/Scripts/python.exe' run_local.py --demo --port 8073
```

275개는 이번 산출물 수이며 다음 모델/경보 정책에서 달라질 수 있다. manifest의 문서 수와 해시를 검증한다.
