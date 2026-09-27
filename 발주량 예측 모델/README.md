# D+3 발주량 예측 — audited_v2

가이드북의 일별 최종 ERP 발주 계획량을 달력 3일 앞서 예측합니다. 일별 마지막 로그 이후 사용하며 실측 소비량·재고 최적화 모델은 아닙니다.

현재 정책은 **7일 이동평균 기본 / LSTM 학습형 보조**입니다. CV MAE는 각각 **34.6604 / 35.1942개**입니다. 마지막 7개 목표일은 이전 실험에서 본 자료의 회고 평가입니다. 미관측 독립 시험이라고 표현하지 않습니다. 원본 117부품 중 충돌한 Part 21·26을 격리하고 115부품의 일별 자료를 사용합니다. 최근값·3일 평균·7일 평균·원본 계획·트리 3종·LSTM·연관 LightGBM을 비교했습니다.

- [학습·검증 설계](docs/TRAINING_PLAN_V2.md)
- [재학습 결과와 한계](docs/RETRAINING_REPORT_2026-09-27.md)
- [보완 전 감사](docs/ML_AUDIT_2026-09-27.md)

## 재현 순서

연구 폴더의 requirements.txt를 Python 3.12 환경에 설치합니다. 저장된 완료 버전은 보존해야 하므로 학습 스크립트는 같은 완료 버전의 덮어쓰기를 거부합니다. 새 실험은 버전을 분리합니다. 아래 순서는 최초 새 버전 생성 시 사용합니다.

```powershell
python src/retrain_audited.py
python src/build_dashboard_assets.py
python src/train_related_part_model.py
python -m unittest discover -s tests -v
```

`train_related_part_model.py`는 통합 학습 결과의 연관 비교 자료를 내보내는 호환 명령입니다. 별도로 holdout 기반 모델을 재학습하지 않습니다. `train_final.py`, `evaluate_purged.py` 등 이전 학습 스크립트와 purged_v1/final_v3 모델은 과거 실험 자료이며 현재 시드 생성 입력이 아닙니다.

루트에서 `firestore/build_unified_seed.py`, `scripts/package_runtime.py` 순으로 실행하면 공식 로컬 시드와 실행 자료를 갱신합니다. Firebase 업로드는 별도 동작이며 이번에는 하지 않았습니다.

## 실행

통합 앱은 루트 `setup_local.cmd` 후 `start_demo.cmd`로 실행합니다. 8070 포트에서 계정·외부 API 없이 테스트합니다. 연구용 단독 화면은 이 폴더에서 `python app.py`로 실행합니다.

CSV는 동일 부품 3~60일 자료, 최근 3일은 연속이어야 하며 8일 이력을 권장합니다. 미학습 부품은 최근값으로 대체하고, 격리된 두 부품은 정정 전까지 예측하지 않습니다.

시간대 합계 민감도에서는 LSTM이 근소하게 1위여서 순위가 완전히 안정적이지 않습니다. 7일 평균 대 3일 평균 차이의 bootstrap 구간도 0을 포함합니다. 새 기간 평가와 원본 정정 없이 현장 성능을 확정할 수 없습니다.
