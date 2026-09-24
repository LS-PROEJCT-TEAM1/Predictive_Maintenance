# 데이터 출처

## 공급망 최적화 AI 데이터셋

- 파일: `Dataset_공급망 최적화 AI 데이터셋/data/data.xls`
- 행 수: 17,364
- 열 수: 84
- 부품 수: 117
- 수집 기간: 2021-09-13~2021-11-01
- 주요 식별자: Part Number
- 시간 필드: CRET_TIME
- 수량 단위: 개수

원본 파일은 수정하지 않는다. D, D+1, D+2, D+3, D+4 수량은 각 시간대별 열의 합을 사용한다. 동일 부품에 하루 여러 로그가 있으면 CRET_TIME 기준 마지막 로그를 일별 대표값으로 사용한다. 계획량과 실제량의 0은 결측치가 아니므로 유지한다.

## 문서

- `Guidebook_공급망 최적화 AI 데이터셋.pdf`: 원 데이터와 기존 LSTM 실습 참고 자료
- `프로젝트_개요_및_설계서_KAMP3종_최종개정본.pdf`: 프로젝트 목표와 산출물 요구사항. 원본은 프로젝트 폴더 밖에 있으며 실행 입력으로 사용하지 않는다.

## 파생 데이터

- `outputs/final_model_evaluation/predictions_test.csv`: 최종 holdout의 실제값과 모델별 예측값
- `outputs/final_walk_forward/predictions_walk_forward.csv`: expanding walk-forward 예측값
- `outputs/dashboard_data`: 원본과 모델 산출물에서 생성한 대시보드용 데이터

파생 데이터는 `src/build_dashboard_assets.py`로 다시 만들 수 있다. Firestore용 파일에는 원본 84개 열 전체를 복제하지 않고 대시보드 조회에 필요한 날짜, 부품, 실제 수량, 계획 수량, 예측값과 평가 지표만 포함한다.
