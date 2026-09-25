# 데이터 출처

## 전자부품(배터리팩) 예지보전 AI 데이터셋

- 제공기관: 중소벤처기업부, Korea AI Manufacturing Platform(KAMP)
- 구축기관: 스마트제조혁신추진단, (주)인터엑스, 네스트필드(주)
- 등록일: 2022-12-23
- 원 출처: https://www.kamp-ai.kr
- 데이터 수집기간: 2022-07-01 ~ 2022-09-30
- 수집주기: 약 3초
- 대상공정: 배터리모듈 레이저 용접 공정
- 로컬 확인일: 2026-09-25
- 다운로드일: 로컬 파일 메타데이터만으로 확인할 수 없어 미상
- 사용조건: 연구·공식 활용 전 KAMP 데이터셋 페이지와 가이드북의 자료 전달·인용 조건을 다시 확인한다.

## 로컬 원본 파일

- `data/raw_data/train/Training_Data.csv`: 정상 패턴 학습용 공정 데이터
- `data/raw_data/test/WeldingTest_01_OK.csv`: 정상 시험 파일
- `data/raw_data/test/WeldingTest_02_OK.csv`: 정상 시험 파일
- `data/raw_data/test/WeldingTest_03_NG.csv`: 고립 이상 시험 파일
- `data/raw_data/test/WeldingTest_04_NG.csv`: 연속 이상 시험 파일
- `data/preprocessed/test/WeldingTest_03_NG_Label.csv`: 03 파일 행 단위 라벨
- `data/preprocessed/test/WeldingTest_04_NG_Label.csv`: 04 파일 행 단위 라벨

실행 시 `outputs/final_track_b/run_manifest.json`에 모든 입력 파일의 SHA-256을 기록한다.

## 참고 문서

- `Guidebook_전자부품(배터리팩) 예지보전 AI 데이터셋.pdf`
- `프로젝트_개요_및_설계서_KAMP3종_최종개정본.pdf`는 현재 프로젝트 디렉터리 밖의 별도 기준 문서다.
