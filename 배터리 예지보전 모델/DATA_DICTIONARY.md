# 트랙 B 데이터 사전

가이드북의 제조AI 데이터셋 주요 변수 정의를 기준으로 정리했다.

| 변수 | 의미 | 단위/형식 | 모델 활용 |
| --- | --- | --- | --- |
| PageNo | 39단계 용접 작업 시퀀스 번호 | Count, 1~39 | 사이클 생성, 공정 위치, 주기형 특징 |
| Speed | 설정 길이를 기준으로 한 모터 속도 | mm/s | 지도·잔차 모델 입력 |
| Length | 용접할 부분의 설정 길이 | mm | 지도·잔차 모델 입력 |
| RealPower | 용접 포인트별 실제 측정 출력 | W | 핵심 실측값, 이상점수 및 지도 모델 입력 |
| SetFrequency | 초당 발광 횟수 설정 | Hz | 데이터 품질·설정값 확인; 현재 학습자료에서는 상수열 |
| SetDuty | 최대 용접 출력 설정 | % | 데이터 품질·설정값 확인; 현재 학습자료에서는 상수열 |
| SetPower | 재질에 따라 설정되는 용접 출력 | % | 설정 대비 실측 편차·비율 계산 |
| GateOnTime | 용접 게이트 오픈 시간 | 원문 표기는 s | 지도·잔차 모델 입력. 값 범위상 실제 단위는 현장 확인 필요 |
| WorkingTime | 작업 수행 시각 | datetime | 사이클 시간순 정렬, 시간 간격·탐지 지연 계산 |
| label | 정상/이상 행 라벨 | 0=정상, 1=이상 | 지도학습 목표와 최종 평가 |
| cycle_local | 파일 내부 용접 사이클 번호 | Count | PageNo=1마다 증가 |
| group_id | 파일명과 cycle_local의 결합키 | 문자열 | train/validation/test 그룹 누수 방지 |

## 파생 특징

| 특징 | 정의 |
| --- | --- |
| PreviousRealPower | 같은 용접 사이클의 직전 RealPower |
| RealPowerDelta | 현재 RealPower - PreviousRealPower |
| TimeGapSeconds | 같은 용접 사이클 내 직전 행과의 시간 차 |
| PageSin, PageCos | PageNo의 39단계 주기형 표현 |
| PhaseSignedZ | 정상 학습 데이터의 같은 PageNo RealPower 중앙값에서 벗어난 방향 포함 robust Z-score |
| PhaseAbsZ | PhaseSignedZ의 절댓값; 정상 공정 위치별 이탈 정도 |
| RelativePowerError | 같은 PageNo의 정상 RealPower 중앙값 대비 현재 절대 편차 비율 |
| PreviousPhaseSignedZ | 직전 PageNo RealPower의 정상 기준 방향 포함 robust Z-score |
| PreviousPhaseAbsZ | 직전 PageNo의 정상 기준 이탈 절댓값 |

GateOnTime의 가이드북 단위 표기와 실제 값 스케일이 일치하는지는 설비 담당자 확인이 필요하다.

RealPower는 W, SetPower는 %이므로 두 값을 직접 빼거나 나눈 파생 특징은 사용하지 않는다. 두 변수의 관계는 모델이 각각의 입력으로 학습한다.

Phase 계열 특징의 중앙값과 스케일은 `Training_Data` 정상 학습 구간에서만 계산한다. 잠금 시험 파일의 통계량은 사용하지 않는다.
