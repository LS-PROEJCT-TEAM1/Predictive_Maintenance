# 배터리팩 용접 이상탐지 모델 비교

가이드북 기준선 2개와 주력 모델 3개를 동일한 데이터 분할로 비교한다.

## 실행

```powershell
& 'C:\Users\USER\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\src\compare_models.py
```

가이드북의 공식 `pytorch_forecasting.NHiTS` 구조와 2구간 Z-score를 재현하려면 프로젝트 전용 가상환경에서 다음을 실행한다.

```powershell
& '.\.venv-guidebook\Scripts\python.exe' .\src\reproduce_guidebook.py
```

재현 의존성은 `requirements-guidebook.txt`에 고정했다.

RobustZ, LightGBM, 가이드북 구조 N-HiTS, 가이드북 2구간 Z-score를 동일한 학습·보정·평가 규칙으로 비교하려면 다음을 실행한다.

```powershell
& '.\.venv-guidebook\Scripts\python.exe' .\src\compare_fair_four_models.py
```

결과는 `outputs/model_comparison`에 생성된다.

## 비교 모델

1. 경량 N-HiTS + Z-score 4 기준선
2. PageNo별 robust Z-score
3. LightGBM 정상 출력 회귀
4. EWMA/CUSUM 변화 탐지
5. 39행 사이클 오토인코더

## 모델 선정 순서

1. 이벤트 단위 이상 누락 최소화
2. 정상 파일의 연속 오경보 이벤트 최소화
3. 03번·04번 Macro F1 최대화
4. 탐지 지연 최소화
5. 추론 속도 비교

테스트 라벨은 최종 평가에만 사용하며 임계값 보정에는 사용하지 않는다.
