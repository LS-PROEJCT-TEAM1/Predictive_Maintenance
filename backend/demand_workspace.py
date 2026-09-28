"""Date-aligned evidence and input checks for the existing frozen demand models."""
import io
import json
from functools import lru_cache
import numpy as np
import pandas as pd
from backend.runtime_assets import RUNTIME


@lru_cache(maxsize=1)
def evidence():
    return json.loads((RUNTIME/'demand/evidence.json').read_text(encoding='utf-8'))


def plan_history(repo, data):
    daily = pd.DataFrame(evidence()['daily'])
    ids = {r['part'] for r in data['rows']}
    target = pd.Timestamp(data['date'])
    rows = []
    for horizon in (5,4,3,2,1):
        issued = (target-pd.Timedelta(days=horizon)).date().isoformat()
        sample = daily[daily.part_number.isin(ids) & daily.date.eq(issued)]
        complete = len(sample)==len(ids) and bool(ids)
        rows.append({'시점':f'{horizon}일 전','기록일':issued,'계획량':float(sample[f'plan_d{horizon}'].sum()) if complete else None,
                     '대상 부품':len(sample),'전체 대상':len(ids),'사용 구분':'예측 당시 확인 가능' if horizon>=3 else '사후 변경 · 예측 미사용'})
    return rows


@lru_cache(maxsize=128)
def diagnostics(part='ALL'):
    source = evidence()
    daily = pd.DataFrame(source['daily'])
    cv = pd.DataFrame(source['cv'])
    if part!='ALL':
        cv = cv[cv.part_number.eq(part)]
    aligned = cv.copy()
    for h in (5,4,3,2,1):
        aligned['issued'] = (pd.to_datetime(aligned.target_date)-pd.Timedelta(days=h)).dt.strftime('%Y-%m-%d')
        aligned = aligned.merge(daily[['part_number','date',f'plan_d{h}']].rename(columns={'date':'issued'}),on=['part_number','issued'],how='left',validate='many_to_one')
    common = aligned.dropna(subset=[f'plan_d{h}' for h in (5,4,3,2,1)])
    metrics = []
    for h in (5,4,3,2,1):
        x,y = common[f'plan_d{h}'],common.actual
        metrics.append({'계획 시점':f'{h}일 전','행 수':len(common),'MAE':float((x-y).abs().mean()) if len(common) else None,
            'Bias':float((x-y).mean()) if len(common) else None,
            '상관계수':float(x.corr(y)) if len(common)>1 and x.std()>0 and y.std()>0 else None,
            '사용 구분':'예측 당시 확인 가능' if h>=3 else '사후 참고 · 예측 미사용'})
    inputs = cv.merge(daily[['part_number','date','actual_d','plan_d3','plan_d4','plan_d5']],left_on=['part_number','origin_date'],right_on=['part_number','date'],how='inner',validate='many_to_one')
    cols = ['actual_d','plan_d3','plan_d4','plan_d5','actual']
    corr = inputs[cols].corr().replace({np.nan:None}).values.tolist()
    return {'metrics':metrics,'correlation':corr,'correlationRows':len(inputs),'quality':source['quality'],
        'importance':source['importance'],'importanceMethod':source['importanceMethod'],
        'period':{'start':cv.target_date.min() if len(cv) else None,'end':cv.target_date.max() if len(cv) else None},
        'part':part,'version':source['version']}


def inspect_csv(contents, quarantined):
    checks = []
    def check(name, ok, detail):
        checks.append({'검사 항목':name,'결과':'정상' if ok else '오류','설명':detail})
    if len(contents)>1_000_000:
        check('파일 크기',False,'1MB 이하 CSV만 지원합니다.')
        return {'valid':False,'checks':checks}
    try:
        frame = pd.read_csv(io.BytesIO(contents),encoding='utf-8-sig')
    except Exception:
        check('CSV 읽기',False,'UTF-8 CSV 템플릿을 사용하세요.')
        return {'valid':False,'checks':checks}
    required = ['part_number','date','actual_d','plan_d3','plan_d4','plan_d5']
    missing = set(required)-set(frame.columns)
    check('필수 6개 열',not missing,'누락: '+', '.join(sorted(missing)) if missing else '필수 열 확인')
    check('입력 행 수',3<=len(frame)<=60,f'{len(frame)}행 · 허용 3~60행')
    if missing:
        return {'valid':False,'checks':checks}
    parts = frame.part_number.astype('string').str.strip()
    check('동일 부품',parts.notna().all() and parts.ne('').all() and parts.nunique()==1,'공백 없이 동일 부품 1종이어야 합니다.')
    check('격리 부품',not parts.isin(quarantined).any(),'Part 21·26은 원본 정정 전 예측 제외')
    dates = pd.to_datetime(frame.date,errors='coerce').dt.normalize().sort_values()
    check('날짜·중복',dates.notna().all() and not dates.duplicated().any(),'유효한 날짜 · 하루 한 행')
    check('최근 3일 연속',len(dates)>=3 and dates.tail(3).diff().dropna().eq(pd.Timedelta(days=1)).all(),'날짜순 정렬 후 최근 3일 확인')
    numbers = frame[required[2:]].apply(pd.to_numeric,errors='coerce')
    check('수량·결측',np.isfinite(numbers.to_numpy()).all() and numbers.ge(0).all().all(),'유한한 0 이상 수량 · 0은 유지')
    checks.append({'검사 항목':'이력 길이','결과':'참고','설명':f'{len(frame)}일 입력 · 8일 권장, 부족한 이력은 기존 모델 대체 규칙 적용'})
    return {'valid':all(x['결과']!='오류' for x in checks),'checks':checks}
