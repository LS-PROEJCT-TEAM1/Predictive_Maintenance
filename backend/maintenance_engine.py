"""Frozen operating pair replay; no fitting, threshold search or input repair."""
import csv
import hashlib
import io
import json
from functools import lru_cache

import numpy as np
import pandas as pd
from backend.runtime_assets import RUNTIME
from backend.maintenance_policy import POLICY, alarm_regions

MAX_BYTES = 5_000_000
MAX_ROWS = 19500
COLUMNS = ['PageNo', 'Speed', 'Length', 'RealPower', 'SetFrequency', 'SetDuty', 'SetPower', 'GateOnTime', 'WorkingTime']


@lru_cache(maxsize=1)
def assets():
    return json.loads((RUNTIME/'maintenance/frozen.json').read_text(encoding='utf-8'))


def read_csv(payload):
    if not payload or len(payload) > MAX_BYTES:
        raise ValueError('CSV는 비어 있지 않은 5MB 이하 파일이어야 합니다.')
    for encoding in ('utf-8-sig', 'cp949'):
        try:
            text = payload.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError('UTF-8 또는 CP949 CSV로 저장하세요.')
    try:
        names = [s.strip() for s in next(csv.reader(io.StringIO(text)))]
        if len(set(names)) != len(names):
            raise ValueError('열 이름이 중복됩니다. 공백 제거 후 이름을 확인하세요.')
        frame = pd.read_csv(io.StringIO(text), nrows=MAX_ROWS+1)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, StopIteration):
        raise ValueError('CSV 표 구조를 확인하세요.') from None
    if not 1 <= len(frame) <= MAX_ROWS:
        raise ValueError(f'최대 {MAX_ROWS:,}행까지 검사할 수 있습니다.')
    frame.columns = names
    return frame, encoding


def inspect(payload, labels=None):
    raw, encoding = read_csv(payload)
    missing = sorted(set(COLUMNS)-set(raw))
    if missing:
        raise ValueError('필수 열 누락: '+', '.join(missing))
    frame = raw[COLUMNS].copy()
    for col in COLUMNS[:-1]:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
        bad = ~np.isfinite(frame[col]) | (frame[col] < 0)
        if bad.any():
            raise ValueError(f'{col}: 숫자·결측·음수 오류, 원본 행 ID {np.flatnonzero(bad)[:5].tolist()} (0부터).')
    if not (frame.PageNo.eq(np.floor(frame.PageNo)) & frame.PageNo.between(1,39)).all():
        raise ValueError('PageNo는 1~39의 정수여야 합니다.')
    if not frame.SetPower.between(0,100).all() or not frame.SetDuty.between(0,100).all():
        raise ValueError('SetPower와 SetDuty는 0~100%여야 합니다.')
    if len(frame)%39 or frame.PageNo.tolist() != list(range(1,40))*(len(frame)//39):
        raise ValueError('완료된 사이클만 분석합니다. PageNo 1~39 순서·누락·중복을 확인하세요. 행은 자동 삭제하지 않습니다.')
    try:
        frame.WorkingTime = pd.to_datetime(frame.WorkingTime, format='ISO8601', errors='raise')
        delta = frame.WorkingTime.diff().dt.total_seconds()
    except (ValueError, TypeError, AttributeError):
        raise ValueError('WorkingTime은 동일 시간대의 ISO 날짜·시각이어야 합니다.') from None
    if frame.WorkingTime.isna().any() or (delta.dropna() <= 0).any():
        raise ValueError('WorkingTime에 결측·중복·시간 역전이 있습니다. 원본을 확인하세요.')
    digest = hashlib.sha256(payload).hexdigest()
    truth = None
    if labels:
        label, _ = read_csv(labels)
        if not {'sourceRow','sourceSha256','label'} <= set(label) or len(label) != len(frame):
            raise ValueError('정답은 원본과 같은 행 수의 sourceRow·sourceSha256·label 열이 필요합니다.')
        if (not np.array_equal(pd.to_numeric(label.sourceRow, errors='coerce'), np.arange(len(frame)))
                or not label.sourceSha256.eq(digest).all() or not label.label.isin([0,1]).all()):
            raise ValueError('정답의 원본 해시·행 ID·0/1 라벨이 일치하지 않습니다.')
        truth = label.label.to_numpy(int)
    points = [{'setPower':float(r.SetPower),'speed':float(r.Speed),'length':float(r.Length),
               'frequency':float(r.SetFrequency),'duty':float(r.SetDuty)} for r in frame.itertuples()]
    unseen = sum(not recipe_known(p) for p in points)
    constant_shift = int((frame.SetFrequency.ne(assets()['frequency']) | frame.SetDuty.ne(assets()['duty'])).sum())
    zero = int(frame.RealPower.eq(0).sum())
    checks = [
        {'항목':'필수 열·숫자·범위','결과':'통과','건수':len(frame),'설명':'필수 9열 · 결측/무한대/음수 차단'},
        {'항목':'시간 순서·사이클','결과':'통과','건수':len(frame)//39,'설명':'시간 증가 · PageNo 1~39 반복'},
        {'항목':'출력 0','결과':'확인 필요' if zero else '없음','건수':zero,'설명':'삭제 없이 모델 입력으로 보존'},
        {'항목':'정상 기준 학습 밖 레시피','결과':'주의' if unseen else '없음','건수':unseen,'설명':'Speed·Length·SetPower 조합 기준'},
        {'항목':'학습 고정 주파수·듀티 변경','결과':'주의' if constant_shift else '없음','건수':constant_shift,'설명':'해당 조건의 성능은 검증되지 않음'},
        {'항목':'120초 초과 수집 공백','결과':'확인','건수':int((delta > 120).sum()),'설명':'경보 이벤트를 분리하는 기준'},
        {'항목':'정답 연결','결과':'연결' if truth is not None else '정답 없음','건수':len(frame) if truth is not None else 0,'설명':'원본 SHA-256 + 행 ID 검사 · 모델 입력에 미사용'}]
    return {'frame':frame,'labels':truth,'public':{'rows':len(frame),'sha256':digest,'encoding':encoding,'checks':checks,
            'beforeAfter':[{'항목':'행 수','처리 전':len(frame),'처리 후':len(frame)},
                           {'항목':'입력 열','처리 전':len(raw.columns),'처리 후':9},
                           {'항목':'출력 0 보존','처리 전':zero,'처리 후':zero},
                           {'항목':'행 삭제·정렬·값 보간','처리 전':'없음','처리 후':'없음'}]}}


def recipe_known(point):
    recipe = [point['speed'], point['length'], point['setPower']]
    return recipe in assets()['recipes']


def replay(frame):
    a = assets()
    page = frame.PageNo.to_numpy(int)
    center = np.array(a['median'])[page-1]
    scale = np.array(a['scale'])[page-1]
    power = frame.RealPower.to_numpy(float)
    prev_page = np.where(page > 1, page-1, 39)
    previous = np.roll(power,1)
    previous[page == 1] = np.array(a['median'])[prev_page[page == 1]-1]
    angle = 2*np.pi*(page-1)/39
    gap = frame.WorkingTime.diff().dt.total_seconds().clip(upper=120).fillna(0).to_numpy()
    gap[page == 1] = 0
    signed = (power-center)/scale
    f = pd.DataFrame({'PageNo':page,'PageSin':np.sin(angle),'PageCos':np.cos(angle),'Speed':frame.Speed,
        'Length':frame.Length,'SetPower':frame.SetPower,'GateOnTime':frame.GateOnTime,'RealPower':power,
        'PreviousRealPower':previous,'RealPowerDelta':power-previous,'PhaseSignedZ':signed,'PhaseAbsZ':abs(signed),
        'RelativePowerError':abs(power-center)/np.maximum(abs(center),1),'TimeGapSeconds':gap})
    logistic = a['logistic']
    x = (f[logistic['features']].to_numpy()-logistic['center'])/logistic['scale']
    logits = x @ np.array(logistic['coef']) + logistic['intercept']
    scores = 1/(1+np.exp(-np.clip(logits,-700,700)))
    return scores, abs(signed), center


def analyze(item):
    frame = item['frame']
    sup, unsup, expected = replay(frame)
    a = assets()
    points = []
    for i,r in enumerate(frame.itertuples()):
        s, u = float(sup[i]/a['logistic']['threshold']), float(unsup[i]/a['threshold'])
        points.append({'row':i,'cycle':i//39+1,'page':int(r.PageNo),'time':str(r.WorkingTime),'power':float(r.RealPower),
            'expected':float(expected[i]),'supRatio':s,'unsupRatio':u,'risk':max(s,u),'prediction':int(s>=1 or u>=1),
            'label':int(item['labels'][i]) if item['labels'] is not None else None,
            'setPower':float(r.SetPower),'speed':float(r.Speed),'length':float(r.Length),'gateOnTime':float(r.GateOnTime)})
    data = {'run':'업로드 '+item['public']['sha256'][:12],'points':points,'supervised':'LogisticCurrent','unsupervised':'RobustPhaseZ',
        'version':'upload:'+item['public']['sha256'],'config':{'pipelineVersion':a['version'],'alarmPolicy':POLICY},
        'events':[], 'cycles':len(frame)//39,'anomalyRows':sum(p['prediction'] for p in points)}
    for i,(start,end) in enumerate(alarm_regions(points,POLICY['gapSeconds']),1):
        data['events'].append({'event':f'EVT-{i:03d}','start':start,'end':end,'rows':end-start+1,'maxRisk':max(p['risk'] for p in points[start:end+1])})
    return enrich(data)


def enrich(data):
    points = data['points']
    for p in points:
        p['unseen'] = not recipe_known(p)
        p['deviation'] = p['power']-p['expected']
        p['agreement'] = ('둘 다 경보' if p['supRatio'] >= 1 and p['unsupRatio'] >= 1 else '판정 불일치' if (p['supRatio'] >= 1) != (p['unsupRatio'] >= 1) else '둘 다 경보 없음')
    cycles = []
    for cid in sorted({p['cycle'] for p in points}):
        group = [p for p in points if p['cycle']==cid]
        related = [e['event'] for e in data['events'] if e['start']<=group[-1]['row'] and e['end']>=group[0]['row']]
        cycles.append({'cycle':cid,'상태':'점검 대상' if any(p['prediction'] for p in group) else '경보 없음',
            '시작 시각':group[0]['time'],'경보 지점':sum(p['prediction'] for p in group),'출력 0':sum(p['power']==0 for p in group),
            '학습 밖':sum(p['unseen'] for p in group),'모델 불일치':sum(p['agreement']=='판정 불일치' for p in group),
            '최대 위험비':round(max(p['risk'] for p in group),2),'관련 이벤트':', '.join(related) or '—'})
    data['cycleDetails'] = sorted(cycles,key=lambda c:(c['상태']!='점검 대상',-c['출력 0'],-c['최대 위험비'],c['cycle']))
    data['unseenRows'] = sum(p['unseen'] for p in points)
    data['alertCycles'] = sum(c['상태']=='점검 대상' for c in cycles)
    data['maxRisk'] = max(p['risk'] for p in points)
    data['referenceNotice'] = 'PageNo별 정상 학습 중앙값 · 미래 예측/공정 합격 한계가 아닙니다.'
    data['normalSummary'] = assets()['normalSummary']
    data['normalRows'] = assets()['normalRows']
    data['fitEnd'] = assets()['fitEnd']
    return data
