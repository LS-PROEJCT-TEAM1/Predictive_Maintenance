"""Current analysis scope for the overview; no synthetic work status or priority."""
from urllib.parse import urlencode


def link(track, **params):
    return '/'+track+'?'+urlencode({'tab':'review',**params})


def overview_data(repo):
    original = repo.get('overview')
    demand = repo.demand(original['demand']['targetDate'],model=original['demand']['operatingModel'])
    maintenance = repo.maintenance(original['maintenance']['sourceFile'])
    quality = original['quality']
    review = [r for r in demand['rows'] if r['review']]
    queue = []
    for r in review:
        queue.append({'track':'demand','trackLabel':'공급망','target':r['part'],
            'title':f"계획 대비 {r['gap']:+,.0f}개 · {r['direction']}",
            'key':f"demand:{r['part']}:{demand['date']}:{demand['model']}",
            'route':link('demand',date=demand['date'],part=r['part'],model=demand['model']),
            'scope':demand['date'],'kind':'review'})
    for e in sorted(maintenance['events'],key=lambda e:-e['maxRisk']):
        queue.append({'track':'maintenance','trackLabel':'예지보전','target':e['event'],
            'title':f"{e['type']} · {e['start']}–{e['end']}행 · 위험비 {e['maxRisk']:.2f}배",
            'key':'maintenance:'+e['id'],'eventId':e['id'],
            'route':link('maintenance',run=maintenance['run'],supervised=maintenance['supervised'],unsupervised=maintenance['unsupervised'],event=e['id']),
            'scope':maintenance['run'],'kind':'review'})
    queue.append({'track':'quality','trackLabel':'품질 보증','target':quality['testId'],
        'title':f"이상 {quality['abnormalSegmentCount']}구간 · {quality['suspectedCell']} 근거 확인",
        'key':'quality:'+quality['testId'],'route':link('quality',test=quality['testId'],cell=quality['suspectedCell']),
        'scope':quality['testId'],'kind':'decision'})
    return {**original,'demand':{**original['demand'],'recommendedForecast':demand['forecast'],
                'planReference':demand['plan'],'planGapPct':demand['gapPct'],'reviewCount':len(review),
                'upwardReviewCount':sum(r['gap']>0 for r in review),'downwardReviewCount':sum(r['gap']<0 for r in review),
                'reviewPath':link('demand',date=demand['date'],part='ALL',model=demand['model'])},
            'maintenance':{**original['maintenance'],'eventCount':len(maintenance['events']),
                'alertCycles':maintenance['alertCycles'],'cycles':maintenance['cycles'],
                'supervised':maintenance['supervised'],'unsupervised':maintenance['unsupervised'],
                'timeStart':maintenance['points'][0]['time'],'timeEnd':maintenance['points'][-1]['time'],
                'reviewPath':link('maintenance',run=maintenance['run'],supervised=maintenance['supervised'],unsupervised=maintenance['unsupervised'])},
            'quality':{**quality,'reviewPath':link('quality',test=quality['testId'],cell=quality['suspectedCell'])},
            'workItems':queue,'trend':demand['trend'],'actions':repo.collection('actions')}


def work_status(data, records):
    unavailable = bool(records.get('__unavailable__'))
    rows = []
    labels = {'clear':'이상 없음','retest':'재시험 요청','hold':'출하 보류'}
    for item in data['workItems']:
        saved = records.get(item['key'])
        stale = bool(saved and saved.get('dataVersion') != data['dataVersion'])
        if unavailable:
            status,done = '조회 불가',None
        elif stale:
            status,done = '재확인 필요',False
        elif item['kind']=='decision':
            status = labels.get((saved or {}).get('decision'),'판정 대기')
            done = (saved or {}).get('decision')=='clear'
        else:
            done = (saved or {}).get('decision')=='reviewed'
            status = '확인 완료' if done else '미확인'
        rows.append({**item,'status':status,'done':done})
    counts = {}
    for track in ('demand','maintenance','quality'):
        group = [r for r in rows if r['track']==track]
        counts[track] = {'total':len(group),'done':None if unavailable else sum(r['done'] for r in group),
                         'remaining':None if unavailable else sum(not r['done'] for r in group)}
    return rows,counts
