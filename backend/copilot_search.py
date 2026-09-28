"""Read-only, allowlisted entity resolution; numbers come from repositories, not embeddings."""
import re
from urllib.parse import urlencode


def resolve(repo, question, screen, history):
    context = dict(screen)
    question = question.strip()
    meta = repo.meta()
    # Carry a referenced target across screens. A new
    # explicit entity always wins. Never interpret arbitrary transcript commands.
    if re.search(r'그\s*(부품|시험|이벤트|셀)|방금|같은\s*대상', question):
        previous = next((m.get('resolvedContext') for m in reversed(history) if m.get('resolvedContext')),None)
        if previous:
            context.update(previous)
    parts = re.findall(r'\bPart\s*[-#]?\s*(\d+)(?!\d)',question,re.I)
    parts += re.findall(r'(\d+)\s*번\s*부품|부품\s*(\d+)(?!\d|\s*개)',question)
    parts = list(dict.fromkeys('Part '+str(int(next(x for x in p if x) if isinstance(p,tuple) else p)) for p in parts))
    tests = re.findall(r'(?<!Welding)\bTest\s*0*(\d+)(?:_(OK|NG)_(chg|dchg))?',question,re.I)
    runs = re.findall(r'WeldingTest[_\s-]*0*(\d+)(?:_(OK|NG))?',question,re.I)
    domains = sum(bool(x) for x in [parts,tests,runs])
    if domains>1 or len(parts)>3 or len(tests)>1 or len(runs)>1:
        return context,'부품은 최대 3개, 시험은 한 개씩 지정해 주세요. 서로 다른 업무의 대상을 한 번에 비교할 수 없습니다.'
    explicit = bool(parts or tests or runs)
    if parts:
        context.update(track='demand',part=parts[0] if len(parts)==1 else 'ALL',parts=parts if len(parts)>1 else [])
        missing=[p for p in parts if p not in meta['parts']]
        if missing:
            return context,f'{", ".join(missing)}은 공식 자료에 없습니다. 등록된 부품 번호를 확인해 주세요.'
    elif tests:
        num,kind,mode=tests[0]
        matches=[t for t in meta['tests'] if t.lower().startswith(f'test{int(num):02d}_') and (not kind or '_'+kind.lower()+'_' in t.lower()) and (not mode or t.lower().endswith('_'+mode.lower()))]
        if len(matches)!=1:
            return context,'해당 품질 시험을 찾지 못했습니다. 현재 제공 시험은 '+', '.join(meta['tests'])+'입니다.'
        context.update(track='quality',test=matches[0],progress=100,basis='clean')
    elif runs:
        num,kind=runs[0]
        matches=[t for t in meta['runs'] if t.lower().startswith(f'weldingtest_{int(num):02d}_') and (not kind or t.lower().endswith('_'+kind.lower()))]
        if len(matches)!=1:
            return context,'해당 용접 시험이 없습니다. '+', '.join(meta['runs'])+' 중에서 지정해 주세요.'
        context.update(track='maintenance',run=matches[0])
    elif re.search('발주|공급망|부품|수요|이동평균',question):
        context['track']='demand'
    elif re.search('예지보전|용접 출력|설비|사이클|이벤트|고장',question):
        context['track']='maintenance'
    elif re.search('품질|용량 불량|용접불량|센서|불량 유형|셀 전압',question):
        context['track']='quality'
    if context['track']=='demand':
        context.setdefault('part','ALL')
        context['date']=context.get('date') or meta['dates'][-1]
        dates=re.findall(r'\b\d{4}-\d{2}-\d{2}\b',question)
        if dates:
            if len(set(dates))!=1 or dates[0] not in meta['dates']:
                return context,'해당 목표일의 예측 자료가 없습니다. 제공 범위는 '+meta['dates'][0]+' ~ '+meta['dates'][-1]+'입니다.'
            context['date']=dates[0]
        if context.get('part') not in meta['parts']+['ALL']:
            return context,'등록된 부품을 선택해 주세요.'
        if not explicit and re.search('가장|상위|큰 부품|검토 부품|목록',question):
            context['part']='ALL'
            context['parts']=[]
    cells=re.findall(r'\bM\d{2}CV\d{2}(?![a-zA-Z0-9])',question,re.I)
    if (cells and (parts or runs)) or (re.search(r'\bEVT[-\s]?\d+',question,re.I) and (parts or tests or cells)):
        return context,'부품·품질 시험·용접 이벤트는 업무별로 나누어 질문해 주세요.'
    if cells:
        if len(set(cells))!=1 or not re.fullmatch(r'M(0[1-9]|1[0-6])CV(0[1-9]|1[01])',cells[0].upper()):
            return context,'셀은 M01~M16, CV01~CV11 범위에서 한 개를 지정해 주세요.'
        context.update(track='quality',cell=cells[0].upper())
    events=re.findall(r'\bEVT[-\s]?0*(\d+)(?!\d)',question,re.I)
    if events:
        if len(events)!=1:
            return context,'이벤트를 하나씩 지정해 주세요.'
        context.update(track='maintenance',event=f'EVT-{int(events[0]):03d}')
    context['resolution']='질문에서 지정한 대상' if explicit else '화면 선택 또는 질문 영역'
    return context,None


def search(repo, context, question, records_provider=None):
    track=context.get('track','overview')
    links=[]
    def link(label,path,**params):
        links.append({'label':label,'url':path+'?'+urlencode(params)})
    facts={'track':track,'dataVersion':repo.manifest['dataVersion'],'dataSource':repo.source_description}
    if track=='demand':
        d=repo.demand(context.get('date'),context.get('part','ALL'),context.get('model'))
        compared=context.get('parts',[])
        if compared:
            if any(repo.parts[p].get('quarantined') for p in compared):
                return facts,[],'비교 대상에 원본 충돌로 격리된 부품이 있습니다. Part 21·26은 예측에서 제외합니다.'
            d['rows']=[r for r in d['rows'] if r['part'] in compared]
            if len(d['rows'])!=len(compared):return facts,[],'선택일에 비교 대상 모두의 예측이 있지 않습니다.'
            d.update(forecast=sum(r['forecast'] for r in d['rows']),plan=sum(r['plan'] for r in d['rows']),
                     gap=sum(r['gap'] for r in d['rows']),count=len(d['rows']),reviewCount=sum(r['review'] for r in d['rows']))
            d['gapPct']=d['gap']/d['plan']*100 if d['plan'] else None
            facts['comparedParts']=compared
        if (d.get('partInfo') or {}).get('quarantined'):
            return facts,[],f"{d['part']}은 동일 시각 원본 수량 충돌로 격리되어 있습니다. 예측값을 제공하지 않습니다."
        if not d['count']:
            return facts,[],'선택한 부품·목표일의 예측 자료가 없습니다.'
        facts.update({k:d[k] for k in ['date','part','model','forecast','plan','gap','gapPct','reviewCount','count']})
        rows=[r for r in d['rows'] if r['review']] if d['part']=='ALL' and not compared else d['rows']
        if re.search('하향|줄일|감소',question):rows=[r for r in rows if r['gap']<0]
        if re.search('상향|늘릴|증가',question):rows=[r for r in rows if r['gap']>0]
        facts['targetDefinition']='일별 최종 ERP 발주 계획량, 실측 소비량·최적 재고 아님'
        facts['primaryModel']=d['config']['primaryModel'];facts['auxiliaryModel']=d['config']['auxiliaryModel']
        facts['validation']=d['config']['walkForwardMetrics']
        facts['matchedCount']=len(rows)
        rows=sorted(rows,key=lambda r:-abs(r['gap']))
        facts['matches']=rows[:5]
        facts['ranking']='계획과 예측 차이의 절댓값 내림차순 · 최대 5개'
        for row in rows[:3]:
            link(row['part']+' 발주 검토','/demand',tab='review',date=d['date'],part=row['part'],model=d['model'])
        record_keys=[f"demand:{r['part']}:{d['date']}:{d['model']}" for r in rows]
    elif track=='maintenance':
        d=repo.maintenance(context.get('run','WeldingTest_04_NG'),context.get('supervised'),context.get('unsupervised'))
        events=sorted(d['events'],key=lambda e:-e['maxRisk'])
        if re.search(r'0\s*(W|와트|출력)|미출력|출력\s*0',question,re.I):
            zero_rows={p['row'] for p in d['points'] if p['power']==0}
            events=[e for e in events if any(e['start']<=r<=e['end'] for r in zero_rows)]
        if context.get('event'):
            events=[e for e in events if e['event']==context['event']]
            if not events:return facts,[],'선택 시험·모델에 해당 이벤트가 없습니다.'
        facts.update({k:d[k] for k in ['run','supervised','unsupervised','anomalyRows','cycles','maxRisk','alertCycles']})
        facts.update(eventCount=len(d['events']),events=events[:5],zeroPowerRows=sum(p['power']==0 for p in d['points']),
                     interpretation='현재 이상 탐지. 위험비는 점수/임계값이며 고장 확률·남은 수명이 아님. 신규 시점 일반화 검증 미완료.')
        for e in events[:3]:link(e['event']+' 이벤트 검토','/maintenance',tab='review',run=d['run'],supervised=d['supervised'],unsupervised=d['unsupervised'],event=e['id'])
        record_keys=['maintenance:'+e['id'] for e in events]
    elif track=='quality':
        d=repo.quality(context.get('test','Test07_NG_dchg'),context.get('cell','M02CV01'),context.get('progress',100),context.get('basis','clean'))
        facts.update({k:d[k] for k in ['testId','selectedCell','rows','abnormalPointCount','abnormalSegmentCount','progress','basis','defectEvidence'] if k in d})
        facts['suspectedCells']=d.get('suspectedCells',[])[:5]
        facts['snapshot']={k:v for k,v in d['snapshot'].items() if k not in ['cells','temperatures']}
        facts['selectedCellValues']=d['cellSeries'][-1] if d['cellSeries'] else None
        facts['interpretation']='이상 수·불량 유형 근거는 전체 시험 기준, 선택 셀 수치는 표시 시점 기준. 의심 셀은 원인 확정이나 셀 정답이 아님. 작업자 판정과 모델 예측은 별도.'
        link(d['testId']+' 품질 판정','/quality',tab='review',test=d['testId'],cell=d['selectedCell'])
        record_keys=['quality:'+d['testId']]
    else:
        from backend.overview import overview_data
        d=overview_data(repo)
        facts.update({k:d[k] for k in ['demand','maintenance','quality']})
        facts['interpretation']='서로 다른 시기의 독립 자료. 분야 간 위험도를 비교하거나 현장 우선순위를 확정할 수 없음.'
        for t,label in [('demand','발주 검토'),('maintenance','이벤트 검토'),('quality','품질 판정')]:
            links.append({'label':label,'url':d[t]['reviewPath']})
        record_keys=[r['key'] for r in d['workItems']]
    if re.search('미확인|확인 완료|작업자|판정 기록|메모|누가|점검 기록|검토 기록',question):
        try:
            records=records_provider() if records_provider else None
            if records is None:raise ValueError('unavailable')
            facts['recordCount']=len(record_keys)
            facts['unreviewedCount']=sum(key not in records or records[key].get('dataVersion')!=facts['dataVersion'] or records[key].get('decision') not in ['reviewed','clear'] for key in record_keys)
            if track=='demand' and '미확인' in question:
                rows=[r for r in rows if (saved:=records.get(f"demand:{r['part']}:{d['date']}:{d['model']}")) is None or saved.get('dataVersion')!=facts['dataVersion'] or saved.get('decision')!='reviewed']
                facts['matches']=rows[:5];facts['matchedCount']=len(rows)
                links.clear()
                for row in rows[:3]:link(row['part']+' 발주 검토','/demand',tab='review',date=d['date'],part=row['part'],model=d['model'])
            facts['records']=[{'key':key,**{k:records.get(key,{}).get(k) for k in ['decision','actor','at','dataVersion']},
                               'note':str(records.get(key,{}).get('note',''))[:300],
                               'state':'미확인' if key not in records else '재확인 필요' if records[key].get('dataVersion')!=facts['dataVersion'] else '저장된 판정'} for key in record_keys[:120]]
            facts['records']=facts['records'][:5]
            facts['recordScope']='선택 분석 대상의 최신 공용 기록 · 전체 변경 이력이 아님'
        except Exception:
            facts['recordScope']='조회 불가 · 미확인 0건으로 해석하지 말 것'
    if track=='demand' and re.search('가장|상위|큰 부품',question) and not context.get('parts'):
        requested=re.search(r'(\d+)\s*개',question)
        count=min(5,max(1,int(requested.group(1)))) if requested else 3
        facts['rankedResults']=facts['matches'][:count]
        facts['rankingLimit']=count
    return facts,links,None


def ranked_answer(facts):
    """Never let a language model reorder signed gaps or substitute percentages."""
    rows=facts['rankedResults']
    head=f"{facts['date']} · {facts['model']} 기준, 계획과 예측 차이의 절댓값이 큰 순서입니다."
    lines=[f"{i}. {r['part']}: 예측 {r['forecast']:,.2f}개 / 계획 {r['plan']:,.2f}개 / 차이 {r['gap']:+,.2f}개 ({r['direction']})" for i,r in enumerate(rows,1)]
    return '\n'.join([head,*lines,'자동 발주 지시가 아닌 검토 대상입니다.' if rows else '현재 조건에 해당하는 검토 대상이 없습니다.'])
