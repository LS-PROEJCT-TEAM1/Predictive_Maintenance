"""Work status and context-preserving entry points, using actual current records."""
from dash import Input, Output, dcc, html, no_update
import dash_mantine_components as dmc
from backend.overview import work_status
from frontend.components import badge, graph, grid, metric_rows, num, panel
from frontend.charts import demand_chart


def remaining(value, unit):
    return '조회 불가' if value is None else f'{value:,}{unit}'


def overview(data, meta, actions):
    d,m,q = data['demand'],data['maintenance'],data['quality']
    rows,counts = work_status(data,actions)
    quality = next(r for r in rows if r['track']=='quality')
    pending_event = next((r for r in rows if r['track']=='maintenance' and not r['done']),None)
    event_path = pending_event['route'] if pending_event else m['reviewPath']
    gap = '—' if d['planGapPct'] is None else f"{d['planGapPct']:+.2f}%"

    def card(title, state, value, unit, label, context, details, task_label, task_value, href, cta, tone='info'):
        return html.Section([
            html.Div([html.Strong(title),badge(state,tone)],className='track-head'),
            html.Div(context,className='overview-context'),
            html.Div([html.Div([html.Div([value,html.Small(unit)],className='track-value'),html.Div(label,className='track-label')]),
                html.Div([html.Div([html.Small(name),html.Strong(val)]) for name,val in details],className='overview-facts')],className='track-summary'),
            html.Div([html.Div([html.Small(task_label),html.Strong(task_value)]),dcc.Link(cta+' →',href=href)],className='overview-action')],className='track-card')

    cards = [
        card('공급망 예측','검토 대상 있음' if d['reviewCount'] else '검토 대상 없음',num(d['recommendedForecast']),'개','D+3 예상 발주량',
            f"목표일 {d['targetDate']} · {d['operatingModel']}",
            [('계획 대비',gap),('상향 검토',f"{d['upwardReviewCount']}개"),('하향 검토',f"{d['downwardReviewCount']}개")],
            '미확인 검토 부품',remaining(counts['demand']['remaining'],'개'),d['reviewPath'],'발주 검토'),
        card('예지보전','점검 대상 있음' if m['alertCycles'] else '경보 없음',str(m['alertCycles']),'회','점검 대상 사이클',
            m['sourceFile']+' · '+m['timeStart'][:10],
            [('전체 사이클',f"{m['cycles']}회"),('경보 이벤트',f"{m['eventCount']}건"),('경보 조합','지도 OR 비지도')],
            '미확인 이벤트',remaining(counts['maintenance']['remaining'],'건'),event_path,'이벤트 검토'),
        card('품질 보증','AI 검토 필요' if q['abnormalSegmentCount'] else 'AI 이상 미탐지',str(q['abnormalSegmentCount']),'구간','AI 이상 구간',
            q['testId']+' · '+('방전' if q['mode']=='discharge' else '충전'),
            [('이상 시점',f"{q['abnormalPointCount']}행"),('우선 확인 셀',q['suspectedCell'])],
            '작업자 판정',quality['status'],q['reviewPath'],'품질 판정')]

    trend = demand_chart(data['trend'])
    trend.update_layout(height=250)
    return html.Div([
        html.Div([html.Span('선택 자료의 검토 현황'),html.Small('서로 다른 기간의 독립 데이터 · 실시간 설비 현황 아님')],className='overview-scope'),
        html.Div(cards,className='track-cards'),
        html.Div([
            panel('발주 계획과 예측 흐름',graph(trend),f"목표일 {d['targetDate']} 기준 · 전체 부품 합산",badge(d['operatingModel'],'info')),
            panel('검토 기록 상태',[
                metric_rows([('발주 검토 완료',remaining(counts['demand']['done'],'개'),f"현재 검토 대상 {counts['demand']['total']}개 기준"),
                    ('이벤트 확인 완료',remaining(counts['maintenance']['done'],'건'),'현재 시험·모델·경보 정책의 기록만 집계'),
                    ('품질 작업자 판정',quality['status'],'AI 이상 탐지와 별도로 기록된 판단')]),
                html.P('재시험·출하 보류는 요청 상태입니다. 자료 변경 시 재확인이 필요합니다.',className='section-note'),
                html.P('기록 조회 불가 · 미확인 수를 계산하지 않았습니다.' if actions.get('__unavailable__') else '조회 시점 기준 · 새로고침으로 갱신',className='section-note')])],className='grid-main'),
        panel('검토 대상 목록',[
            dmc.SegmentedControl(id='overview-work-track',value='all',data=[{'label':label,'value':value} for value,label in [('all','전체'),('demand','공급망'),('maintenance','예지보전'),('quality','품질 보증')]],className='overview-work-filter'),
            grid(rows,'main-grid',[{'field':'trackLabel','headerName':'영역','maxWidth':135},
                {'field':'target','headerName':'대상','minWidth':175,'tooltipField':'title'},{'field':'scope','headerName':'자료 기준','minWidth':205},
                {'field':'gap','headerName':'계획 대비 (개)','type':'numericColumn','minWidth':160,
                 'valueFormatter':{'function':"params.value == null ? '—' : (params.value > 0 ? '+' : '') + params.value.toLocaleString('ko-KR', {maximumFractionDigits: 0})"}},
                {'field':'direction','headerName':'검토 방향','minWidth':150,'cellClassRules':{
                    'review-direction-up':"params.data && params.data.track === 'demand' && params.data.gap > 0",
                    'review-direction-down':"params.data && params.data.track === 'demand' && params.data.gap < 0"}},{'field':'status','headerName':'기록 상태','minWidth':140}],340)],
            '행 선택 시 검토 화면으로 이동 · 대상에 마우스를 올리면 상세 근거 표시'),
        html.Div([html.Span('기록: '+('조회 불가' if actions.get('__unavailable__') else '공유 기록 조회됨')),
            html.Span('대상을 선택하면 해당 검토 화면으로 이동합니다.')],className='overview-scope')],className='overview-workspace')


def register(app):
    @app.callback(Output('main-grid','rowData'),Input('overview-work-track','value',allow_optional=True),Input('view-data','data'),prevent_initial_call=True)
    def filter_work(track,view):
        if not view or view['domain']!='overview' or view['tab']!='summary' or not track:
            return no_update
        return [r for r in view['payload'].get('workRows',[]) if track=='all' or r['track']==track]
