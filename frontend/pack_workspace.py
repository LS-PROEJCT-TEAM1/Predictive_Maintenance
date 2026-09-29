"""Pack inspection and automatic result browsing, separate from staff decisions."""
from uuid import uuid4

import httpx
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dcc, html, no_update
import dash_mantine_components as dmc

from frontend import spatial
from frontend.navigation import legacy_quality_review
from frontend.components import BLUE, RED, badge, callout, graph, grid, kpi, metric_rows, num, panel, select

STATUS = {'normal': '정상', 'warning': '경고', 'danger': '위험', 'missing': '자료 없음'}
DECISIONS = {'clear': '이상 없음', 'retest': '재시험 요청', 'hold': '출하 보류'}
DEFECT_TYPES = (
    ('capacity', '용량불량', '시점 간 최대 전압 변화', 'mV'),
    ('weld', '용접불량', '팩 평균 대비 최대 낙폭', 'mV'),
    ('wire', '센서와이어불량', '모듈 내 이웃 셀 최대 차이', 'mV'),
    ('sensor', '센서불량', '온도 중앙값 대비 최대 이탈', '°C'),
)


def defect_cards(data):
    """Display stored pack flags; never infer a defect from the selected cell."""
    row = data['summary']
    flags, metrics = row.get('flags') or {}, row.get('metrics') or {}
    thresholds = data.get('defectThresholds') or {}
    cards = []
    for key, name, description, unit in DEFECT_TYPES:
        flagged = flags.get(key)
        state = 'suspected' if flagged is True else 'clear' if flagged is False else 'unavailable'
        label = '의심' if flagged is True else '해당 없음' if flagged is False else '자료 없음'
        cards.append(html.Li([
            html.Div([html.H3(name), badge(label, 'danger' if flagged is True else 'neutral')], className='pack-defect-heading'),
            html.P(description, className='pack-defect-description'),
            html.Div([html.Strong(num(metrics.get(key), 2)), html.Span(' / '+num(thresholds.get(key), 2)+' '+unit)],
                     className='pack-defect-measure'),
        ], className='pack-defect-card '+state))
    return html.Section([
        html.Div([html.H2('4대 불량 의심 유형', id='pack-defect-title'),
                  html.Span('팩 전체 · 측정값 / 임계값')], className='pack-defect-title'),
        html.Ul(cards, className='pack-defect-grid'),
    ], className='pack-defect-section', **{'aria-labelledby': 'pack-defect-title'})


def controls():
    return html.Div([
        dcc.Store(id='pack-index'),
        dcc.Store(id='pack-selected-cell', data='M01CV01'),
        select('pack-number', '팩 번호 검색', [], '1000'),
        select('pack-mode', '공정', [{'label': '충전', 'value': 'chg'}, {'label': '방전', 'value': 'dchg'}], 'dchg'),
        select('pack-snapshot', '셀 조회 시점', [{'label': '측정 종료', 'value': 'last'}, {'label': '최대 이상 점수', 'value': 'peak'}], 'last'),
        dmc.SegmentedControl(id='pack-view', value='inspection', data=[{'label': '셀 검사', 'value': 'inspection'}, {'label': '자동 분석 결과', 'value': 'list'}]),
    ], id='pack-controls', className='filter-toolbar pack-controls', style={'display': 'none'})


def unresolved_defect(row):
    return row.get('ai_verdict') == 'NG' and not any(
        (row.get('flags') or {}).get(key) is True for key, *_ in DEFECT_TYPES)


def technician_notice(row, action=False):
    if not unresolved_defect(row):
        return None
    return html.Div([
        callout('불량 의심 유형 미확정 · 숙련 기술자 확인 필요',
                'AI 이상이 감지되었습니다. 판정·조치에서 숙련 기술자가 원인을 확인하고 후속 조치를 기록해 주세요.', 'warning'),
        dmc.Button('판정·조치로 이동', id='pack-go-review', variant='outline') if action else None,
    ], className='pack-review-notice', role='status')


def review_page(data, request):
    row = data['summary']
    try:
        records = request(f"/api/battery-packs/{row['pack_id']}/history")['records']
        latest = records[0] if records else {}
        unavailable = False
        history = [html.Article([
            badge(DECISIONS.get(r['decision'], r['decision'])), html.P(r.get('note') or '메모 없음'),
            html.Small(f"{r.get('actor', '')} · {r.get('at', '')} · 수정 {r.get('revision', '')}"),
            html.Small('이전 판정: '+DECISIONS.get(r.get('previousDecision'), '없음')),
        ], className='record-item') for r in records] or html.P('저장된 작업자 기록이 없습니다.')
    except (ValueError, httpx.HTTPError) as exc:
        records, latest, unavailable = [], {}, True
        history = callout('기록 조회 불가', str(exc) if isinstance(exc, ValueError) else '연결을 확인하세요.', 'warning')
    names = [name for key, name, *_ in DEFECT_TYPES if (row.get('flags') or {}).get(key) is True]
    form = panel('작업자 판정', [
        dmc.Select(id='pack-decision', label='최종 판정', data=[{'label':v,'value':k} for k,v in DECISIONS.items()],
                   value=latest.get('decision','retest'), allowDeselect=False, persistence=row['pack_id'], persistence_type='memory'),
        dmc.Textarea(id='pack-note', label='검토 메모', value=latest.get('note',''), minRows=4,
                     placeholder='확인한 원인과 후속 조치를 입력하세요.', inputProps={'maxLength':2000},
                     persistence=row['pack_id'], persistence_type='memory'),
        dmc.Button('저장 내용 확인', id='pack-prepare', className='spaced-button', disabled=unavailable),
        html.P('확인 후 저장하면 작성자와 시각이 함께 기록됩니다.', className='section-note'),
    ])
    evidence = panel(f"팩 {row['pack_no']} · {row['mode']} 판정 근거", [
        metric_rows([('AI 품질 판정', row['ai_verdict'], '팩 전체'),
                     ('이상 시점 비율', num(row['anomalyPercent'],1)+' %', '기준 5%'),
                     ('불량 의심 유형', ' · '.join(names) or ('미확정' if row['ai_verdict']=='NG' else '해당 없음'), '숙련 기술자 확인 후 최종 판정'),
                     ('작업자 판정', '조회 불가' if unavailable else DECISIONS.get(latest.get('decision'),'미확정'), '최근 저장 기록')]),
    ])
    return html.Div([technician_notice(row), html.Div([form,evidence],className='grid-equal'),
                     panel('작업자 판정 이력',history)],className='pack-workspace pack-review-workspace'), records


def result_rows(index):
    return [{'id': r['pack_id'], '팩 번호': r['pack_no'], '공정': r['mode'], 'AI 판정': r['ai_verdict'],
             '이상 비율 (%)': r['anomalyPercent'], '전압 편차 (mV)': r.get('dv_mv'),
             '온도 편차 (°C)': r.get('temp_dev')} for r in index['rows']]


def results(index):
    rows = result_rows(index)
    columns = [{'field': k, 'headerName': k, 'filter': 'agNumberColumnFilter' if k.endswith(')') else 'agTextColumnFilter'} for k in rows[0] if k != 'id'] if rows else []
    for col in columns:
        if col['field'] == 'AI 판정':
            col['cellClassRules'] = {'pack-ng': "params.value === 'NG'", 'pack-ok': "params.value === 'OK'"}
        if col['field'].endswith(')'):
            col['valueFormatter'] = {'function': "params.value == null ? '—' : Number(params.value).toLocaleString('ko-KR', {maximumFractionDigits: 2})"}
    return html.Div([
        html.Div([kpi('고유 팩', len(index['packNumbers']), '충전·방전 결과'), kpi('자동 분석 결과', index['count'], '팩·공정별 집계'),
                  kpi('AI NG', index['ngCount'], '우선 검토 대상', 'danger'), kpi('AI OK', sum(r['AI 판정']=='OK' for r in rows), '분석 결과')], className='kpi-strip'),
        panel('자동 분석 결과', [
            html.Div([dmc.TextInput(id='pack-search', placeholder='팩 번호 검색', **{'aria-label':'결과 목록 팩 번호 검색'}),
                      dmc.SegmentedControl(id='pack-verdict-filter', value='all', data=[{'label':'전체','value':'all'},{'label':'AI NG','value':'NG'}])], className='pack-list-tools'),
            grid(rows, 'pack-results-grid', columns, height=454)], subtitle='행을 선택하면 해당 팩·공정의 셀 검사로 이동합니다.')
    ], className='pack-workspace'), rows


def score_chart(data):
    series = data['series']
    t, phi = series.get('t', []), series.get('phi', [])
    figure = go.Figure()
    if len(t) == len(phi):
        figure.add_trace(go.Scatter(x=t, y=phi, mode='lines', name='이상 점수 φ', line={'color': BLUE, 'width': 1.5}))
        preds = series.get('pred', [])
        flagged = [(x,y) for x,y,p in zip(t,phi,preds) if p]
        if flagged:
            figure.add_trace(go.Scatter(x=[p[0] for p in flagged], y=[p[1] for p in flagged], mode='markers', name='이상 시점', marker={'color': RED, 'size': 4}))
        figure.add_hline(y=2, line_dash='dash', line_color=spatial.STATUS_COLORS['warning'])
    figure.update_layout(height=245, margin={'l':45,'r':15,'t':20,'b':45}, paper_bgcolor='white', plot_bgcolor='white',
                          font={'family':'SUIT Variable','color':BLUE,'size':13}, legend={'orientation':'h','y':1.17},
                          xaxis={'title':'측정 행','gridcolor':'#E7EAF0'}, yaxis={'title':'φ','gridcolor':'#E7EAF0'})
    return figure


def inspection(data, selected, actions):
    row, cells = data['summary'], data['snapshot']['cells']
    chosen = next((c for c in cells if c['id'] == selected), cells[0] if cells else None)
    data['selectedCell'] = chosen['id'] if chosen else None
    data['statusColors'] = True
    decision = actions.get('quality:'+data['testId'], {})
    mismatch = row.get('rule_verdict') in ('OK','NG') and row['ai_verdict'] != row['rule_verdict']
    counts = html.Div([html.Span([html.I(className='pack-dot '+key), f'{label} {data["counts"][key]}'], title={'normal':'|z| < 2','warning':'2 ≤ |z| < 3','danger':'|z| ≥ 3','missing':'측정값 또는 상대 편차 없음'}[key]) for key,label in STATUS.items() if key != 'missing' or data['counts'][key]], className='pack-legend')
    figure = spatial.quality_figure(data) if cells else None
    detail = html.Div([
        select('pack-cell', '선택 셀', [c['id'] for c in cells], data['selectedCell']),
        badge(STATUS[chosen['status']], {'normal':'success','warning':'warning','danger':'danger','missing':'neutral'}[chosen['status']]) if chosen else None,
        metric_rows([('셀 전압',num(chosen['value'],3)+' V',''), ('상대 편차',num(abs(chosen['z']) if chosen['z'] is not None else None,2)+' σ','선택 시점의 셀 간 비교')]) if chosen else None,
        html.P('셀 상태는 상대 편차이며 불량 확정이 아닙니다.', className='section-note'),
        html.Div([html.Small('작업자 판정'), badge('조회 불가' if actions.get('__unavailable__') else DECISIONS.get(decision.get('decision'), '미확정'))], className='pack-decision-line'),
        dmc.Button('판정·조치', id='pack-open-review', variant='outline', fullWidth=True),
    ], className='pack-cell-detail')
    evidence = html.Details([
        html.Summary(['판정 상세 근거', badge('판정 차이 있음','warning') if mismatch else None]),
        html.Div([
            metric_rows([('전압 편차',num(row.get('dv_mv'),2)+' mV','공정 임계값 '+num(row.get('thr_dv_mv'),2)+' mV'),
                         ('온도 편차',num(row.get('temp_dev'),2)+' °C','공정 임계값 '+num(row.get('thr_temp'),2)+' °C')]),
            metric_rows([('AI / 규칙 판정',row['ai_verdict']+' / '+str(row.get('rule_verdict') or '미확인'),'서로 다른 판정 기준'),
                         ('최대 이상 점수 φ',num(row.get('max_phi'),2),f"전체 {num(row.get('n_rows'))}개 측정 행")]),
        ], className='pack-evidence-metrics'),
        html.P('팩 AI 판정: 이상 시점 비율 5% 기준 · 규칙 판정: 전압·온도 편차의 공정 임계값 비교',className='section-note'),
        graph(score_chart(data)),
        html.P(f"점수 추이 {len(data['series'].get('t',[])):,}점 표시 · 최대값과 이상 비율은 전체 측정 기준",className='section-note'),
        html.Details([html.Summary('T² / SPE 수치'),grid([
            {'측정 행': t, 'T²': t2, 'SPE': spe} for t,t2,spe in zip(data['series'].get('t',[]),data['series'].get('t2',[]),data['series'].get('spe',[]))], 'pack-score-grid', height=280),
            html.P(f"T² 관리한계 {num(data['series'].get('ucl_t2'),3)} · SPE 관리한계 {num(data['series'].get('ucl_spe'),3)}",className='section-note')])
    ], className='pack-evidence')
    return html.Div([
        html.Div([kpi('AI 품질 판정', row['ai_verdict'], '검토 필요' if row['ai_verdict']=='NG' else '이상 비율 기준 이내' if row['ai_verdict']=='OK' else '결과 확인 필요', 'danger' if row['ai_verdict']=='NG' else ''),
                  kpi('이상 시점 비율', num(row['anomalyPercent'],1), '팩 전체 · 기준 5%', unit='%'),
                  kpi('전압 편차', num(row.get('dv_mv'),1), '측정 종료 기준', unit='mV'),
                  kpi('온도 편차', num(row.get('temp_dev'),2), '측정 종료 기준', unit='°C')], className='kpi-strip'),
        defect_cards(data),
        technician_notice(row, action=True),
        panel(f"팩 {row['pack_no']} · {row['mode']} 셀 검사", [counts, html.Div([
            spatial.graph3d(figure,'pack-3d') if figure else callout('셀 자료 없음','이 팩의 선택 시점에 조회할 셀 자료가 없습니다.','warning'), detail], className='pack-inspection-grid')],
            subtitle=('측정 종료' if data['snapshot']['kind']=='last' else '최대 이상 점수')+f" · 측정 행 {num(data['snapshot']['t'])} · 16모듈 / 176셀",
            action=badge('판정 차이 있음','warning') if mismatch else None), evidence
    ], className='pack-workspace')


def register(app, request):
    @app.callback(Output('pack-controls','style'), Output('pack-index','data'), Output('pack-number','data'),
                  Input('url','pathname'), Input('tabs','value'), Input('refresh','n_clicks'), Input('url','search'))
    def load(path, tab, _, search):
        if path != '/quality' or tab not in ('packs','review') or (tab=='review' and legacy_quality_review(search)):
            return {'display':'none'}, no_update, no_update
        try:
            index = request('/api/battery-packs', params={'refresh':ctx.triggered_id=='refresh'})
            return {}, index, index['packNumbers']
        except (ValueError,httpx.HTTPError) as exc:
            return {}, {'error':str(exc) if isinstance(exc,ValueError) else '연결을 확인하세요.'}, []

    @app.callback(Output('pack-number','value'), Output('pack-mode','value'), Output('pack-mode','data'), Output('pack-view','value'),
                  Input('pack-results-grid','selectedRows',allow_optional=True), Input('pack-number','value'), Input('pack-index','data'), State('pack-mode','value'), prevent_initial_call=True)
    def choose(rows, number, index, mode):
        if not index or not index.get('rows'):
            return no_update,no_update,no_update,no_update
        view = no_update
        if ctx.triggered_id == 'pack-results-grid' and rows:
            selected = next((r for r in index['rows'] if r['pack_id']==rows[0].get('id')),None)
            if selected:
                number, mode, view = selected['pack_no'], selected['process'], 'inspection'
        if number not in index['packNumbers']:
            number = index['packNumbers'][0]
        modes = [r['process'] for r in index['rows'] if r['pack_no']==number]
        mode = mode if mode in modes else modes[0]
        return number, mode, [{'value':m,'label':'충전' if m=='chg' else '방전'} for m in modes], view

    @app.callback(Output('pack-results-grid','rowData'),Input('pack-search','value',allow_optional=True),Input('pack-verdict-filter','value',allow_optional=True),State('pack-index','data'),prevent_initial_call=True)
    def filter_rows(search, verdict, index):
        return [r for r in result_rows(index) if (search or '').strip() in r['팩 번호'] and (verdict in (None,'all') or r['AI 판정']==verdict)] if index and index.get('rows') else []

    @app.callback(Output('pack-selected-cell','data'),Input('pack-3d','clickData',allow_optional=True),Input('pack-cell','value',allow_optional=True),State('view-data','data'),State('pack-selected-cell','data'),prevent_initial_call=True)
    def select_cell(click, selected, view, current):
        if not view or view.get('tab') != 'packs':
            return no_update
        allowed = {c['id'] for c in view['payload'].get('snapshot',{}).get('cells',[])}
        chosen = spatial.selection(click,allowed) if ctx.triggered_id=='pack-3d' else selected if selected in allowed else None
        return chosen if chosen and chosen != current else no_update

    @app.callback(Output('tabs','value',allow_duplicate=True),Output('url','search',allow_duplicate=True),
                  Input('pack-open-review','n_clicks',allow_optional=True),
                  Input('pack-go-review','n_clicks',allow_optional=True),prevent_initial_call=True)
    def open_review(click, notice_click):
        return ('review','?tab=review') if click or notice_click else (no_update,no_update)

    @app.callback(Output('pack-snapshot','style'),Output('pack-view','style'),Input('tabs','value'))
    def inspection_controls(tab):
        style = {'display':'none'} if tab=='review' else {}
        return style, style

    @app.callback(Output('confirm-modal','opened',allow_duplicate=True),Output('confirm-content','children',allow_duplicate=True),Output('pending-action','data',allow_duplicate=True),
                  Input('pack-prepare','n_clicks',allow_optional=True),State('pack-decision','value',allow_optional=True),State('pack-note','value',allow_optional=True),State('view-data','data'),prevent_initial_call=True)
    def prepare(click, decision, note, view):
        if not click or not view or view.get('tab')!='review' or not view.get('payload',{}).get('summary'):
            return no_update,no_update,no_update
        try:
            pending = {'track':'quality','target':view['payload']['testId'],'decision':decision,'note':note or '', 'dataVersion':view['dataVersion']}
            validated = request('/api/preview/decision','POST',json=pending)
            previous = request('/api/records/state')['records'].get(validated['key'],{})
            pending.update(requestId=uuid4().hex,expectedRevision=previous.get('revision',0))
            return True,html.Div([html.H3(pending['target'].removeprefix('pack:')),html.P(DECISIONS[decision]),html.P(note or '메모 없음')]),pending
        except (ValueError,httpx.HTTPError):
            return True,callout('저장 준비 실패','새로고침 후 팩과 기록을 다시 확인하세요.','warning'),None
