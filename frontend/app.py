from __future__ import annotations

import base64
import os
from pathlib import Path
from uuid import uuid4

import httpx
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update
import dash_mantine_components as dmc
from flask import request as flask_request
from frontend import copilot_ui, quality_workspace, maintenance_workspace, overview_workspace, spatial, pack_workspace
from frontend.navigation import navigation_context
from frontend.palette import BLUE, BLUE_SHADES

from frontend.components import badge, callout, graph, grid, icon, metric_rows, num, panel, select, settings_disclosure
from frontend.charts import demand_chart, maintenance_chart
from frontend.views import DOMAIN, TABS, conclusion_view, demand_view, maintenance_view, overview, project_view, quality_view, validation_view

API = os.environ.get("MANUFACTURING_API_URL", "http://127.0.0.1:8070")


def request(path, method="GET", **kwargs):
    headers = {'Cookie': flask_request.headers.get('Cookie', ''), 'Origin': API,
               'X-CSRF-Token': flask_request.cookies.get('manufacturing_csrf', '')}
    with httpx.Client(base_url=API, timeout=90, trust_env=False, headers=headers) as client:
        response = client.request(method, path, **kwargs)
    if not response.is_success:
        try:
            detail = response.json().get("detail", "데이터를 읽을 수 없습니다.")
            if not isinstance(detail, str):
                detail = "입력 조건을 확인하세요."
        except ValueError:
            detail = "백엔드 응답을 확인할 수 없습니다."
        raise ValueError(detail)
    return response.json()


def domain(path):
    key = (path or "/").strip("/") or "overview"
    return key if key in DOMAIN else "overview"


def create_dashboard(meta):
    app = Dash(__name__, assets_folder=str(Path(__file__).parent / "assets"), suppress_callback_exceptions=True,
               title="BatteryFlow AI 운영센터", update_title="분석 중…")
    shades = BLUE_SHADES
    app.layout = dmc.MantineProvider(theme={"primaryColor": "lsblue", "colors": {"lsblue": shades}, "fontFamily": "'SUIT Variable', sans-serif", "defaultRadius": "sm"}, children=[
        dcc.Store(id='sidebar-collapsed', storage_type='local', data=False),
        dcc.Location(id="url", refresh='callback-nav'), dcc.Store(id="view-data"), dcc.Store(id='linked-event'), dcc.Store(id="q-defect", data="capacity"), dcc.Download(id="quality-history-download"), dcc.Store(id="export-data"),
        dcc.Store(id="session-actions", data={}), dcc.Store(id="selected-target"), dcc.Store(id="pending-action"),
        dmc.Modal(id="export-modal", title="CSV 내보내기", centered=True, closeButtonProps={"aria-label": "닫기"}, children=html.Div(id="export-ready")),
        html.Div([
            html.Aside([
                html.Button([icon('panel-left-close'), html.Span('접기', className='sidebar-toggle-label')], id='sidebar-toggle', className='sidebar-toggle', title='사이드바 접기', **{'aria-label':'사이드바 접기', 'aria-expanded':'true'}),
                dcc.Link([html.Div(html.Img(src='/assets/ci_img20.png', alt='LS', className='brand-logo'), className='brand-mark'), html.Strong(["BatteryFlow AI", html.Span("운영센터", className='brand-center')])], href="/", className="brand", title='BatteryFlow AI 운영센터 홈'),
                html.Nav([dcc.Link([icon(glyph,20), html.Span(label)], href="/" if key == "overview" else f"/{key}", id=f"nav-{key}", className="nav-link", title=label) for key, label, glyph in [("overview", "통합 현황", "layout-dashboard"), ("demand", "공급망 예측", "chart-no-axes-combined"), ("maintenance", "예지보전", "activity"), ("quality", "품질 보증", "shield-check")]], **{'aria-label':'주요 업무'}),
                html.Div(className="sidebar-spacer"),
                html.Button([icon("message-square", 20), html.Div(html.Strong("AI Copilot")), icon("chevron-right", 16)], id="open-copilot", className="copilot-entry", title='AI Copilot 열기', **{'aria-label':'AI Copilot 열기', 'data-tooltip':'AI Copilot'}),
                html.Div([html.Span("L", className="user-avatar"), html.Div(id="signed-user"), html.Button([icon('log-out',16),html.Span('로그아웃')], id='logout-button', type='button', title='로그아웃', **{'aria-label': '로그아웃', 'data-tooltip':'로그아웃'})], className="sidebar-user")], className="app-sidebar"),
            html.Main([
                html.Div([html.Div([html.Span("BatteryFlow AI"), icon("chevron-right", 14), html.Strong(id="breadcrumb")], className="breadcrumb"),
                          html.Div([badge("저장 검증" if meta.get('verification') else "로컬 체험" if meta.get('demo') else "직원 전용", "success"), html.Button('업무 기록', id='open-records', className='utility-button'), html.Span("공식 분석 자료", title="자료 버전: "+meta["dataVersion"])], className="utility-right")], className="utility-bar"),
                html.Div([
                    html.Div([html.Div([html.H1(id="page-title")]), html.Div([html.Details([html.Summary("자료 작업"),dmc.Button("CSV로 재예측", id="open-inference", leftSection=icon("upload", 16), variant="light", className="filter-button"),dmc.Button('CSV 검사·분석',id='open-maintenance-upload',variant='outline',className='filter-button'),dmc.Button('보고서',id='open-maintenance-report',variant='default',className='filter-button'),dmc.Button("CSV 검사·분석", id="open-quality-upload", variant="outline"),dmc.Button("CSV 내보내기", id="export", variant="outline", leftSection=icon("download", 15), size="sm")],id='file-actions',className='file-actions'), dmc.Button("검토·기록", id="go-review", leftSection=icon("check-check", 15)), dmc.Button("새로고침", id="refresh", variant="default", leftSection=icon("refresh-cw", 15), size="sm")], className="page-actions")], className="page-heading"),
                    dmc.Tabs(id="tabs", value="summary", children=[], className="domain-tabs"),
                    html.Div([
                        pack_workspace.controls(),
                        html.Div([select("d-date", "목표일", meta["dates"], meta["dates"][-1]), select("d-part", "부품", [{"label": "전체 부품", "value": "ALL"}]+[{"label": p, "value": p} for p in meta["parts"]], "ALL"),
                                  settings_disclosure('분석 설정',[select("d-model", "예측 모델", [{"label": name + (" · 운영 기본" if name == meta["demandPrimary"] else " · 학습형 보조" if name == meta["demandAuxiliary"] else " · 비교"), "value": name} for name in meta["demandModels"]], meta["demandPrimary"])]),
                                  ], id="demand-controls", className="filter-toolbar"),
                        html.Div([select("m-run", "시험 파일", meta["runs"], "WeldingTest_04_NG"), settings_disclosure('분석 설정',[select("m-sup", "지도 모델", meta["supervised"], meta["defaultSupervised"]), select("m-unsup", "비지도 모델", meta["unsupervised"], meta["defaultUnsupervised"])])], id="maintenance-controls", className="filter-toolbar"),
                        html.Div([select("q-test", "시험 ID", meta["tests"], "Test07_NG_dchg"), select("q-cell", "선택 셀", [f"M{m:02d}CV{c:02d}" for m in range(1, 17) for c in range(1, 12)], "M02CV01"),
                                  html.Div([html.Label("조회 시점 (%)"), dmc.Slider(id="q-progress", value=100, min=1, max=100, step=1, marks=[{"value": 10, "label": "10%"}, {"value": 100, "label": "100%"}])], className="progress-control"),
                                  settings_disclosure('표시 설정',[select("q-map", "위치 지도", [{"label": "셀 전압 16×11", "value": "cell"}, {"label": "모듈 온도 16×2", "value": "temperature"}], "cell"), select("q-basis", "표시 기준", [{"label": "보정값", "value": "clean"}, {"label": "원본값", "value": "raw"}], "clean")]),html.Div([dmc.SegmentedControl(id="q-section", value="cell", data=[{"label":"셀 검사", "value":"cell"},{"label":"이상 구간", "value":"anomaly"}], persistence=True, persistence_type="session")], id="quality-analysis-local", className="quality-local")], id="quality-controls", className="filter-toolbar"),

                        html.Div([select("q-eval-scope", "평가 범위", [{"label":"선택 시험", "value":"selected"},{"label":"잠금 시험 전체", "value":"all"}], "selected"), html.Span("시점 단위 평가 · 조회 시점 슬라이더와 별도로 시험 전체를 평가합니다.", className="section-note")], id="quality-eval-local", className="filter-toolbar compact"),
                        html.Div([select("q-history-scope", "이력 조회 범위", [{"label":"선택 시험", "value":"selected"},{"label":"전체 품질 · 최근 업무 100건 내", "value":"all"}], "selected")], id="quality-history-local", className="filter-toolbar compact"),
                        html.Div([select("overview-track", "분석 영역", [{"label": DOMAIN[t], "value": t} for t in ["demand", "maintenance", "quality"]], "demand"),
                                  select("result-kind", "결과 유형", [{"label": v, "value": k} for k, v in [("prediction", "예측 결과"), ("evaluation", "평가 결과"), ("raw", "원본 측정·이력"), ("processed", "가공 데이터")]], "prediction")], id="overview-controls", className="filter-toolbar"),
                        html.Div([dmc.TextInput(id="part-search", label="부품 검색", placeholder="Part 번호 검색", leftSection=icon("search", 15), className="context-select"), select("direction-filter", "검토 방향", [{"label": "전체", "value": "all"}, {"label": "상향 검토", "value": "up"}, {"label": "하향 검토", "value": "down"}, {"label": "미확인", "value": "open"}], "all")], id="demand-local", className="filter-toolbar compact"),
                        html.Div([select("event-status", "이벤트 상태", [{"label": "전체", "value": "all"}, {"label": "미확인", "value": "open"}, {"label": "확인 완료", "value": "done"}], "all")], id="maintenance-local", className="filter-toolbar compact"),
                    ]),
                    html.Div(id="save-feedback", role="status"),
                    dcc.Loading(html.Div(id="page-content"), type="circle", color=BLUE, delay_show=300),
                    html.Footer([html.Span("KAMP 제조 데이터 기반 · 서로 다른 트랙의 원본은 병합하지 않습니다."), html.Span("로컬 체험 · 외부 서비스 미연결" if meta.get('demo') else "공식 분석 연결 확인 중", id='analysis-source')], className="footer")
                ], className="dashboard-container")], className="app-main")], className="app-shell", id='app-shell'),
        dmc.Drawer(closeButtonProps={"aria-label": "닫기"}, id="detail-drawer", title="분석 상세", position="right", size=560, children=[html.Div(id="detail-content"),
            html.Div([dmc.Textarea(id="review-note", label="검토 메모", placeholder="확인한 내용을 입력하세요.", minRows=3, inputProps={"maxLength": 2000}), dmc.Button("확인 처리", id="prepare-review", leftSection=icon("check"), className="spaced-button")], id="review-form")]),
        dmc.Modal(closeButtonProps={"aria-label": "닫기"}, id="confirm-modal", title="업무 기록 저장 확인", centered=True, children=[html.Div(id="confirm-content"),
            callout("직원 공용 업무 기록", "작성자와 시각을 포함해 Firebase에 저장합니다. 이전 기록도 이력에 남습니다."), dmc.Group([dmc.Button("취소", id="cancel-save", variant="default"), dmc.Button("확인 후 저장", id="confirm-save")], justify="flex-end")]),
        copilot_ui.drawer(), quality_workspace.upload_modal(), maintenance_workspace.modals(), pack_workspace.drawer(),
        dmc.Drawer(id='records-drawer', title='업무 기록', position='right', size=600, closeButtonProps={'aria-label': '닫기'}, children=[html.P('직원 공용 · 최신 100건 · 수정 전 판정도 이력으로 보존됩니다.', className='section-note'), dcc.Loading(html.Div(id='records-content'))]),
        dmc.Modal(closeButtonProps={"aria-label": "닫기"}, id="inference-modal", title="CSV로 D+3 재예측", centered=True, size="xl", children=[
            html.P("동일 부품의 3~60일 자료를 업로드하세요. 최근 3일은 연속이어야 합니다. 8일 이력을 권장하며 저장된 모델로 예측합니다."),
            html.A("입력 템플릿 다운로드", href="/api/demand/template", className="template-link"),
            dcc.Upload(id="csv-upload", children=html.Div([icon("upload", 24), html.Strong("CSV 파일 선택 또는 여기에 끌어놓기"), html.Small("UTF-8 · 최대 1MB · 3~60행")]), accept=".csv", max_size=1000000, className="upload-zone"),
            dcc.Loading(html.Div(id="inference-result"), type="circle")])
    ])

    app.clientside_callback("""function(n, collapsed) {
        return n ? !collapsed : window.dash_clientside.no_update;
    }""", Output('sidebar-collapsed','data'), Input('sidebar-toggle','n_clicks'), State('sidebar-collapsed','data'), prevent_initial_call=True)
    app.clientside_callback("""function(collapsed) {
        window.setTimeout(function(){ window.dispatchEvent(new Event('resize')); }, 240);
        return [collapsed ? 'app-shell sidebar-collapsed' : 'app-shell',
                collapsed ? '사이드바 펼치기' : '사이드바 접기', collapsed ? 'false' : 'true',
                collapsed ? '사이드바 펼치기' : '사이드바 접기'];
    }""", Output('app-shell','className'), Output('sidebar-toggle','title'), Output('sidebar-toggle','aria-expanded'), Output('sidebar-toggle','aria-label'), Input('sidebar-collapsed','data'))
    copilot_ui.register(app, request)
    quality_workspace.register(app, request)
    pack_workspace.register(app, request)
    maintenance_workspace.register(app, request)
    overview_workspace.register(app)
    from frontend import demand_workspace
    demand_workspace.register(app)

    @app.callback(Output('signed-user', 'children'), Input('url', 'pathname'))
    def signed_user(_):
        try:
            user = request('/api/me')
            return [html.Strong(user['name']), html.Small('분석 체험' if meta.get('demo') else '관리자' if user['role'] == 'admin' else '직원')]
        except (ValueError, httpx.HTTPError):
            return html.A('다시 로그인', href='/login')

    @app.callback(Output('records-drawer', 'opened'), Output('records-content', 'children'), Input('open-records', 'n_clicks'), prevent_initial_call=True)
    def history(_):
        try:
            rows = request('/api/records')['records']
            labels = {'reviewed': '확인 완료', 'clear': '이상 없음', 'retest': '재시험 요청', 'hold': '출하 보류'}
            return True, [html.Article([html.Strong(row['target']), badge(labels[row['decision']]), html.P(row.get('note') or '메모 없음'),
                html.Small(f"{row['actor']} · {row['at'][:19].replace('T', ' ')} UTC · 이력 {row['revision']}"),
                html.Small(f"{row.get('date') or ''} {row.get('model') or ''} · {row['dataVersion']}")], className='record-item') for row in rows] or callout('저장된 기록이 없습니다', '분석 화면에서 검토 내용을 남겨 주세요.')
        except (ValueError, httpx.HTTPError):
            return True, callout('이력을 불러오지 못했습니다', '연결을 확인하고 다시 열어 주세요.', 'warning')

    @app.callback(Output("tabs", "children"), Output("tabs", "value"), Output("page-title", "children"), Output("breadcrumb", "children"),
                  *[Output(f"nav-{key}", "className") for key in DOMAIN],
                  *[Output(id,'value',allow_duplicate=id in ['q-test','q-cell']) for id in ['d-date','d-part','d-model','m-run','m-sup','m-unsup','q-test','q-cell']],
                  Output('linked-event','data'),Output('part-search','value'),Output('direction-filter','value'),Output('event-status','value'),
                  Output('q-history-scope','value'),
                  Input("url", "pathname"),Input('url','search'),prevent_initial_call='initial_duplicate')
    def route(path,search):
        key = domain(path)
        context = navigation_context(key,search,meta)
        tab = context['tab'] if context['tab'] in dict(TABS[key]) else TABS[key][0][0]
        if key == 'quality' and context.get('q-test') and not context['tab']:
            tab = 'analysis'
        title = "D+3 발주량 예측" if key == "demand" else DOMAIN[key]
        return ([dmc.TabsList([dmc.TabsTab(label, value=value) for value, label in TABS[key]])], tab, title, DOMAIN[key],
                *["nav-link active" if k == key else "nav-link" for k in DOMAIN],
                *[context.get(id,no_update) for id in ['d-date','d-part','d-model','m-run','m-sup','m-unsup','q-test','q-cell']],
                context['event'],'' if context['resetLocalFilters'] else no_update,
                'all' if context['resetLocalFilters'] else no_update,'all' if context['resetLocalFilters'] else no_update,
                'selected' if key=='quality' and context['resetLocalFilters'] else no_update)

    @app.callback(*[Output(id, "style") for id in ["demand-controls", "maintenance-controls", "quality-controls", "overview-controls", "demand-local", "maintenance-local", "result-kind", "quality-analysis-local", "quality-eval-local", "quality-history-local"]], Input("url", "pathname"), Input("tabs", "value"), Input("q-section", "value"))
    def control_visibility(path, tab, quality_section):
        key = domain(path)
        flags = [key == "demand", key == "maintenance", key == "quality" and tab != 'packs', key == "overview" and tab in ["data", "results"], key == "demand" and tab == "review", key == "maintenance" and tab == "review", key == "overview" and tab == "results", key == "quality" and tab == "analysis", False, key == "quality" and tab == "review"]
        return [{} if show else {"display": "none"} for show in flags]

    @app.callback(Output("page-content", "children"), Output("view-data", "data"), Output("export-data", "data"), Output('analysis-source', 'children'),
        Input("url", "pathname"), Input("tabs", "value"), Input("d-date", "value"), Input("d-part", "value"), Input("d-model", "value"),
        Input("m-run", "value"), Input("m-sup", "value"), Input("m-unsup", "value"), Input("q-test", "value"), Input("q-cell", "value"), Input("q-progress", "value"), Input("q-map", "value"), Input("q-basis", "value"), Input("q-defect", "data"), Input("q-section", "value"), Input("q-eval-scope", "value"), Input("q-history-scope", "value"),
        Input("overview-track", "value"), Input("result-kind", "value"), Input("session-actions", "data"), Input("part-search", "value"), Input("direction-filter", "value"), Input("event-status", "value"), Input("refresh", "n_clicks"),Input('linked-event','data'),
        Input('pack-index','data'),Input('pack-number','value'),Input('pack-mode','value'),Input('pack-snapshot','value'),Input('pack-view','value'),Input('pack-selected-cell','data'))
    def render(path, tab, date, part, model, run, sup, unsup, test, cell, progress, mapmode, basis, defect, quality_section, eval_scope, history_scope, track, kind, actions, search, direction, status, _, linked_event, pack_index, pack_number, pack_mode, pack_snapshot, pack_view, pack_cell):
        key = domain(path)
        tab = tab if tab in dict(TABS[key]) else TABS[key][0][0]
        actions = actions or {}
        try:
            if key == 'quality' and tab == 'packs':
                if not pack_index:
                    return html.P('팩 분석 결과를 불러오는 중…'), {'domain':key,'tab':tab,'payload':{}}, [], ''
                if pack_index.get('error'):
                    raise ValueError(pack_index['error'])
                if not pack_index['rows']:
                    return callout('팩 결과 없음','조회할 자동 분석 결과가 없습니다.'), {'domain':key,'tab':tab,'payload':{}}, [], ''
                rows = pack_workspace.result_rows(pack_index)
                if pack_view == 'list':
                    content, rows = pack_workspace.results(pack_index)
                    data = {'scope':'pack-list'}
                else:
                    choices = [r for r in pack_index['rows'] if r['pack_no']==pack_number]
                    chosen = next((r for r in choices if r['process']==pack_mode),choices[0] if choices else pack_index['rows'][0])
                    data = request('/api/battery-packs/'+chosen['pack_id'],params={'snapshot':pack_snapshot,'refresh':ctx.triggered_id=='refresh'})
                    try:
                        actions = request('/api/records/state')['records']
                    except (ValueError,httpx.HTTPError):
                        actions = {'__unavailable__':True}
                    content = pack_workspace.inspection(data,pack_cell,actions)
                return content, {'domain':key,'tab':tab,'payload':data,'dataVersion':data.get('dataVersion',pack_index['dataVersion'])}, rows, ''
            current_meta = request('/api/analysis/refresh', 'POST') if ctx.triggered_id == 'refresh' else request('/api/meta')
            source = current_meta.get('analysisSource') or {}
            source_hint = html.Span()
            record_warning = None
            try:
                actions = request('/api/records/state')['records']
            except (ValueError, httpx.HTTPError):
                actions = {'__unavailable__': True}
                record_warning = (callout('로컬 체험 모드', '분석·필터·CSV 추론을 체험할 수 있습니다. 업무 기록 저장·조회와 Copilot은 비활성화되어 있습니다.') if meta.get('demo') else
                    callout('공유 기록 조회 불가', 'Firestore 연결 또는 무료 할당량을 확인하세요. 아래 판정·확인 상태는 최신 저장 상태를 반영하지 않습니다.', 'warning'))
            if key == "overview":
                data = request("/api/overview")
                data['analysisSource'] = source
                content = overview(data, current_meta, actions)
                from backend.overview import work_status
                rows,_ = work_status(data,actions)
                data['workRows'] = rows
            elif key == "demand":
                data = request("/api/demand", params={"date": date, "part": part, "model": model})
                content, rows = demand_view(data, tab, actions, search if tab == "review" else None, direction if tab == "review" else "all")
            elif key == "maintenance":
                data = request("/api/maintenance", params={"run": run, "supervised": sup, "unsupervised": unsup})
                content, rows = maintenance_view(data, tab, actions, status if tab == "review" else "all", linked_event)
            else:
                data = request("/api/quality", params={"test": test, "cell": cell, "progress": progress, "basis": basis})
                if tab == "review":
                    try:
                        data['history'] = request('/api/quality/history', params={'test': test if history_scope == 'selected' else 'all'})
                    except (ValueError, httpx.HTTPError):
                        data['history'] = {'unavailable':True}
                    content, rows = quality_view(data, tab, actions, mapmode, defect)
                elif quality_section == 'anomaly':
                    evidence = request('/api/quality/diagnostics', params={'test':test})
                    content, rows = quality_workspace.detection(evidence, eval_scope), evidence['rows']
                else:
                    content, rows = quality_view(data, tab, actions, mapmode, defect)
            if record_warning:
                if meta.get('demo'):
                    record_warning.className += ' demo-notice'
                content = html.Div([record_warning, content])
            if source.get('state') == 'stale':
                content = html.Div([callout('마지막 확인 자료로 조회 중', (source.get('message') or '') + ' 최신 자료 확인 전까지 기록 저장·새 분석은 중단됩니다.', 'warning'), content])
            return content, {"domain": key, "tab": tab, "payload": data, 'dataVersion': current_meta['dataVersion']}, rows, source_hint
        except (ValueError, httpx.HTTPError) as exc:
            message = str(exc) if isinstance(exc, ValueError) else "백엔드 연결을 확인하고 새로고침을 눌러 주세요."
            return callout("분석 데이터를 불러오지 못했습니다", message, "warning"), {"domain": key, "tab": tab, "payload": {}}, [], '공식 분석 자료 조회 실패'

    @app.callback(Output('tabs','value',allow_duplicate=True),Input('go-review','n_clicks'),Input('quality-go-review','n_clicks',allow_optional=True),prevent_initial_call=True)
    def go_review(*clicks):
        return 'review' if any(clicks) else no_update

    @app.callback(Output('tabs','style'),Output('go-review','style'),Output('quality-controls','className'),
        *[Output(id,'style') for id in ('open-inference','open-maintenance-upload','open-maintenance-report','open-quality-upload')],
        Input('url','pathname'),Input('tabs','value'))
    def workflow_actions(path,tab):
        key=domain(path); hidden={'display':'none'}
        return (hidden if key=='overview' else {},{} if key!='overview' and tab=='analysis' else hidden,'filter-toolbar'+(' review-mode' if tab=='review' else ''),
                {} if key=='demand' else hidden,{} if key=='maintenance' else hidden,{} if key=='maintenance' else hidden,{} if key=='quality' else hidden)

    @app.callback(Output("detail-drawer", "opened"), Output("detail-content", "children"), Output("selected-target", "data"), Output("review-form", "style"),
                  Input("main-grid", "selectedRows", allow_optional=True),Input('url','pathname'), State("view-data", "data"), prevent_initial_call=True)
    def detail(selected, path, view):
        if ctx.triggered_id=='url':
            return False,[],None,{'display':'none'}
        if not selected or not view or view["tab"] == "validation":
            return no_update, no_update, no_update, no_update
        row = selected[0]
        key = view["domain"]
        try:
            if key == "demand" and "part" in row:
                data = request("/api/demand", params={"date": view["payload"]["date"], "part": row["part"], "model": view["payload"]["model"]})
                content = [badge("부품 상세", "info"), html.H2(row["part"]), graph(demand_chart(data["trend"])), metric_rows([("예측", num(row["forecast"])+"개", row["model"]), ("계획", num(row["plan"])+"개", row["date"]), ("차이", f"{row['gap']:+,.0f}개", "정책과 재고 상황을 함께 검토")])]
                target = {"track": "demand", "target": row["part"], "decision": "reviewed", "date": row["date"], "model": row["model"]}
            elif key == "maintenance" and "event" in row:
                data = dict(view["payload"])
                data["points"] = [p for p in data["points"] if max(0, row["start"]-50) <= p["row"] <= row["end"]+50]
                content = [badge(row["severity"], "warning"), html.H2(row["event"]+" · "+row["type"]), graph(maintenance_chart(data)), metric_rows([("구간", f"{row['start']}–{row['end']}", f"{row['rows']}개 시점"), ("위험비", num(row["maxRisk"], 2), "선택 모델 점수 ÷ 임계값")]), callout("현장 SOP 미연결", "신호와 공정 조건을 확인한 뒤 검토 내용을 남겨 주세요.")]
                target = {"track": "maintenance", "target": row["id"], "decision": "reviewed"}
            elif key == "overview" and "route" in row:
                return False,[],None,{"display":"none"}
            else:
                return no_update, no_update, no_update, no_update
            return True, content, target, {}
        except (ValueError, httpx.HTTPError):
            return True, callout("상세 조회 실패", "선택을 다시 확인하세요.", "warning"), None, {"display": "none"}

    @app.callback(Output('url','href'),Input('main-grid','selectedRows',allow_optional=True),State('view-data','data'),prevent_initial_call=True)
    def open_work(rows,view):
        if rows and view and view['domain']=='overview' and view['tab']=='summary':
            valid = {r['route'] for r in view['payload'].get('workRows',[])}
            route = rows[0].get('route')
            if route in valid:
                return route
        return no_update

    @app.callback(Output("q-test", "value"), Input("quality-file-grid", "selectedRows", allow_optional=True), prevent_initial_call=True)
    def choose_quality_test(rows):
        return rows[0]['시험'] if rows else no_update

    @app.callback(Output("q-cell", "value"), Input("quality-heatmap", "clickData", allow_optional=True), Input("quality-3d", "clickData", allow_optional=True), Input("main-grid", "selectedRows", allow_optional=True), State("view-data", "data"), State("q-map", "value"), prevent_initial_call=True)
    def choose_cell(click, spatial_click, selected, view, mapmode):
        if not view or view["domain"] != "quality" or view["tab"] != "analysis":
            return no_update
        if ctx.triggered_id == 'quality-3d' and mapmode == 'cell':
            chosen = spatial.selection(spatial_click, {p['id'] for p in view['payload'].get('snapshot',{}).get('cells',[])})
            return chosen or no_update
        if ctx.triggered_id == "main-grid" and selected and "셀" in selected[0]:
            return selected[0]["셀"]
        if ctx.triggered_id == "quality-heatmap" and click and mapmode == "cell":
            point = click["points"][0]
            return f"{point['x']}{point['y']}"
        return no_update

    @app.callback(Output("confirm-modal", "opened"), Output("confirm-content", "children"), Output("pending-action", "data"),
                  Input("prepare-review", "n_clicks"), Input("prepare-quality", "n_clicks", allow_optional=True), Input("cancel-save", "n_clicks"), Input("confirm-save", "n_clicks"), Input('chat-draft', 'n_clicks'),
                  State("selected-target", "data"), State("review-note", "value"), State("q-test", "value"), State("decision-code", "value", allow_optional=True), State("decision-note", "value", allow_optional=True), State('chat-state', 'data'), State('view-data', 'data'), prevent_initial_call=True)
    def prepare(review_clicks, quality_clicks, cancel_clicks, confirm_clicks, draft_clicks, selected, review_note, test, code, note, chat, view):
        if ctx.triggered_id in ["cancel-save", "confirm-save"]:
            return False, no_update, no_update
        if ctx.triggered_id == 'chat-draft':
            try:
                if not chat or not chat.get('id'):
                    raise ValueError('저장된 Copilot 답변이 필요합니다.')
                pending = request('/api/copilot/draft', 'POST', json={'threadId': chat['id'], 'target': (selected or {}).get('target')})
                pending.pop('key', None)
                pending.pop('persistence', None)
            except (ValueError, httpx.HTTPError) as exc:
                return True, callout('초안을 만들지 못했습니다', str(exc) if isinstance(exc, ValueError) else '연결을 확인하세요.', 'warning'), None
        elif ctx.triggered_id == "prepare-quality":
            if not quality_clicks:
                return no_update, no_update, no_update
            pending = {"track": "quality", "target": test, "decision": code, "note": note or ""}
        elif selected and review_clicks:
            pending = {**selected, "note": review_note or ""}
        else:
            return no_update, no_update, no_update
        labels = {"reviewed": "확인 완료", "clear": "이상 없음", "retest": "재시험 요청", "hold": "출하 보류"}
        try:
            if ctx.triggered_id != 'chat-draft':
                pending['dataVersion'] = (view or {}).get('dataVersion')
                if not pending['dataVersion']:
                    raise ValueError('분석 자료를 새로고침 후 다시 검토하세요.')
            validated = request('/api/preview/decision', 'POST', json=pending)
            previous = request('/api/records/state')['records'].get(validated['key'], {})
            pending.update(requestId=uuid4().hex, expectedRevision=previous.get('revision', 0), dataVersion=validated['dataVersion'])
        except (ValueError, httpx.HTTPError) as exc:
            return True, callout('저장 준비 실패', str(exc) if isinstance(exc, ValueError) else '연결을 확인하세요.', 'warning'), None
        return True, html.Div([html.H3(pending["target"]), html.P(labels.get(pending["decision"], "확인")), html.P(pending["note"] or "메모 없음")]), pending

    @app.callback(Output("session-actions", "data"), Output("save-feedback", "children"), Input("confirm-save", "n_clicks"), Input("url", "pathname"), State("pending-action", "data"), State("session-actions", "data"), prevent_initial_call=True)
    def save(_, path, pending, actions):
        if ctx.triggered_id == "url":
            return no_update, ""
        if not pending:
            return no_update, no_update
        try:
            result = request("/api/records", "POST", json=pending)
            updated = dict(actions or {})
            updated[result["key"]] = result
            return updated, html.Div([icon("circle-check", 16), "업무 기록을 Firebase에 저장했습니다. 상단 업무 기록에서 이력을 확인하세요."], className="save-toast")
        except (ValueError, httpx.HTTPError) as exc:
            return no_update, callout("기록 저장 실패", str(exc) if isinstance(exc, ValueError) else "대상과 API 연결을 확인하세요.", "warning")

    @app.callback(Output('confirm-save', 'disabled'), Input('pending-action', 'data'))
    def confirmation_enabled(pending):
        return not bool(pending)

    @app.callback(Output("copilot-drawer", "opened"), Input("open-copilot", "n_clicks"), Input('chat-draft', 'n_clicks'), Input('url','pathname'), Input('url','search'), prevent_initial_call=True)
    def copilot(_, draft, path, search):
        return ctx.triggered_id == 'open-copilot'

    @app.callback(Output("inference-modal", "opened"), Input("open-inference", "n_clicks"), prevent_initial_call=True)
    def inference_modal(_):
        return True

    @app.callback(Output("inference-result", "children"), Input("csv-upload", "contents"), State("csv-upload", "filename"), prevent_initial_call=True)
    def infer(contents, filename):
        if not contents:
            return no_update
        try:
            payload = base64.b64decode(contents.split(",", 1)[1], validate=True)
            checked = request('/api/demand/check','POST',files={'file':(filename or 'input.csv',payload,'text/csv')})
            checks = panel('CSV 입력 검사',grid(checked['checks'],'d-upload-checks',[
                {'field':'검사 항목','headerName':'검사 항목','flex':1},
                {'field':'결과','headerName':'결과','maxWidth':85,'minWidth':75},
                {'field':'설명','headerName':'설명','flex':3,'wrapText':True,'autoHeight':True}],height=390))
            if not checked['valid']:
                return html.Div([callout('입력 수정 필요','오류 항목을 수정한 뒤 다시 업로드하세요.','warning'),checks])
            result = request("/api/demand/infer", "POST", files={"file": (filename or "input.csv", payload, "text/csv")})
            return html.Div([checks,badge("추론 완료", "success"), metric_rows([("D+3 목표일", result["target_date"], result["part_number"]), ("운영 예측", num(result["recommended_forecast"])+"개", result["recommended_model"]), ("학습형 보조", num(result["auxiliary_prediction"])+"개", result["auxiliary_model"] + " · 알려진 부품만 제공"), ("기존 계획", num(result["plan_d3_reference"])+"개", "공식 시드는 변경하지 않습니다.")]), html.P(result.get("fallback_reason") or result.get("input_warning") or result["evaluation_note"], className="section-note")])
        except (ValueError, httpx.HTTPError):
            return callout("CSV 추론 실패", "템플릿의 필수 열, 동일 부품의 연속 3일, 0 이상 수량을 확인하세요.", "warning")

    @app.callback(Output("export-modal", "opened"), Output("export-ready", "children"), Input("export", "n_clicks"), State("main-grid", "virtualRowData", allow_optional=True), State("export-data", "data"), State("view-data", "data"), State('pack-results-grid','virtualRowData',allow_optional=True), prevent_initial_call=True)
    def export(_, visible, rows, view, pack_rows):
        selected = (pack_rows if pack_rows is not None else rows or []) if (view or {}).get('tab')=='packs' else visible if visible is not None else rows or []
        try:
            result = request("/api/exports", "POST", json={"rows": selected, "name": f"{view['domain']}_{view['tab']}"})
            return True, html.Div([html.P(f"현재 필터를 적용한 {result['count']:,}행이 준비되었습니다."), html.A("CSV 파일 다운로드", href=result["url"], className="template-link"), html.P("다운로드 링크는 15분 동안 유효합니다.", className="section-note")])
        except (ValueError, httpx.HTTPError):
            return True, callout("내보내기 실패", "선택 조건과 API 연결을 확인하고 다시 시도하세요.", "warning")

    return app


def kpi_small(label, text):
    return html.Div([html.Small(label), html.Strong(text)], className="metadata-item")
