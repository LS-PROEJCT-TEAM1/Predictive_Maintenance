from __future__ import annotations

import base64
import os
from pathlib import Path
from uuid import uuid4

import httpx
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
import dash_mantine_components as dmc
from flask import request as flask_request
from frontend import copilot_ui

from frontend.components import badge, callout, graph, grid, icon, metric_rows, num, panel, select
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
    shades = ["#F0F3FA", "#DCE4F3", "#B8C8E5", "#8EA6D3", "#607FBF", "#3358AA", "#0A1E5A", "#08184A", "#061239", "#040C29"]
    app.layout = dmc.MantineProvider(theme={"primaryColor": "lsblue", "colors": {"lsblue": shades}, "fontFamily": "'noto_d', sans-serif", "defaultRadius": "sm"}, children=[
        dcc.Location(id="url", refresh=False), dcc.Store(id="view-data"), dcc.Store(id="export-data"),
        dcc.Store(id="session-actions", data={}), dcc.Store(id="selected-target"), dcc.Store(id="pending-action"),
        dmc.Modal(id="export-modal", title="CSV 내보내기", centered=True, closeButtonProps={"aria-label": "닫기"}, children=html.Div(id="export-ready")),
        html.Div([
            html.Aside([
                dcc.Link([html.Div(html.Img(src='/assets/ci_img20.png', alt='LS', className='brand-logo'), className='brand-mark'), html.Strong(["BatteryFlow AI", html.Span("운영센터", className='brand-center')]), html.Small("MANUFACTURING INTELLIGENCE")], href="/", className="brand", title='BatteryFlow AI 운영센터 홈'),
                html.Div("WORKSPACE", className="nav-caption"),
                html.Nav([dcc.Link([icon(glyph), html.Span(label)], href="/" if key == "overview" else f"/{key}", id=f"nav-{key}", className="nav-link") for key, label, glyph in [("overview", "통합 현황", "layout-dashboard"), ("demand", "공급망 예측", "chart-no-axes-combined"), ("maintenance", "예지보전", "activity"), ("quality", "품질 보증", "shield-check")]]),
                html.Div(className="sidebar-spacer"),
                html.Div([html.Div([html.Span(className="connection-dot"), "공식 데이터 기준"], className="sidebar-data-title"), html.Small(meta["dataVersion"]), html.Small(f"{meta['documents']}개 문서 · 로컬 시드")], className="sidebar-data"),
                html.Button([icon("sparkles", 20), html.Div([html.Strong("AI Copilot"), html.Small("근거와 함께 질문하기")]), icon("chevron-right", 16)], id="open-copilot", className="copilot-entry"),
                html.Div([html.Span("L", className="user-avatar"), html.Div(id="signed-user"), html.Button(icon('log-out', 16), id='logout-button', title='로그아웃', **{'aria-label': '로그아웃'})], className="sidebar-user")], className="app-sidebar"),
            html.Main([
                html.Div([html.Div([html.Span("BatteryFlow AI"), icon("chevron-right", 14), html.Strong(id="breadcrumb")], className="breadcrumb"),
                          html.Div([badge("로컬 체험" if meta.get('demo') else "직원 전용", "success"), html.Button('업무 기록', id='open-records', className='utility-button'), html.Span("시드 생성 "+meta["generatedAt"][:10]), html.A([icon("code", 15), "API 문서"], href="/docs", target="_blank")], className="utility-right")], className="utility-bar"),
                html.Div([
                    html.Div([html.Div([html.H1(id="page-title"), html.P(id="page-description")]), html.Div([dmc.Button("새로고침", id="refresh", variant="default", leftSection=icon("refresh-cw", 15), size="sm"), dmc.Button("CSV 내보내기", id="export", variant="outline", leftSection=icon("download", 15), size="sm")], className="page-actions")], className="page-heading"),
                    dmc.Tabs(id="tabs", value="summary", children=[], className="domain-tabs"),
                    html.Div([
                        html.Div([select("d-date", "목표일", meta["dates"], meta["dates"][-1]), select("d-part", "부품", [{"label": "전체 부품", "value": "ALL"}]+[{"label": p, "value": p} for p in meta["parts"]], "ALL"),
                                  select("d-model", "예측 모델", [{"label": "이동평균 · 운영 기본", "value": "3-day Moving Average"}, {"label": "XGBoost · 학습형 보조", "value": "XGBoost"}, {"label": "LightGBM · 비교", "value": "LightGBM"}, {"label": "CatBoost · 비교", "value": "CatBoost"}, {"label": "LSTM · 비교", "value": "LSTM"}], "3-day Moving Average"),
                                  dmc.Button("CSV로 재예측", id="open-inference", leftSection=icon("upload", 16), variant="light", className="filter-button")], id="demand-controls", className="filter-toolbar"),
                        html.Div([select("m-run", "시험 파일", meta["runs"], "WeldingTest_04_NG"), select("m-sup", "지도 모델", meta["supervised"], meta["defaultSupervised"]), select("m-unsup", "비지도 모델", meta["unsupervised"], meta["defaultUnsupervised"]), badge("과거 시험 재생", "info")], id="maintenance-controls", className="filter-toolbar"),
                        html.Div([select("q-test", "시험 ID", meta["tests"], "Test07_NG_dchg"), select("q-cell", "선택 셀", [f"M{m:02d}CV{c:02d}" for m in range(1, 17) for c in range(1, 12)], "M02CV01"),
                                  html.Div([html.Label("분석 진행률"), dmc.Slider(id="q-progress", value=100, min=10, max=100, step=10, marks=[{"value": 10, "label": "10%"}, {"value": 100, "label": "100%"}])], className="progress-control"),
                                  select("q-map", "위치 지도", [{"label": "셀 전압 16×11", "value": "cell"}, {"label": "모듈 온도 16×2", "value": "temperature"}], "cell"), badge("PCA (T²·SPE)", "info")], id="quality-controls", className="filter-toolbar"),
                        html.Div([select("overview-track", "분석 영역", [{"label": DOMAIN[t], "value": t} for t in ["demand", "maintenance", "quality"]], "demand"),
                                  select("result-kind", "결과 유형", [{"label": v, "value": k} for k, v in [("prediction", "예측 결과"), ("evaluation", "평가 결과"), ("raw", "원본 측정·이력"), ("processed", "가공 데이터")]], "prediction")], id="overview-controls", className="filter-toolbar"),
                        html.Div([dmc.TextInput(id="part-search", label="부품 검색", placeholder="Part 번호 검색", leftSection=icon("search", 15), className="context-select"), select("direction-filter", "검토 방향", [{"label": "전체", "value": "all"}, {"label": "상향 검토", "value": "up"}, {"label": "하향 검토", "value": "down"}], "all")], id="demand-local", className="filter-toolbar compact"),
                        html.Div([select("event-status", "이벤트 상태", [{"label": "전체", "value": "all"}, {"label": "미확인", "value": "open"}, {"label": "확인 완료", "value": "done"}], "all")], id="maintenance-local", className="filter-toolbar compact"),
                    ]),
                    html.Div(id="save-feedback", role="status"),
                    dcc.Loading(html.Div(id="page-content"), type="circle", color="#0A1E5A", delay_show=300),
                    html.Footer([html.Span("KAMP 제조 데이터 기반 · 서로 다른 트랙의 원본은 병합하지 않습니다."), html.Span("로컬 체험 · 외부 서비스 미연결" if meta.get('demo') else "분석: 로컬 시드 / 인증·기록: Firebase")], className="footer")
                ], className="dashboard-container")], className="app-main")], className="app-shell"),
        dmc.Drawer(closeButtonProps={"aria-label": "닫기"}, id="detail-drawer", title="분석 상세", position="right", size=560, children=[html.Div(id="detail-content"),
            html.Div([dmc.Textarea(id="review-note", label="검토 메모", placeholder="확인한 내용을 입력하세요.", minRows=3, inputProps={"maxLength": 2000}), dmc.Button("확인 처리", id="prepare-review", leftSection=icon("check"), className="spaced-button")], id="review-form")]),
        dmc.Modal(closeButtonProps={"aria-label": "닫기"}, id="confirm-modal", title="업무 기록 저장 확인", centered=True, children=[html.Div(id="confirm-content"),
            callout("직원 공용 업무 기록", "작성자와 시각을 포함해 Firebase에 저장합니다. 이전 기록도 이력에 남습니다."), dmc.Group([dmc.Button("취소", id="cancel-save", variant="default"), dmc.Button("확인 후 저장", id="confirm-save")], justify="flex-end")]),
        copilot_ui.drawer(),
        dmc.Drawer(id='records-drawer', title='업무 기록', position='right', size=600, closeButtonProps={'aria-label': '닫기'}, children=[html.P('직원 공용 · 최신 100건 · 수정 전 판정도 이력으로 보존됩니다.', className='section-note'), dcc.Loading(html.Div(id='records-content'))]),
        dmc.Modal(closeButtonProps={"aria-label": "닫기"}, id="inference-modal", title="CSV로 D+3 재예측", centered=True, size="lg", children=[
            html.P("동일 부품의 연속 3일 자료를 업로드하세요. 기존 모델로 추론하며 재학습하지 않습니다."),
            html.A("입력 템플릿 다운로드", href="/api/demand/template", className="template-link"),
            dcc.Upload(id="csv-upload", children=html.Div([icon("upload", 24), html.Strong("CSV 파일 선택 또는 여기에 끌어놓기"), html.Small("UTF-8 · 최대 1MB · 3행")]), accept=".csv", max_size=1000000, className="upload-zone"),
            dcc.Loading(html.Div(id="inference-result"), type="circle")])
    ])

    copilot_ui.register(app, request)

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

    @app.callback(Output("tabs", "children"), Output("tabs", "value"), Output("page-title", "children"), Output("page-description", "children"), Output("breadcrumb", "children"),
                  *[Output(f"nav-{key}", "className") for key in DOMAIN], Input("url", "pathname"))
    def route(path):
        key = domain(path)
        descriptions = {"overview": "세 가지 분석을 하나의 의사결정으로.", "demand": "계획과 예측의 차이에서, 먼저 검토할 부품을 찾습니다.", "maintenance": "용접 신호의 변화에서, 확인이 필요한 구간을 찾습니다.", "quality": "시험의 이상 근거와 기여 위치를 확인합니다."}
        title = "D+3 발주량 예측" if key == "demand" else DOMAIN[key]
        return ([dmc.TabsList([dmc.TabsTab(label, value=value) for value, label in TABS[key]])], TABS[key][0][0], title, descriptions[key], DOMAIN[key],
                *["nav-link active" if k == key else "nav-link" for k in DOMAIN])

    @app.callback(*[Output(id, "style") for id in ["demand-controls", "maintenance-controls", "quality-controls", "overview-controls", "demand-local", "maintenance-local", "result-kind"]], Input("url", "pathname"), Input("tabs", "value"))
    def control_visibility(path, tab):
        key = domain(path)
        flags = [key == "demand", key == "maintenance", key == "quality", key == "overview" and tab in ["data", "results"], key == "demand" and tab == "review", key == "maintenance" and tab == "review", key == "overview" and tab == "results"]
        return [{} if show else {"display": "none"} for show in flags]

    @app.callback(Output("page-content", "children"), Output("view-data", "data"), Output("export-data", "data"),
        Input("url", "pathname"), Input("tabs", "value"), Input("d-date", "value"), Input("d-part", "value"), Input("d-model", "value"),
        Input("m-run", "value"), Input("m-sup", "value"), Input("m-unsup", "value"), Input("q-test", "value"), Input("q-cell", "value"), Input("q-progress", "value"), Input("q-map", "value"),
        Input("overview-track", "value"), Input("result-kind", "value"), Input("session-actions", "data"), Input("part-search", "value"), Input("direction-filter", "value"), Input("event-status", "value"), Input("refresh", "n_clicks"))
    def render(path, tab, date, part, model, run, sup, unsup, test, cell, progress, mapmode, track, kind, actions, search, direction, status, _):
        key = domain(path)
        tab = tab if tab in dict(TABS[key]) else TABS[key][0][0]
        actions = actions or {}
        try:
            record_warning = None
            try:
                actions = request('/api/records/state')['records']
            except (ValueError, httpx.HTTPError):
                actions = {'__unavailable__': True}
                record_warning = (callout('로컬 체험 모드', '분석·필터·CSV 추론을 체험할 수 있습니다. 업무 기록 저장·조회와 Copilot은 비활성화되어 있습니다.') if meta.get('demo') else
                    callout('공유 기록 조회 불가', 'Firestore 연결 또는 무료 할당량을 확인하세요. 아래 판정·확인 상태는 최신 저장 상태를 반영하지 않습니다.', 'warning'))
            if tab == "validation":
                data = request(f"/api/validation/{key}")
                content, rows = validation_view(data, key)
            elif key == "overview":
                data = {}
                if tab == "project":
                    content, rows = project_view()
                elif tab == "conclusion":
                    content, rows = conclusion_view()
                elif tab == "data":
                    rows = request(f"/api/data-quality/{track}")["rows"]
                    content = html.Div([callout("처리 근거를 먼저 확인하세요", "트랙별 분석 단위와 제공 항목이 다릅니다. 없는 품질 검사를 정상값으로 채우지 않습니다."), panel(f"{DOMAIN[track]} · 데이터 품질", grid(rows, "main-grid", height=550), f"{len(rows)}개 품질 항목 · 공식 시드 v2 · 표시된 항목 다운로드 가능")])
                elif tab == "results":
                    data = request(f"/api/results/{track}", params={"kind": kind})
                    rows = data["rows"]
                    content = html.Div([html.Div([kpi_small("데이터 버전", data["dataVersion"]), kpi_small("생성 시각", data["generatedAt"][:19].replace("T", " ")+" UTC"), kpi_small("표시 행", f"{len(rows):,}")], className="metadata-strip"),
                        callout("결과 범위", data["notice"]), panel("결과 조회", grid(rows, "main-grid", height=560), "열별 필터와 정렬 후 CSV 내보내기로 현재 결과를 내려받으세요."),
                        callout("재현과 오류 복구", "원본 시드 해시를 검사한 뒤 API가 시작됩니다. 실행 오류는 LOCAL_RUN.md의 환경 설치와 시드 검증 절차를 확인하세요.")])
                else:
                    data = request("/api/overview")
                    content = overview(data, meta, actions)
                    rows = data["actions"]
            elif key == "demand":
                data = request("/api/demand", params={"date": date, "part": part, "model": model})
                content, rows = demand_view(data, tab, actions, search if tab == "review" else None, direction if tab == "review" else "all")
            elif key == "maintenance":
                data = request("/api/maintenance", params={"run": run, "supervised": sup, "unsupervised": unsup})
                content, rows = maintenance_view(data, tab, actions, status if tab == "review" else "all")
            else:
                data = request("/api/quality", params={"test": test, "cell": cell, "progress": progress})
                content, rows = quality_view(data, tab, actions, mapmode)
            if record_warning:
                content = html.Div([record_warning, content])
            return content, {"domain": key, "tab": tab, "payload": data}, rows
        except (ValueError, httpx.HTTPError) as exc:
            message = str(exc) if isinstance(exc, ValueError) else "백엔드 연결을 확인하고 새로고침을 눌러 주세요."
            return callout("분석 데이터를 불러오지 못했습니다", message, "warning"), {"domain": key, "tab": tab, "payload": {}}, []

    @app.callback(Output("detail-drawer", "opened"), Output("detail-content", "children"), Output("selected-target", "data"), Output("review-form", "style"),
                  Input("main-grid", "selectedRows", allow_optional=True), State("view-data", "data"), prevent_initial_call=True)
    def detail(selected, view):
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
                return True, [html.H2(row["target"]), html.P(row["title"]), dcc.Link("해당 분석 화면 열기 →", href=row["route"])], None, {"display": "none"}
            else:
                return no_update, no_update, no_update, no_update
            return True, content, target, {}
        except (ValueError, httpx.HTTPError):
            return True, callout("상세 조회 실패", "선택을 다시 확인하세요.", "warning"), None, {"display": "none"}

    @app.callback(Output("q-cell", "value"), Input("quality-heatmap", "clickData", allow_optional=True), Input("main-grid", "selectedRows", allow_optional=True), State("view-data", "data"), State("q-map", "value"), prevent_initial_call=True)
    def choose_cell(click, selected, view, mapmode):
        if not view or view["domain"] != "quality" or view["tab"] != "analysis":
            return no_update
        if ctx.triggered_id == "main-grid" and selected and "셀" in selected[0]:
            return selected[0]["셀"]
        if ctx.triggered_id == "quality-heatmap" and click and mapmode == "cell":
            point = click["points"][0]
            return f"{point['y']}CV{int(point['x']):02d}"
        return no_update

    @app.callback(Output("confirm-modal", "opened"), Output("confirm-content", "children"), Output("pending-action", "data"),
                  Input("prepare-review", "n_clicks"), Input("prepare-quality", "n_clicks", allow_optional=True), Input("cancel-save", "n_clicks"), Input("confirm-save", "n_clicks"), Input('chat-draft', 'n_clicks'),
                  State("selected-target", "data"), State("review-note", "value"), State("q-test", "value"), State("decision-code", "value", allow_optional=True), State("decision-note", "value", allow_optional=True), State('chat-state', 'data'), prevent_initial_call=True)
    def prepare(review_clicks, quality_clicks, cancel_clicks, confirm_clicks, draft_clicks, selected, review_note, test, code, note, chat):
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
            validated = request('/api/preview/decision', 'POST', json=pending)
            previous = request('/api/records/state')['records'].get(validated['key'], {})
            pending.update(requestId=uuid4().hex, expectedRevision=previous.get('revision', 0))
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

    @app.callback(Output("copilot-drawer", "opened"), Input("open-copilot", "n_clicks"), Input('chat-draft', 'n_clicks'), prevent_initial_call=True)
    def copilot(_, draft):
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
            result = request("/api/demand/infer", "POST", files={"file": (filename or "input.csv", payload, "text/csv")})
            return html.Div([badge("추론 완료", "success"), metric_rows([("D+3 목표일", result["target_date"], result["part_number"]), ("운영 예측", num(result["recommended_forecast"])+"개", "3일 이동평균"), ("학습형 보조", num(result["xgboost_prediction"])+"개", "XGBoost · 알려진 부품만 제공"), ("기존 계획", num(result["plan_d3_reference"])+"개", "공식 시드는 변경하지 않습니다.")])])
        except (ValueError, httpx.HTTPError):
            return callout("CSV 추론 실패", "템플릿의 필수 열, 동일 부품의 연속 3일, 0 이상 수량을 확인하세요.", "warning")

    @app.callback(Output("export-modal", "opened"), Output("export-ready", "children"), Input("export", "n_clicks"), State("main-grid", "virtualRowData", allow_optional=True), State("export-data", "data"), State("view-data", "data"), prevent_initial_call=True)
    def export(_, visible, rows, view):
        selected = visible if visible is not None else rows or []
        try:
            result = request("/api/exports", "POST", json={"rows": selected, "name": f"{view['domain']}_{view['tab']}"})
            return True, html.Div([html.P(f"현재 필터를 적용한 {result['count']:,}행이 준비되었습니다."), html.A("CSV 파일 다운로드", href=result["url"], className="template-link"), html.P("다운로드 링크는 15분 동안 유효합니다.", className="section-note")])
        except (ValueError, httpx.HTTPError):
            return True, callout("내보내기 실패", "선택 조건과 API 연결을 확인하고 다시 시도하세요.", "warning")

    return app


def kpi_small(label, text):
    return html.Div([html.Small(label), html.Strong(text)], className="metadata-item")
