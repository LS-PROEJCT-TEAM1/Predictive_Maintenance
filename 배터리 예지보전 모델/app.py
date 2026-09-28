from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import dash_ag_grid as dag
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, callback, callback_context, dcc, html, no_update

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from backend.maintenance_policy import POLICY, alarm_regions

from components.maintenance_charts import event_context_figure, heatmap_figure, power_figure, score_figure  # noqa: E402
from components.maintenance_ui import app_header, field_pair, icon, kpi_card, modal_header, numbered_action  # noqa: E402

OUTPUT_DIR = ROOT / "outputs" / "track_b_final_v2"
MODEL_DIR = OUTPUT_DIR / "models"
required = [OUTPUT_DIR / name for name in ("metrics.csv", "data_quality.csv", "run_manifest.json")]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError("Run src/track_b_final_v2.py first. Missing: " + ", ".join(missing))

metrics = pd.read_csv(OUTPUT_DIR / "metrics.csv")
data_quality = pd.read_csv(OUTPUT_DIR / "data_quality.csv")
manifest = json.loads((OUTPUT_DIR / "run_manifest.json").read_text(encoding="utf-8"))
locked_metrics = metrics[metrics["split"].eq("locked_test")].copy()
supervised_models = locked_metrics.loc[locked_metrics["family"].eq("supervised"), "model"].tolist()
unsupervised_models = locked_metrics.loc[locked_metrics["family"].eq("unsupervised"), "model"].tolist()
DEFAULT_SUPERVISED = manifest.get('default_supervised', 'LogisticCurrent')
DEFAULT_UNSUPERVISED = manifest.get('default_unsupervised', manifest['winner_selected_on_validation'])


def load_replay_from_firestore_seed(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)["data"]
            common = {
                "source_file": data["sourceFile"], "source_row": data["sourceRow"],
                "cycle_local": data["cycle"], "group_id": data["groupId"], "PageNo": data["pageNo"],
                "WorkingTime": data["workingTime"], "RealPower": data["signals"]["realPower"],
                "SetPower": data["signals"]["setPower"], "GateOnTime": data["signals"]["gateOnTime"],
                "Speed": data["signals"]["speed"], "Length": data["signals"]["length"],
                "label": data["actualLabel"], "expected_power": data["normalReference"]["expectedPower"],
                "normal_scale": data["normalReference"]["scale"],
            }
            for model_name, model in data["models"].items():
                rows.append({**common, "model": model_name, "family": model["family"], "score": model["score"], "threshold": model["threshold"], "prediction": model["prediction"]})
    if not rows:
        raise ValueError(f"Firestore replay seed is empty: {path}")
    return pd.DataFrame(rows)


def load_replay() -> pd.DataFrame:
    seed_path = ROOT / "firestore" / "seed" / "measurements.jsonl"
    if not seed_path.exists():
        raise FileNotFoundError(f"Dashboard replay seed is missing: {seed_path}")
    return load_replay_from_firestore_seed(seed_path)


replay = load_replay()
source_files = replay["source_file"].drop_duplicates().tolist()


def model_rows(source: str, model: str) -> pd.DataFrame:
    return replay[replay["source_file"].eq(source) & replay["model"].eq(model)].sort_values("source_row").reset_index(drop=True)


def combined_rows(source: str, supervised: str, unsupervised: str) -> pd.DataFrame:
    sup, unsup = model_rows(source, supervised), model_rows(source, unsupervised)
    columns = ["source_file", "source_row", "cycle_local", "group_id", "PageNo", "WorkingTime", "RealPower", "SetPower", "GateOnTime", "Speed", "Length", "label", "expected_power", "normal_scale"]
    result = sup[columns].copy()
    result["supervised_score"], result["supervised_threshold"] = sup["score"], sup["threshold"]
    result["supervised_prediction"] = sup["prediction"]
    result["unsupervised_score"], result["unsupervised_threshold"] = unsup["score"], unsup["threshold"]
    result["unsupervised_prediction"] = unsup["prediction"]
    result["combined_prediction"] = (result["supervised_prediction"].astype(bool) | result["unsupervised_prediction"].astype(bool)).astype(int)
    result["risk_ratio"] = np.maximum(
        result["supervised_score"] / result["supervised_threshold"].clip(lower=1e-12),
        result["unsupervised_score"] / result["unsupervised_threshold"].clip(lower=1e-12),
    )
    return result


def regions(values: np.ndarray) -> list[tuple[int, int]]:
    changes = np.diff(np.pad(np.asarray(values, dtype=int), (1, 1)))
    return list(zip(np.flatnonzero(changes == 1).tolist(), (np.flatnonzero(changes == -1) - 1).tolist()))


def event_type(event: pd.DataFrame) -> str:
    direction = "저출력" if float((event["RealPower"] - event["expected_power"]).median()) < 0 else "고출력"
    if len(event) == 1:
        return f"고립형 {direction} 이상"
    if len(event) >= 39 or event["cycle_local"].nunique() >= 2:
        return f"연속 {direction} 이상"
    if event["PageNo"].nunique() <= 2 and event["cycle_local"].nunique() >= 2:
        return f"반복 위치형 {direction} 이상"
    return f"단기 {direction} 이상"


def severity(ratio: float, length: int) -> str:
    if ratio >= 3 or length >= 39:
        return "위험"
    if ratio >= 1.5 or length >= 5:
        return "주의"
    return "관찰"


def event_key(source: str, supervised: str, unsupervised: str, event_id: int) -> str:
    return f"{source}|{supervised}|{unsupervised}|{manifest['pipeline_version']}|{POLICY['version']}|{event_id}"


def events_for(frame: pd.DataFrame, source: str, supervised: str, unsupervised: str, review_state: dict | None = None) -> pd.DataFrame:
    review_state = review_state or {}
    rows: list[dict] = []
    signal_rows = [dict(row=int(r.source_row), time=str(r.WorkingTime), prediction=int(r.combined_prediction))
                   for r in frame.itertuples()]
    for event_id, (start, end) in enumerate(alarm_regions(signal_rows, POLICY['gapSeconds']), 1):
        event = frame.iloc[start:end + 1]
        pages = sorted(event["PageNo"].unique().tolist())
        page_text = f"{pages[0]}–{pages[-1]}" if len(pages) > 1 else str(pages[0])
        sup, unsup = bool(event["supervised_prediction"].any()), bool(event["unsupervised_prediction"].any())
        agreement = f"{supervised} + {unsupervised}" if sup and unsup else supervised if sup else unsupervised
        ratio = float(event["risk_ratio"].max())
        key = event_key(source, supervised, unsupervised, event_id)
        rows.append(
            {
                "event_id": event_id, "event_label": f"#{event_id:03d}", "event_key": key,
                "severity": severity(ratio, len(event)), "event_type": event_type(event),
                "start_row": int(event["source_row"].iloc[0]), "end_row": int(event["source_row"].iloc[-1]),
                "start_time": str(event["WorkingTime"].iloc[0]),
                "start_time_display": str(event["WorkingTime"].iloc[0]).split(" ")[-1][:8],
                "duration_rows": int(len(event)),
                "cycles": int(event["cycle_local"].nunique()), "pages": page_text,
                "max_risk_ratio": ratio, "ratio_display": f"{ratio:.2f}×", "model_agreement": agreement,
                "review_status": review_state.get(key, {}).get("status", "미확인"), "detail": "상세",
            }
        )
    return pd.DataFrame(rows)


def quality_summary(source: str) -> list[tuple[str, str, str]]:
    row = data_quality[data_quality["file"].eq(source)].iloc[0]
    rows = max(float(row["rows"]), 1.0)
    completeness = max(0.0, 100 - float(row["missing_cells"]) / (rows * 9) * 100)
    uniqueness = max(0.0, 100 - float(row["duplicate_rows"]) / rows * 100)
    integrity = max(0.0, 100 - float(row["nonpositive_time_diffs"]) / rows * 100)
    consistency = min(100.0, float(row["complete_cycles"]) * 39 / rows * 100)
    return [
        ("완전성", f"{completeness:.1f}%", "필수 열 결측 기준"), ("유일성", f"{uniqueness:.1f}%", "중복 행 기준"),
        ("유효성", "100.0%", "필수 열·자료형 검사 통과"), ("일관성", f"{consistency:.1f}%", "PageNo 1→39 완전 Cycle 기준"),
        ("정확성", "–", "현장 기준값 부재로 평가 제외"), ("무결성", f"{integrity:.1f}%", "WorkingTime 순서 기준"),
    ]


def toolbar() -> html.Div:
    return html.Div(
        [
            html.Div([html.Label("시험 파일"), dmc.Select(id="source-file", data=source_files, value="WeldingTest_04_NG", allowDeselect=False, size="sm")], className="filter-field source-field"),
            html.Div([html.Label("지도 모델"), dmc.Select(id="supervised-model", data=supervised_models, value=DEFAULT_SUPERVISED, allowDeselect=False, size="sm")], className="filter-field"),
            html.Div([html.Label("비지도 모델"), dmc.Select(id="unsupervised-model", data=unsupervised_models, value=DEFAULT_UNSUPERVISED, allowDeselect=False, size="sm")], className="filter-field"),
            dmc.Button("시험 데이터 재생", id="open-quality", leftSection=icon("lucide:circle-play", 16), variant="light", color="teal", className="toolbar-button"),
            dmc.Button("모델 정보", id="open-model", leftSection=icon("lucide:file-chart-column", 16), variant="outline", color="indigo", className="toolbar-button"),
            dmc.Button("이벤트 내보내기", id="export-events", leftSection=icon("lucide:download", 16), color="indigo", className="toolbar-button toolbar-primary"),
        ],
        className="filter-toolbar",
    )


def action_panel() -> html.Div:
    return html.Div(
        [
            html.Div([html.H2("우선 확인 권장"), html.Span("원인 추정이 아닌 확인 순서", className="section-caption")], className="panel-header"),
            html.Div(id="focus-event", className="focus-event"),
            html.Div(
                [numbered_action(1, "이상 PageNo 현장 확인", "표시된 용접 위치의 실제 상태를 확인합니다."), numbered_action(2, "해당 시점 설비 조건 확인", "전류·제어 신호·출력 이상 유무를 점검합니다."), numbered_action(3, "해당 구간 생산품 품질 확인", "동일 위치 전후 제품의 품질 결과를 확인합니다.")],
                className="action-list",
            ),
            html.Div([dmc.Select(id="assignee", data=["박정비", "김품질", "이생산"], value="박정비", allowDeselect=False, size="xs", className="assignee-select"), html.Span(id="focus-status", className="review-badge pending"), dmc.Button("상세 보기", id="open-latest-event", variant="outline", color="indigo", size="xs"), dmc.Button("확인 처리", id="confirm-focus-review", color="indigo", size="xs")], className="action-footer"),
            html.Div([icon("lucide:info", 16), html.Span("AI는 원인을 확정하지 않습니다. 현장 점검과 품질 결과로 판단해야 합니다.")], className="ai-disclaimer"),
        ],
        className="panel action-panel",
    )


event_columns = [
    {"field": "event_label", "headerName": "이벤트", "width": 76, "pinned": "left"},
    {"field": "start_time_display", "headerName": "시각", "width": 82},
    {"field": "event_type", "headerName": "유형", "minWidth": 126, "flex": 1},
    {"field": "pages", "headerName": "PageNo", "width": 78},
    {"field": "ratio_display", "headerName": "초과", "width": 72},
    {"field": "review_status", "headerName": "상태", "width": 78, "cellClassRules": {"status-complete": "params.value === '확인 완료'", "status-pending": "params.value === '미확인'"}},
    {"field": "detail", "headerName": "상세", "width": 76, "cellRenderer": "EventDetailButton", "sortable": False, "filter": False},
]

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "배터리 용접 예지보전 | LS 제조 AI 운영센터"
app.layout = dmc.MantineProvider(
    theme={"fontFamily": "Segoe UI, Malgun Gothic, sans-serif", "primaryColor": "indigo"},
    children=html.Div(
        [
            app_header(),
            html.Main(
                [
                    html.Div([html.Div([html.H1("배터리 용접 예지보전"), html.P("용접 위치별 이상 징후를 조기에 확인하고 현장 점검으로 연결합니다.")]), html.Div([html.Span("검증 기준 적용", className="page-chip"), html.Span("재생 데이터", className="page-chip secondary")], className="page-status")], className="page-heading"),
                    toolbar(),
                    html.Div([kpi_card("종합 상태", "kpi-status", "kpi-status-detail", "lucide:triangle-alert", "danger"), kpi_card("최대 임계값 초과배수", "kpi-ratio", "kpi-ratio-detail", "lucide:activity", "danger"), kpi_card("이상 이벤트", "kpi-events", "kpi-events-detail", "lucide:clipboard-list"), kpi_card("영향 용접 위치", "kpi-pages", "kpi-pages-detail", "lucide:map-pin"), kpi_card("미확인 경보", "kpi-unreviewed", "kpi-unreviewed-detail", "lucide:bell-ring", "danger")], className="kpi-grid"),
                    html.Div(
                        [
                            html.Section([html.Div([html.H2("실측 RealPower와 정상 기준"), html.Span("위·아래 그래프는 같은 행 구간을 공유합니다.", className="section-caption")], className="panel-header"), dcc.Graph(id="power-chart", config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}, className="power-chart"), html.Div([html.H3("모델 이상점수 및 판정 기준"), html.Span("점수 ÷ 임계값 · 1.0 이상 경보", className="section-caption")], className="score-title"), dcc.Graph(id="score-chart", config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}, className="score-chart")], className="panel evidence-panel"),
                            action_panel(),
                        ],
                        className="main-grid",
                    ),
                    html.Div(
                        [
                            html.Section([html.Div([html.Div([html.H2("반복 Cycle × PageNo 이상 강도"), html.Span("반복 Cycle은 PageNo 1→39 시퀀스 기반 파생", className="derived-note")]), html.Div([html.Span(className="legend-box normal"), "정상", html.Span(className="legend-box warning"), "주의", html.Span(className="legend-box danger"), "위험"], className="heat-legend")], className="panel-header"), dcc.Graph(id="heatmap-chart", config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})], className="panel heatmap-panel"),
                            html.Section([html.Div([html.H2("이벤트 관리대장"), html.Span(id="event-count", className="section-caption")], className="panel-header"), dag.AgGrid(id="event-grid", columnDefs=event_columns, rowData=[], defaultColDef={"sortable": True, "filter": True, "resizable": True}, dashGridOptions={"rowHeight": 38, "headerHeight": 38, "animateRows": False, "suppressCellFocus": True}, className="ag-theme-quartz event-grid")], className="panel event-panel"),
                        ],
                        className="bottom-grid",
                    ),
                    html.Div([html.Span("지도: 알려진 불량 유사성"), html.Span("비지도: 정상 패턴 이탈"), html.Span("제품 검사·정비 이력과 함께 판단")], className="page-footnote"),
                ],
                className="dashboard-container",
            ),
            dcc.Store(id="review-state-v3", storage_type="session", data={}), dcc.Store(id="selected-event", data=None), dcc.Download(id="download-events"),
            dmc.Modal(id="quality-modal", opened=False, title=None, size="lg", centered=True, children=html.Div(id="quality-content")),
            dmc.Modal(id="model-modal", opened=False, title=None, size="lg", centered=True, children=html.Div(id="model-content")),
            dmc.Drawer(id="event-drawer", opened=False, position="right", size=560, title="이벤트 상세", overlayProps={"backgroundOpacity": 0.42, "blur": 1}, children=html.Div(id="drawer-content")),
        ],
        className="app-shell",
    ),
)


@callback(
    Output("kpi-status", "children"), Output("kpi-status-detail", "children"), Output("kpi-ratio", "children"), Output("kpi-ratio-detail", "children"),
    Output("kpi-events", "children"), Output("kpi-events-detail", "children"), Output("kpi-pages", "children"), Output("kpi-pages-detail", "children"),
    Output("kpi-unreviewed", "children"), Output("kpi-unreviewed-detail", "children"), Output("power-chart", "figure"), Output("score-chart", "figure"), Output("heatmap-chart", "figure"),
    Output("event-grid", "rowData"), Output("event-count", "children"), Output("focus-event", "children"), Output("focus-status", "children"), Output("focus-status", "className"),
    Input("source-file", "value"), Input("supervised-model", "value"), Input("unsupervised-model", "value"), Input("review-state-v3", "data"),
)
def update_dashboard(source: str, supervised: str, unsupervised: str, review_state: dict | None):
    frame = combined_rows(source, supervised, unsupervised)
    events = events_for(frame, source, supervised, unsupervised, review_state)
    anomaly = frame[frame["combined_prediction"].eq(1)]
    pages = sorted(anomaly["PageNo"].unique().tolist())
    maximum = float(frame["risk_ratio"].max())
    max_duration = int(events["duration_rows"].max()) if not events.empty else 0
    status = severity(maximum, max_duration) if not events.empty else "정상"
    unreviewed = int(events["review_status"].eq("미확인").sum()) if not events.empty else 0
    if events.empty:
        focus = html.Div([html.Strong("탐지된 이상 이벤트 없음"), html.P("선택 구간은 두 모델의 판정 기준 이내입니다.")], className="focus-empty")
        focus_status, focus_class = "정상", "review-badge complete"
    else:
        selected = events.sort_values(["max_risk_ratio", "duration_rows"], ascending=False).iloc[0]
        focus = html.Div([html.Span(selected["severity"], className=f"severity-badge severity-{selected['severity']}"), html.Div([html.Strong(f"{selected['event_label']} · {selected['event_type']}"), html.P(f"PageNo {selected['pages']} · {selected['cycles']} 반복 Cycle · {selected['model_agreement']}")])], className="focus-row")
        focus_status = selected["review_status"]
        focus_class = "review-badge complete" if focus_status == "확인 완료" else "review-badge pending"
    rows = [] if events.empty else events.sort_values("max_risk_ratio", ascending=False).to_dict("records")
    return status, "선택한 두 모델의 OR 판정", f"{maximum:.2f}×", "두 모델 중 최대 score ÷ threshold", f"{len(events)}건", "연속 경보를 하나의 이벤트로 집계", f"{len(pages)}곳", ", ".join(map(str, pages[:8])) or "해당 없음", f"{unreviewed}건", f"전체 {len(events)}건 중 확인 전", power_figure(frame), score_figure(frame, supervised, unsupervised), heatmap_figure(frame), rows, f"총 {len(events)}건", focus, focus_status, focus_class


@callback(Output("event-drawer", "opened"), Output("selected-event", "data"), Input("event-grid", "cellRendererData"), Input("open-latest-event", "n_clicks"), State("source-file", "value"), State("supervised-model", "value"), State("unsupervised-model", "value"), State("review-state-v3", "data"), prevent_initial_call=True)
def open_event_drawer(cell_data, latest_clicks, source: str, supervised: str, unsupervised: str, review_state: dict | None):
    frame = combined_rows(source, supervised, unsupervised)
    events = events_for(frame, source, supervised, unsupervised, review_state)
    if events.empty:
        return False, None
    if callback_context.triggered_id == "event-grid" and cell_data:
        event_id = int(cell_data["data"]["event_id"])
    else:
        event_id = int(events.sort_values(["max_risk_ratio", "duration_rows"], ascending=False).iloc[0]["event_id"])
    return True, {"source": source, "supervised": supervised, "unsupervised": unsupervised, "event_id": event_id}


@callback(Output("drawer-content", "children"), Input("selected-event", "data"), Input("review-state-v3", "data"))
def render_drawer(selected: dict | None, review_state: dict | None):
    if not selected:
        return html.Div("이벤트를 선택하면 상세 정보가 표시됩니다.", className="drawer-empty")
    source, supervised, unsupervised = selected["source"], selected["supervised"], selected["unsupervised"]
    frame = combined_rows(source, supervised, unsupervised)
    events = events_for(frame, source, supervised, unsupervised, review_state)
    row = events[events["event_id"].eq(int(selected["event_id"]))].iloc[0]
    event = frame[frame["source_row"].between(int(row["start_row"]), int(row["end_row"]))]
    context = frame[frame["source_row"].between(max(int(frame["source_row"].min()), int(row["start_row"]) - 20), min(int(frame["source_row"].max()), int(row["end_row"]) + 20))]
    peak = event.loc[event["risk_ratio"].idxmax()]
    reviewed = row["review_status"] == "확인 완료"
    return html.Div(
        [
            html.Div([html.H2(f"이벤트 {row['event_label']} 상세"), html.Div([html.Span(row["severity"], className=f"severity-badge severity-{row['severity']}"), html.Span(row["review_status"], className=f"review-badge {'complete' if reviewed else 'pending'}")])], className="drawer-heading"),
            html.Section([html.H3("1. 발생 정보"), html.Div([field_pair("발생시각", str(row["start_time"])), field_pair("PageNo", row["pages"]), field_pair("반복 Cycle", f"{int(row['cycles'])}"), field_pair("지속 구간", f"{int(row['duration_rows'])} rows")], className="detail-grid")], className="drawer-section"),
            html.Section([html.H3("2. 신호와 판정"), html.Div([field_pair("실측 RealPower", f"{float(peak['RealPower']):,.1f} W"), field_pair("PageNo별 정상 기준값", f"{float(peak['expected_power']):,.1f} W"), field_pair("편차", f"{float(peak['RealPower'] - peak['expected_power']):+,.1f} W", "danger-text"), field_pair(f"{unsupervised} 이상점수", f"{float(peak['unsupervised_score']):.3f}"), field_pair(f"{unsupervised} 판정 기준", f"{float(peak['unsupervised_threshold']):.3f}"), field_pair(f"{supervised} 점수", f"{float(peak['supervised_score']):.3f}"), field_pair(f"{supervised} 임계값", f"{float(peak['supervised_threshold']):.3f}")], className="detail-grid signal-grid")], className="drawer-section"),
            html.Section([html.H3("3. 공정 조건"), html.Div([field_pair("SetPower", f"{float(peak['SetPower']):g}%"), field_pair("GateOnTime", f"{float(peak['GateOnTime']):g} · 단위 현장 확인"), field_pair("Speed", f"{float(peak['Speed']):g} mm/s"), field_pair("Length", f"{float(peak['Length']):g} mm")], className="detail-grid")], className="drawer-section"),
            html.Section([html.H3("4. 이벤트 구간 신호 추이"), dcc.Graph(figure=event_context_figure(context, int(row["start_row"]), int(row["end_row"])), config={"displayModeBar": False})], className="drawer-section drawer-chart"),
            html.Section([html.H3("5. 탐지 방식"), html.Div([field_pair("적용 모델", row["model_agreement"]), field_pair("판정 로직", "두 모델 중 하나가 임계값 이상이면 경보")], className="detail-grid one-column")], className="drawer-section"),
            html.Section([html.H3("6. 우선 확인 권장"), numbered_action(1, "이상 PageNo 현장 확인", "해당 용접 위치의 실제 상태를 확인합니다."), numbered_action(2, "해당 시점 설비 조건 확인", "설정값과 제어 신호 이상 유무를 확인합니다."), numbered_action(3, "해당 구간 생산품 품질 확인", "전후 생산품의 품질 결과를 확인합니다."), html.Div([icon("lucide:info", 15), "AI는 원인을 확정하지 않습니다."], className="drawer-disclaimer")], className="drawer-section"),
            html.Div([dmc.Select(data=["박정비", "김품질", "이생산"], value="박정비", allowDeselect=False, size="xs"), html.Span(row["review_status"], className=f"review-badge {'complete' if reviewed else 'pending'}"), dmc.Button("SOP 미연결", variant="outline", color="gray", size="xs", disabled=True), dmc.Button("확인 처리", id="confirm-drawer-review", n_clicks=0, color="indigo", size="xs", disabled=reviewed)], className="drawer-actions"),
        ],
        className="event-drawer-content",
    )


@callback(Output("review-state-v3", "data"), Input("confirm-drawer-review", "n_clicks"), Input("confirm-focus-review", "n_clicks"), State("selected-event", "data"), State("source-file", "value"), State("supervised-model", "value"), State("unsupervised-model", "value"), State("review-state-v3", "data"), prevent_initial_call=True)
def confirm_review(drawer_clicks, focus_clicks, selected: dict | None, source: str, supervised: str, unsupervised: str, review_state: dict | None):
    if callback_context.triggered_id == "confirm-drawer-review" and not drawer_clicks:
        return no_update
    if callback_context.triggered_id == "confirm-focus-review" and not focus_clicks:
        return no_update
    state = dict(review_state or {})
    if callback_context.triggered_id == "confirm-drawer-review" and selected:
        target = selected
    else:
        events = events_for(combined_rows(source, supervised, unsupervised), source, supervised, unsupervised, state)
        if events.empty:
            return no_update
        row = events.sort_values(["max_risk_ratio", "duration_rows"], ascending=False).iloc[0]
        target = {"source": source, "supervised": supervised, "unsupervised": unsupervised, "event_id": int(row["event_id"])}
    key = event_key(target["source"], target["supervised"], target["unsupervised"], int(target["event_id"]))
    state[key] = {"status": "확인 완료"}
    return state


@callback(Output("quality-modal", "opened"), Output("quality-content", "children"), Input("open-quality", "n_clicks"), State("source-file", "value"), prevent_initial_call=True)
def open_quality_modal(_, source: str):
    items = [html.Div([html.Div([html.Strong(label), html.Span(note)]), html.B(value)], className="quality-row") for label, value, note in quality_summary(source)]
    return True, html.Div([modal_header("lucide:database-check", "데이터 품질", f"{source} · 원본 파일 검사 결과"), html.Div(items, className="quality-list"), html.Div([html.Span("수신 상태"), html.Strong("시험 데이터 재생")], className="modal-foot")])


@callback(Output("model-modal", "opened"), Output("model-content", "children"), Input("open-model", "n_clicks"), State("supervised-model", "value"), State("unsupervised-model", "value"), prevent_initial_call=True)
def open_model_modal(_, supervised: str, unsupervised: str):
    selected = locked_metrics[locked_metrics["model"].isin([supervised, unsupervised])]
    cards = []
    for _, row in selected.iterrows():
        cards.append(html.Div([html.Div([html.Strong(row["model"]), html.Span("지도학습" if row["family"] == "supervised" else "비지도학습")], className="model-card-head"), html.Div([field_pair("Threshold", f"{float(row['threshold']):.4f}"), field_pair("Precision", f"{float(row['precision']):.4f}"), field_pair("Recall", f"{float(row['recall']):.4f}"), field_pair("F1", f"{float(row['f1']):.4f}"), field_pair("FN / FP", f"{int(row['fn'])} / {int(row['fp'])}"), field_pair("이벤트 Recall", f"{float(row['event_recall']):.4f}")], className="detail-grid"), html.P(f"임계값 출처: {row['threshold_source']}", className="model-source")], className="model-info-card"))
    return True, html.Div([modal_header("lucide:brain-circuit", "모델 정보", "독립 잠금 시험과 검증 단계에서 확정된 운영 근거"), html.Div(cards, className="model-info-grid"), dmc.Alert("현재 모델은 현시점 이상 탐지기입니다. RUL, 고장 시점 또는 물리적 원인을 예측하지 않습니다.", color="yellow", icon=icon("lucide:triangle-alert", 18))])


@callback(Output("download-events", "data"), Input("export-events", "n_clicks"), State("source-file", "value"), State("supervised-model", "value"), State("unsupervised-model", "value"), State("review-state-v3", "data"), prevent_initial_call=True)
def export_events(_, source: str, supervised: str, unsupervised: str, review_state: dict | None):
    events = events_for(combined_rows(source, supervised, unsupervised), source, supervised, unsupervised, review_state)
    return dcc.send_data_frame(events.to_csv, f"{source}_maintenance_events.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=int(os.environ.get("DASH_PORT", "8050")))
