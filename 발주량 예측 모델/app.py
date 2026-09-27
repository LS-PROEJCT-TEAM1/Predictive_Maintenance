from __future__ import annotations

import base64
import json
import io
import sys
from pathlib import Path

import dash_ag_grid as dag
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from inference import predict_records  # noqa: E402
from components.charts import forecast_figure, trend_figure  # noqa: E402
from components.common import app_header, decision_button, icon, kpi_card, status_badge  # noqa: E402
from components.part_drawer import part_drawer  # noqa: E402


DATA_DIR = ROOT / "outputs" / "dashboard_data" / "csv"
RELATED_DIR = ROOT / "outputs" / "related_part_model"
META = json.loads((ROOT / "outputs/audited_v2/metadata.json").read_text(encoding="utf-8"))
PRIMARY = META["selection"]["overall_method"]
AUXILIARY = META["selection"]["ml_model"]
RESIDUAL_BUFFER = META["uncertainty"]["buffer"]


def load_csv(name: str, *, directory: Path = DATA_DIR, dates: list[str] | None = None) -> pd.DataFrame:
    path = directory / name
    if not path.exists():
        raise FileNotFoundError(f"Dashboard data is missing: {path}")
    return pd.read_csv(path, parse_dates=dates or [])


PREDICTIONS = load_csv("predictions_test.csv", dates=["origin_date", "target_date"])
PART_INFO = load_csv("parts.csv")
DAILY_HISTORY = load_csv("daily_history.csv", dates=["date"])
WALK_METRICS = load_csv("walk_forward_metrics_pooled.csv")
RELATED_LATEST = load_csv("latest_predictions.csv", directory=RELATED_DIR, dates=["origin_date", "target_date"])
RELATED_METRICS = load_csv("part_model_metrics.csv", directory=RELATED_DIR)
RELATED_CANDIDATES = load_csv("related_part_candidates.csv", directory=RELATED_DIR)

PARTS = sorted(PREDICTIONS["part_number"].unique(), key=lambda value: int(value.split()[-1]))
TARGET_DATES = sorted(PREDICTIONS["target_date"].dt.strftime("%Y-%m-%d").unique())
RECOMMENDED_MODEL = PART_INFO.set_index("part_number")["recommended_model"].dropna().to_dict()


def fmt_quantity(value: float) -> str:
    return f"{value:,.0f}개"


def fmt_gap(value: float, percent: float) -> str:
    return f"{value:+,.0f}개 / {percent:+.1f}%" if np.isfinite(percent) else f"{value:+,.0f}개"


def with_operating_forecast(frame: pd.DataFrame) -> pd.DataFrame:
    current = frame.copy()
    current["operating_model"] = current["part_number"].map(RECOMMENDED_MODEL).fillna(PRIMARY)
    current["ai_forecast"] = [
        float(row[model]) if model in current.columns else float(row[PRIMARY])
        for (_, row), model in zip(current.iterrows(), current["operating_model"])
    ]
    current["plan"] = current["D+3 Plan Reference"].astype(float)
    return current


def trend_label(values: pd.Series) -> str:
    clean = values.astype(float).tail(7)
    if len(clean) < 2:
        return "→ 데이터 부족"
    slope = float(np.polyfit(np.arange(len(clean)), clean.to_numpy(), 1)[0])
    scale = max(float(clean.mean()), 1.0)
    if slope / scale > 0.03:
        return "↗ 상승"
    if slope / scale < -0.03:
        return "↘ 하락"
    return "→ 보합"


def latest_snapshot(reference_date: str, part_number: str = "ALL") -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = with_operating_forecast(PREDICTIONS[PREDICTIONS["target_date"] <= pd.Timestamp(reference_date)])
    if part_number != "ALL":
        prepared = prepared[prepared["part_number"] == part_number]
    latest = prepared[prepared["target_date"].eq(pd.Timestamp(reference_date))].copy()
    latest["gap_value"] = latest["ai_forecast"] - latest["plan"]
    latest["gap_pct"] = np.where(latest["plan"] > 0, latest["gap_value"] / latest["plan"] * 100, np.nan)
    threshold = np.maximum(10.0, latest["plan"] * 0.20)
    latest["review_required"] = latest["gap_value"].abs() >= threshold
    latest["recommended_quantity"] = np.ceil(latest["ai_forecast"] + RESIDUAL_BUFFER)
    return prepared, latest


def toolbar() -> html.Div:
    return html.Div(
        [
            html.Div(
                [html.Label("기준일", className="filter-label"), dmc.Select(id="date-filter", data=[{"label": date, "value": date} for date in TARGET_DATES], value=TARGET_DATES[-1], allowDeselect=False, size="sm")],
                className="filter-field filter-field-date",
            ),
            html.Div(
                [html.Label("부품", className="filter-label"), dmc.Select(id="part-filter", data=[{"label": f"전체 {len(PARTS)}개", "value": "ALL"}] + [{"label": part, "value": part} for part in PARTS], value="ALL", allowDeselect=False, searchable=True, size="sm")],
                className="filter-field",
            ),
            html.Div(
                [html.Label("예측 모델", className="filter-label"), dmc.Select(id="model-filter", data=[{"label": "권장 모델", "value": "recommended"}], value="recommended", allowDeselect=False, size="sm")],
                className="filter-field filter-field-model",
            ),
            html.Div(
                [html.Label("부품 검색", className="filter-label"), dmc.TextInput(id="part-search", placeholder="Part 번호 검색", leftSection=icon("lucide:search", 16), size="sm")],
                className="filter-search",
            ),
            html.Div(
                [
                    dmc.Button("CSV로 재예측", id="open-inference", leftSection=icon("lucide:refresh-cw", 16), variant="outline", color="indigo", className="toolbar-button"),
                    dmc.Button("검토 목록 내보내기", id="export-review", leftSection=icon("lucide:download", 16), color="indigo", className="toolbar-button toolbar-primary"),
                ],
                className="toolbar-actions",
            ),
        ],
        className="filter-toolbar",
    )


def performance_panel() -> html.Div:
    values = WALK_METRICS.set_index("Model")["MAE"].to_dict()
    moving = float(values[PRIMARY])
    xgboost = float(values[AUXILIARY])
    maximum = max(moving, xgboost, 1.0)
    return html.Div(
        [
            html.Div([html.H2("모델 성능 비교", className="rail-title"), html.Span("MAE · 낮을수록 우수", className="performance-caption")], className="rail-header"),
            html.Div([html.Span(PRIMARY, className="performance-name"), html.Div(html.Div(className="performance-bar", style={"width": f"{moving / maximum * 100:.1f}%"}), className="performance-track"), html.Span(f"{moving:.2f}", className="performance-value")], className="performance-row"),
            html.Div([html.Span(AUXILIARY, className="performance-name"), html.Div(html.Div(className="performance-bar performance-bar-secondary", style={"width": f"{xgboost / maximum * 100:.1f}%"}), className="performance-track"), html.Span(f"{xgboost:.2f}", className="performance-value")], className="performance-row"),
            html.Div([html.Span("운영 기본"), html.Span("학습형 보조")], className="performance-legend"),
        ],
        className="panel rail-panel",
    )


COLUMN_DEFS = [
    {"headerName": "우선순위", "field": "priority", "width": 82, "minWidth": 76, "maxWidth": 90, "cellStyle": {"fontWeight": 800, "color": "#0A1E5A"}},
    {"headerName": "부품", "field": "part_number", "width": 140, "cellStyle": {"fontWeight": 700}},
    {"headerName": "AI 예측", "field": "forecast_display", "width": 135},
    {"headerName": "D+3 계획", "field": "plan_display", "width": 130},
    {"headerName": "차이", "field": "difference_display", "width": 220, "cellClassRules": {"gap-positive": "params.data.gap_value > 0", "gap-negative": "params.data.gap_value < 0"}},
    {"headerName": "참고 상한량", "field": "recommended_display", "width": 150},
    {"headerName": "최근 추세", "field": "trend", "width": 130},
    {"headerName": "검토 상태", "field": "review_status", "width": 130, "cellClassRules": {"review-needed": "params.value === '검토 필요'"}},
    {"headerName": "상세", "field": "detail", "width": 104, "minWidth": 100, "maxWidth": 112, "sortable": False, "filter": False, "cellRenderer": "DetailButton"},
]


app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = "D+3 발주량 예측 | LS 제조 AI 운영센터"

app.layout = dmc.MantineProvider(
    theme={"fontFamily": 'Arial, "Noto Sans KR", sans-serif', "primaryColor": "indigo", "defaultRadius": "sm"},
    children=[
        dcc.Store(id="decision-filter-store", data="all"),
        dcc.Store(id="selected-part-store"),
        dcc.Store(id="review-status-store", data={}),
        dcc.Download(id="review-download"),
        app_header(),
        html.Main(
            html.Div(
                [
                    html.Div(
                        [
                            html.Div([html.H1("D+3 발주량 예측"), html.P("계획과 AI 예측의 차이를 검토하고 생산·구매 준비량에 반영합니다.")]),
                            html.Div([status_badge("데이터 정상", "success"), status_badge("검증 기준 적용", "normal")], className="page-status"),
                        ],
                        className="page-heading",
                    ),
                    toolbar(),
                    html.Div(
                        [
                            kpi_card("D+3 AI 예측", "kpi-forecast", "lucide:box", "kpi-forecast-caption"),
                            kpi_card("D+3 계획", "kpi-plan", "lucide:file-text", "kpi-plan-caption"),
                            kpi_card("계획 대비 증감", "kpi-gap", "lucide:chart-no-axes-column-increasing", "kpi-gap-caption", "red"),
                            kpi_card("참고 상한량", "kpi-recommended", "lucide:shield-check", "kpi-recommended-caption", "green"),
                            kpi_card("검토 필요", "kpi-review", "lucide:circle-alert", "kpi-review-caption", "red"),
                        ],
                        className="kpi-grid",
                    ),
                    html.Div(
                        [
                            html.Section(
                                [html.Div([html.H2("실제 발주 · AI 예측 · D+3 계획", className="panel-title"), html.Span("수량 단위: 개", className="panel-meta")], className="panel-header"), dcc.Graph(id="forecast-chart", config={"displayModeBar": False, "responsive": True}, className="forecast-chart")],
                                className="panel forecast-panel",
                            ),
                            html.Aside(
                                [
                                    performance_panel(),
                                    html.Div(
                                        [
                                            html.Div([html.H2("오늘의 판단", className="rail-title"), icon("lucide:info", 16)], className="rail-header"),
                                            html.P("선택하면 아래 검토 목록이 필터링됩니다.", className="decision-intro"),
                                            html.Div([decision_button("decision-above", "계획 상향 검토", "above-count", "above"), decision_button("decision-below", "계획 하향 검토", "below-count", "below")], className="decision-grid"),
                                        ],
                                        className="panel rail-panel",
                                    ),
                                ],
                                className="right-rail",
                            ),
                        ],
                        className="analysis-layout",
                    ),
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.Div([html.H2("계획 대비 우선 검토", className="panel-title"), html.Span(id="active-filter-chip", className="filter-chip"), dmc.Button("필터 초기화", id="reset-decision-filter", variant="outline", color="indigo", className="reset-filter")], className="review-heading-group"),
                                    html.Span(id="review-row-count", className="review-count"),
                                ],
                                className="review-panel-header",
                            ),
                            dag.AgGrid(
                                id="review-grid",
                                columnDefs=COLUMN_DEFS,
                                rowData=[],
                                defaultColDef={"sortable": True, "filter": True, "resizable": True, "suppressMovable": True},
                                dashGridOptions={"animateRows": False, "suppressHorizontalScroll": True, "getRowId": {"function": "params.data.part_number"}},
                                className="ag-theme-quartz ls-grid",
                            ),
                        ],
                        className="panel review-panel",
                    ),
                ],
                className="dashboard-container",
            ),
            className="dashboard-page",
        ),
        part_drawer(),
        dmc.Modal(
            id="inference-modal",
            opened=False,
            title="CSV로 D+3 재예측",
            size="lg",
            centered=True,
            children=[
                dmc.Text("동일 부품 3~60일 CSV를 업로드하세요. 최근 3일은 연속이어야 하며 8일 이력을 권장합니다.", size="sm", c="dimmed"),
                dmc.Text("연관 부품 모델은 Drawer에서 검증 완료된 부품에만 선택할 수 있습니다.", size="xs", c="dimmed", mt=4),
                dcc.Upload(id="inference-upload", children=html.Div([icon("lucide:upload", 18), " CSV 파일을 끌어 놓거나 선택"]), className="inference-upload"),
                dmc.Group([dmc.Button("닫기", id="close-inference", variant="outline", color="dark"), dmc.Button("예측 실행", id="run-inference", color="indigo")], justify="flex-end", mt="md"),
                html.Pre(id="inference-result", className="inference-result"),
            ],
        ),
    ],
)


@app.callback(
    Output("decision-filter-store", "data"),
    Input("decision-above", "n_clicks"), Input("decision-below", "n_clicks"), Input("reset-decision-filter", "n_clicks"),
    prevent_initial_call=True,
)
def choose_decision_filter(_: int, __: int, ___: int) -> str:
    triggered = callback_context.triggered_id
    if triggered == "decision-above":
        return "above"
    if triggered == "decision-below":
        return "below"
    return "all"


@app.callback(
    Output("kpi-forecast", "children"), Output("kpi-forecast-caption", "children"),
    Output("kpi-plan", "children"), Output("kpi-plan-caption", "children"),
    Output("kpi-gap", "children"), Output("kpi-gap-caption", "children"),
    Output("kpi-recommended", "children"), Output("kpi-recommended-caption", "children"),
    Output("kpi-review", "children"), Output("kpi-review-caption", "children"),
    Output("forecast-chart", "figure"), Output("review-grid", "rowData"),
    Output("above-count", "children"), Output("below-count", "children"),
    Output("review-row-count", "children"), Output("active-filter-chip", "children"),
    Output("active-filter-chip", "className"), Output("decision-above", "className"), Output("decision-below", "className"),
    Input("part-filter", "value"), Input("date-filter", "value"), Input("decision-filter-store", "data"),
    Input("part-search", "value"), Input("review-status-store", "data"),
)
def update_dashboard(part_number: str, reference_date: str, decision_filter: str, search: str | None, review_statuses: dict[str, str]):
    prepared, latest = latest_snapshot(reference_date, part_number)
    above_count = int((latest["gap_value"] > 0).sum())
    below_count = int((latest["gap_value"] < 0).sum())
    review_count = int(latest["review_required"].sum())
    total_forecast = float(latest["ai_forecast"].sum())
    total_plan = float(latest["plan"].sum())
    total_gap = total_forecast - total_plan
    total_gap_pct = total_gap / total_plan * 100 if total_plan else np.nan
    total_recommended = float(latest["recommended_quantity"].sum())
    total_buffer = total_recommended - total_forecast

    table = latest.copy()
    if decision_filter == "above":
        table = table[table["gap_value"] > 0]
    elif decision_filter == "below":
        table = table[table["gap_value"] < 0]
    if search:
        table = table[table["part_number"].str.contains(search.strip(), case=False, regex=False)]
    table = table.assign(abs_gap=table["gap_value"].abs()).sort_values(["review_required", "abs_gap"], ascending=[False, False])
    rows = []
    for priority, row in enumerate(table.itertuples(index=False), start=1):
        history = DAILY_HISTORY[(DAILY_HISTORY["part_number"] == row.part_number) & (DAILY_HISTORY["date"] <= row.target_date)].sort_values("date")
        default_status = "검토 필요" if row.review_required else "검토 대기"
        rows.append({
            "priority": priority, "part_number": row.part_number, "target_date": pd.Timestamp(row.target_date).date().isoformat(),
            "forecast_display": fmt_quantity(row.ai_forecast), "plan_display": fmt_quantity(row.plan),
            "difference_display": fmt_gap(row.gap_value, row.gap_pct), "gap_value": round(float(row.gap_value), 4),
            "recommended_display": fmt_quantity(row.recommended_quantity), "trend": trend_label(history["actual_d"]),
            "review_status": review_statuses.get(row.part_number, default_status), "detail": "상세 검토",
        })

    filter_labels = {"all": "", "above": "계획 상향 검토", "below": "계획 하향 검토"}
    filter_label = filter_labels.get(decision_filter, "")
    return (
        fmt_quantity(total_forecast), "검증 결과에 따른 운영 기본 예측",
        fmt_quantity(total_plan), f"기준일 {reference_date}",
        fmt_gap(total_gap, total_gap_pct), f"계획 {total_plan:,.0f} → 예측 {total_forecast:,.0f}",
        fmt_quantity(total_recommended), f"AI 예측 + CV 잔차 참고 상한 {total_buffer:,.0f}개",
        f"{review_count}개 Part", f"조회 대상 {len(latest)}개 중 {review_count}개",
        forecast_figure(prepared, part_number, RESIDUAL_BUFFER), rows,
        f"{above_count}건", f"{below_count}건", f"총 {len(rows)}개 부품", filter_label,
        "filter-chip is-visible" if filter_label else "filter-chip",
        "decision-button decision-above is-active" if decision_filter == "above" else "decision-button decision-above",
        "decision-button decision-below is-active" if decision_filter == "below" else "decision-button decision-below",
    )


@app.callback(
    Output("selected-part-store", "data"), Output("part-drawer", "opened"),
    Input("review-grid", "cellRendererData"), Input("close-drawer", "n_clicks"), Input("drawer-close-button", "n_clicks"),
    prevent_initial_call=True,
)
def control_drawer(cell_event: dict | None, _: int, __: int):
    triggered = callback_context.triggered_id
    if triggered in {"close-drawer", "drawer-close-button"}:
        return no_update, False
    if triggered == "review-grid" and cell_event:
        value = cell_event.get("value")
        if isinstance(value, dict) and value.get("part_number"):
            return value, True
        if isinstance(value, str):
            return {"part_number": value, "target_date": TARGET_DATES[-1]}, True
    return no_update, no_update


@app.callback(Output("related-model-toggle", "checked"), Input("selected-part-store", "data"), prevent_initial_call=True)
def reset_related_toggle(_: dict | None) -> bool:
    return False


@app.callback(
    Output("drawer-title", "children"), Output("drawer-target-date", "children"),
    Output("drawer-status", "children"), Output("drawer-status", "className"),
    Output("drawer-plan", "children"), Output("drawer-forecast", "children"),
    Output("drawer-gap", "children"), Output("drawer-recommended", "children"), Output("drawer-buffer-caption", "children"),
    Output("drawer-trend-label", "children"), Output("drawer-trend-chart", "figure"),
    Output("drawer-input-table", "children"), Output("related-parts-list", "children"),
    Output("related-model-comparison", "children"), Output("related-model-note", "children"), Output("related-model-toggle", "disabled"),
    Input("selected-part-store", "data"), Input("related-model-toggle", "checked"), Input("review-status-store", "data"),
)
def update_drawer(selection: dict | None, use_related: bool, review_statuses: dict[str, str]):
    if not selection:
        return ("Part 상세 분석", "", "검토 필요", "status-badge status-danger", "-", "-", "-", "-", "", "", trend_figure(pd.DataFrame()), "", "", "", "", True)
    part = selection["part_number"]
    target_date = pd.Timestamp(selection.get("target_date", TARGET_DATES[-1]))
    prepared = with_operating_forecast(PREDICTIONS[(PREDICTIONS["part_number"] == part) & (PREDICTIONS["target_date"] == target_date)])
    if prepared.empty:
        prepared = with_operating_forecast(PREDICTIONS[PREDICTIONS["part_number"] == part].sort_values("target_date").tail(1))
    row = prepared.iloc[0]
    related_row = RELATED_LATEST[(RELATED_LATEST["part_number"] == part) & (RELATED_LATEST["target_date"] == row["target_date"])]
    metric_row = RELATED_METRICS[RELATED_METRICS["part_number"] == part]
    enabled = bool(metric_row.iloc[0]["related_model_enabled"]) if not metric_row.empty else False
    related_available = enabled and not related_row.empty
    forecast = float(related_row.iloc[0]["related_prediction"]) if use_related and related_available else float(row["ai_forecast"])
    plan = float(row["plan"])
    gap = forecast - plan
    gap_pct = gap / plan * 100 if plan else np.nan
    recommended = float(np.ceil(forecast + RESIDUAL_BUFFER))
    history = DAILY_HISTORY[(DAILY_HISTORY["part_number"] == part) & (DAILY_HISTORY["date"] <= row["target_date"])].sort_values("date").tail(14)
    origin_data = DAILY_HISTORY[(DAILY_HISTORY["part_number"] == part) & (DAILY_HISTORY["date"] <= row["origin_date"])].sort_values("date").tail(3)
    latest_input = origin_data.iloc[-1] if not origin_data.empty else None
    recent_values = [fmt_quantity(value) for value in origin_data["actual_d"].tolist()]
    while len(recent_values) < 3:
        recent_values.insert(0, "-")
    input_table = html.Table([
        html.Thead(html.Tr([html.Th("구분"), html.Th("D+3 계획"), html.Th("D+4 계획"), html.Th("D+5 계획")])),
        html.Tbody([
            html.Tr([html.Td("계획 값"), html.Td(fmt_quantity(float(latest_input["plan_d3"])) if latest_input is not None else "-"), html.Td(fmt_quantity(float(latest_input["plan_d4"])) if latest_input is not None else "-"), html.Td(fmt_quantity(float(latest_input["plan_d5"])) if latest_input is not None else "-")]),
            html.Tr([html.Td("최근 실제 발주"), html.Td(recent_values[0]), html.Td(recent_values[1]), html.Td(recent_values[2])]),
        ]),
    ], className="input-table")
    relations = RELATED_CANDIDATES[RELATED_CANDIDATES["target_part"] == part].sort_values("rank")
    relation_table = html.Table([
        html.Thead(html.Tr([html.Th("연관 부품"), html.Th("관계 근거"), html.Th("데이터 근거")])),
        html.Tbody([html.Tr([html.Td(item.part_number), html.Td("발주량 상관 후보"), html.Td(f"{float(item.correlation):.2f} · {int(item.overlap_days)}일")]) for item in relations.itertuples(index=False)]),
    ], className="relation-table") if not relations.empty else html.P("표시할 연관 부품 후보가 없습니다.", className="drawer-help")
    if not metric_row.empty:
        metrics = metric_row.iloc[0]
        improvement = float(metrics["improvement_pct"])
        comparison = html.Div([
            html.Div([html.Span("단독 모델", className="model-compare-label"), html.Strong(f"MAE {float(metrics['baseline_mae']):.2f}", className="model-compare-value")], className="model-compare-block"),
            html.Span("→"),
            html.Div([html.Span("연관 모델", className="model-compare-label"), html.Strong(f"MAE {float(metrics['related_model_mae']):.2f}", className="model-compare-value")], className="model-compare-block"),
            html.Div([html.Span("개선율", className="model-compare-label"), html.Strong(f"{improvement:+.1f}%", className=f"model-compare-value {'model-improved' if improvement > 0 else 'model-worse'}")], className="model-compare-block"),
            status_badge("사용 가능" if enabled else "검증 미통과", "success" if enabled else "danger"),
        ], className="model-comparison-strip")
    else:
        comparison = html.Div("연관 모델 검증 결과가 없습니다.", className="drawer-help")
    current_status = review_statuses.get(part, "검토 필요" if abs(gap) >= max(10, plan * 0.2) else "검토 대기")
    status_tone = "success" if current_status == "검토 완료" else "danger"
    note = "전체 시간순 CV MAE 개선이 확인되어 재예측에 사용할 수 있습니다." if enabled else "전체 시간순 CV MAE 개선이 확인되지 않아 토글을 비활성화했습니다."
    return (
        f"{part} — D+3 발주량 분석", f"기준일(목표일) {pd.Timestamp(row['target_date']).date().isoformat()}",
        current_status, f"status-badge status-{status_tone}", fmt_quantity(plan), fmt_quantity(forecast), fmt_gap(gap, gap_pct), fmt_quantity(recommended),
        f"AI 예측 + CV 잔차 참고 상한 {RESIDUAL_BUFFER:.0f}개", trend_label(history["actual_d"]), trend_figure(history),
        input_table, relation_table, comparison, note, not related_available,
    )


@app.callback(Output("review-status-store", "data"), Input("review-status-button", "n_clicks"), State("selected-part-store", "data"), State("review-status-store", "data"), prevent_initial_call=True)
def mark_review_complete(_: int, selection: dict | None, statuses: dict[str, str]):
    if not selection:
        return no_update
    updated = dict(statuses or {})
    updated[selection["part_number"]] = "검토 완료"
    return updated


@app.callback(Output("inference-modal", "opened"), Input("open-inference", "n_clicks"), Input("close-inference", "n_clicks"), prevent_initial_call=True)
def control_inference_modal(_: int, __: int) -> bool:
    return callback_context.triggered_id == "open-inference"


@app.callback(Output("inference-result", "children"), Input("run-inference", "n_clicks"), State("inference-upload", "contents"), State("inference-upload", "filename"), prevent_initial_call=True)
def run_inference(_: int, contents: str | None, filename: str | None) -> str:
    if not contents:
        return "CSV 파일을 먼저 선택하세요."
    try:
        _, encoded = contents.split(",", 1)
        records = pd.read_csv(io.BytesIO(base64.b64decode(encoded)))
        result = predict_records(records)
        forecast = float(result["recommended_forecast"])
        return (f"파일: {filename}\n부품: {result['part_number']}\n목표일: {result['target_date']}\n권장 모델: {result['recommended_model']}\nAI 예측: {forecast:,.0f}개\n참고 상한량: {np.ceil(forecast + RESIDUAL_BUFFER):,.0f}개\nCV 잔차 상한 (보장 아님): {RESIDUAL_BUFFER:.2f}개")
    except Exception as error:
        return f"예측 실패: {error}"


@app.callback(Output("review-download", "data"), Input("export-review", "n_clicks"), State("review-grid", "rowData"), prevent_initial_call=True)
def export_review_rows(_: int, rows: list[dict[str, object]]):
    if not rows:
        return no_update
    export_columns = ["priority", "part_number", "target_date", "forecast_display", "plan_display", "difference_display", "recommended_display", "trend", "review_status"]
    return dcc.send_data_frame(pd.DataFrame(rows)[export_columns].to_csv, "d3_demand_review.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
