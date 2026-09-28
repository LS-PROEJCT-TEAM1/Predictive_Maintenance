from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
import dash_ag_grid as dag
import dash_mantine_components as dmc

from components.charts import evidence_figure, location_figure
from components.ui import decision_modal, header, icon, kpi_card, model_drawer, status_badge, toolbar


BASE = Path(__file__).resolve().parent
SCORE_PATH = BASE / "output" / "models" / "model_C_pca_t2_spe_이상점수.csv"
RANK_PATH = BASE / "output" / "models" / "model_C_pca_t2_spe_이상셀순위.csv"
RAW_DIR = BASE / "data" / "preprocessed" / "test"

SCORES = pd.read_csv(SCORE_PATH)
RANKS = pd.read_csv(RANK_PATH)
TESTS = SCORES["파일"].drop_duplicates().tolist()
TEST_OPTIONS = [{"label": t, "value": t} for t in TESTS]


@lru_cache(maxsize=12)
def load_raw(test_id: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / f"{test_id}.csv")


def test_scores(test_id: str) -> pd.DataFrame:
    return SCORES.loc[SCORES["파일"] == test_id].reset_index(drop=True)


def test_ranks(test_id: str) -> pd.DataFrame:
    return RANKS.loc[RANKS["파일"] == test_id].sort_values("순위").reset_index(drop=True)


def anomaly_regions(prediction) -> int:
    values = np.asarray(prediction, dtype=int)
    return int(((values == 1) & ~np.r_[False, values[:-1] == 1]).sum())


def selected_cell_for(test_id: str) -> str:
    ranks = test_ranks(test_id)
    return str(ranks.iloc[0]["셀"]) if not ranks.empty else "M01CV01"


def action_timeline(test_id: str, decision: dict | None = None):
    scores = test_scores(test_id)
    has_alert = bool(scores["예측"].sum())
    items = [
        ("시험 시작", f"{len(scores):,}개 시점 분석", "neutral"),
        ("AI 검토 필요" if has_alert else "AI 판정 정상", "PCA 관리한계 기준 적용", "danger" if has_alert else "success"),
    ]
    if decision and decision.get("test_id") == test_id:
        items.append((decision["label"], decision.get("note") or "작업자 판정 기록", "danger" if decision["code"] == "hold" else "info"))
    else:
        items.append(("작업자 판정 대기", "작업자 검토와 최종 판정 필요", "pending"))
    return html.Div(
        [html.Div([html.Span(className=f"timeline-dot timeline-{tone}"), html.Div([html.Strong(title), html.P(desc)])], className="timeline-item") for title, desc, tone in items],
        className="timeline",
    )


def action_panel():
    return html.Section(
        [
            html.H2("판정 근거와 조치", className="section-title"),
            html.Div([icon("lucide:circle-alert", 19), html.Div([html.Strong(id="ai-action-title"), html.Span(id="ai-action-copy")])], id="ai-action-banner", className="action-banner"),
            html.Div(
                [
                    html.Div([html.Span("SPE 초과 시점"), html.Strong(id="spe-evidence"), html.Small("한계 2.046")], className="evidence-box"),
                    html.Div([html.Span("T² 초과 시점"), html.Strong(id="t2-evidence"), html.Small("한계 20.427")], className="evidence-box"),
                ],
                className="evidence-grid",
            ),
            html.H3("우선 확인 권장", className="subsection-title"),
            html.Ol(
                [
                    html.Li([html.Span("1"), html.Div([html.Strong(id="recommend-module"), html.Small("하네스·센싱선 연결 상태")])]),
                    html.Li([html.Span("2"), html.Div([html.Strong(id="recommend-cell"), html.Small("원본 측정값 및 접점")])]),
                    html.Li([html.Span("3"), html.Div([html.Strong(id="recommend-step3"), html.Small(id="recommend-step3-meta")])]),
                ],
                className="recommend-list",
            ),
            html.Div([icon("lucide:info", 16), html.Span("AI는 원인을 확정하지 않으며 검사 우선순위를 지원합니다.")], className="caveat"),
            html.Div(
                [
                    html.Div([html.H3("작업자 최종 판정"), html.Div([html.Span("담당"), html.Strong("김품질"), html.Span("판정"), status_badge("미확정", "pending", "decision-badge")], className="decision-meta")], className="decision-heading"),
                    html.Div(
                        [
                            dmc.Button("이상 없음", id="decision-clear", variant="outline", size="xs"),
                            dmc.Button("재시험 요청", id="decision-retest", variant="outline", size="xs"),
                            dmc.Button("출하 보류", id="decision-hold", variant="outline", color="red", size="xs"),
                        ],
                        className="decision-buttons",
                    ),
                ],
                className="decision-zone",
            ),
        ],
        className="panel action-panel",
    )


def rank_grid():
    return dag.AgGrid(
        id="rank-grid",
        columnDefs=[
            {"field": "순위", "width": 62, "suppressSizeToFit": True},
            {"field": "셀", "headerName": "셀 ID", "flex": 1},
            {"field": "이상점수", "headerName": "Z-score", "flex": 1,
             "cellStyle": {"styleConditions": [{"condition": "params.value >= 3.33", "style": {"color": "#FA002D", "fontWeight": "700"}}]}},
        ],
        defaultColDef={"sortable": True, "resizable": False, "suppressMovable": True},
        dashGridOptions={"headerHeight": 34, "rowHeight": 38},
        className="ag-theme-quartz rank-grid",
        style={"height": "230px"},
    )


app = Dash(__name__, title="LS 제조 AI 운영센터 · 품질보증", update_title=None, suppress_callback_exceptions=True)
server = app.server

app.layout = dmc.MantineProvider(
    theme={"primaryColor": "lsblue", "colors": {"lsblue": ["#F0F3FA", "#DCE4F3", "#B8C8E5", "#8EA6D3", "#607FBF", "#3358AA", "#0A1E5A", "#08184A", "#061239", "#040C29"]}, "fontFamily": "Arial, Malgun Gothic, sans-serif"},
    children=[
        dcc.Store(id="selected-cell", data="M02CV01"),
        dcc.Store(id="pending-decision"),
        dcc.Store(id="worker-decision"),
        dcc.Download(id="download-results"),
        header(),
        html.Main(
            [
                html.Div([html.H1("배터리 품질보증"), html.P("시험 데이터에서 이상 신호와 기여 셀을 확인하고 최종 출하 판단을 기록합니다.")], className="page-heading"),
                toolbar(TEST_OPTIONS),
                html.Div(
                    [
                        kpi_card("AI 품질 판정", "kpi-ai", meta_id="kpi-ai-meta", tone="danger"),
                        kpi_card("이상 구간", "kpi-regions"),
                        kpi_card("최대 셀 위치점수", "kpi-cell-score", meta_id="kpi-cell-score-meta", tone="danger"),
                        kpi_card("우선 확인 모듈", "kpi-module"),
                        kpi_card("우선 확인 셀", "kpi-cell"),
                        kpi_card("작업자 판정", "kpi-decision", tone="pending"),
                    ],
                    className="kpi-strip",
                ),
                html.Div(
                    [
                        html.Section(
                            [
                                html.Div(
                                    [
                                        html.H2("시험 진행률별 이상 근거", className="section-title"),
                                        html.Div(
                                            [
                                                html.Span([html.I(className="legend-line legend-module"), "선택 모듈"]),
                                                html.Span([html.I(className="legend-line legend-cell"), "선택 셀"]),
                                                html.Span([html.I(className="legend-line legend-spe"), "SPE 비율"]),
                                                html.Span([html.I(className="legend-line legend-t2"), "T² 비율"]),
                                                html.Span([html.I(className="legend-dot"), "관리한계 초과"]),
                                            ],
                                            className="evidence-legend",
                                        ),
                                    ],
                                    className="panel-header evidence-panel-header",
                                ),
                                dcc.Graph(id="evidence-chart", config={"displayModeBar": False}, className="evidence-chart"),
                            ],
                            className="panel evidence-panel",
                        ),
                        action_panel(),
                    ],
                    className="main-grid",
                ),
                html.Div(
                    [
                        html.Section(
                            [html.Div([html.H2("기여 위치 지도", className="section-title"), dmc.SegmentedControl(id="location-mode", value="cell", data=[{"label": "셀 전압 16×11", "value": "cell"}, {"label": "모듈 온도 16×2", "value": "temperature"}], size="xs")], className="panel-header"),
                             dcc.Graph(id="location-map", config={"displayModeBar": False}, className="location-map")],
                            className="panel location-panel",
                        ),
                        html.Section([html.H2("우선 확인 셀", className="section-title"), rank_grid()], className="panel rank-panel"),
                        html.Section([html.Div([html.H2("검사·조치 이력", className="section-title"), html.Button("전체 이력 보기", className="text-button")], className="panel-header"), html.Div(id="timeline-content"), html.Div(id="test-meta", className="test-meta")], className="panel history-panel"),
                    ],
                    className="bottom-grid",
                ),
            ],
            className="dashboard-container",
        ),
        model_drawer(),
        decision_modal(),
    ],
)


@app.callback(
    Output("selected-cell", "data"),
    Input("test-select", "value"),
    Input("location-map", "clickData"),
    Input("rank-grid", "cellClicked"),
    State("location-mode", "value"),
    prevent_initial_call=True,
)
def choose_cell(test_id, heat_click, cell_clicked, mode):
    trigger = ctx.triggered_id
    if trigger == "rank-grid" and cell_clicked:
        return cell_clicked["data"]["셀"]
    if trigger == "location-map" and heat_click and mode == "cell":
        p = heat_click["points"][0]
        return f"{p['y']}CV{int(p['x']):02d}"
    return selected_cell_for(test_id)


@app.callback(
    Output("process-value", "children"), Output("progress-label", "children"), Output("progress-fill", "style"),
    Output("kpi-ai", "children"), Output("kpi-ai", "className"), Output("kpi-ai-meta", "children"), Output("kpi-regions", "children"),
    Output("kpi-cell-score", "children"), Output("kpi-cell-score", "className"), Output("kpi-cell-score-meta", "children"), Output("kpi-module", "children"),
    Output("kpi-cell", "children"), Output("kpi-decision", "children"), Output("evidence-chart", "figure"),
    Output("location-map", "figure"), Output("rank-grid", "rowData"), Output("ai-action-title", "children"),
    Output("ai-action-copy", "children"), Output("ai-action-banner", "className"), Output("spe-evidence", "children"),
    Output("t2-evidence", "children"), Output("recommend-module", "children"), Output("recommend-cell", "children"),
    Output("recommend-step3", "children"), Output("recommend-step3-meta", "children"),
    Output("decision-badge", "children"), Output("decision-badge", "className"), Output("timeline-content", "children"),
    Output("test-meta", "children"),
    Input("test-select", "value"), Input("selected-cell", "data"), Input("location-mode", "value"), Input("worker-decision", "data"),
)
def render_dashboard(test_id, selected_cell, location_mode, decision):
    raw = load_raw(test_id)
    scores = test_scores(test_id)
    ranks = test_ranks(test_id)
    if selected_cell not in raw.columns:
        selected_cell = selected_cell_for(test_id)
    process = str(scores.iloc[0]["충방전"])
    has_alert = bool(scores["예측"].sum())
    regions = anomaly_regions(scores["예측"])
    top_cell = str(ranks.iloc[0]["셀"]) if not ranks.empty else "—"
    top_score = float(ranks.iloc[0]["이상점수"]) if not ranks.empty else 0
    module = top_cell[:3] if top_cell != "—" else "—"
    decision_label = decision["label"] if decision and decision.get("test_id") == test_id else "미확정"
    spe_points = int((scores["SPE"] > 2.046).sum())
    t2_points = int((scores["Hotelling_T2"] > 20.427).sum())
    ai_text = "검토 필요" if has_alert else "정상 범위"
    ai_meta = "작업자 확정 전" if has_alert else "관리한계 이내"
    banner_class = "action-banner action-danger" if has_alert else "action-banner action-success"
    banner_copy = "PCA 관리한계 초과 구간이 감지되었습니다." if has_alert else "연속 관리한계 초과 구간이 없습니다."
    badge_tone = "danger" if decision_label == "출하 보류" else ("info" if decision_label != "미확정" else "pending")
    rows = ranks[["순위", "셀", "이상점수"]].to_dict("records")
    return (
        process, f"{len(scores):,} / {len(scores):,}", {"width": "100%"},
        ai_text, f"kpi-value {'kpi-danger' if has_alert else 'kpi-success'}", ai_meta, f"{regions}개",
        f"{top_score:.2f}σ" if top_score else "—", f"kpi-value {'kpi-danger' if top_score >= 3.33 else ''}", "셀 Z-score · 위치 추적용",
        module, top_cell, decision_label, evidence_figure(raw, scores, selected_cell), location_figure(ranks, raw, location_mode, selected_cell), rows,
        f"AI {ai_text}", banner_copy, banner_class, f"{spe_points:,}시점", f"{t2_points:,}시점",
        f"{module} 하네스·센싱선 확인" if module != "—" else "시험 데이터 확인", f"{top_cell} 측정값 재확인" if top_cell != "—" else "원본 측정값 확인",
        "동일 조건 재시험" if has_alert else "표준 출하 검사 계속", "동일 공정에서 재현 여부" if has_alert else "기준 절차에 따라 완료",
        decision_label, f"status-badge status-{badge_tone}", action_timeline(test_id, decision),
        [html.Span("실제 판정"), html.Strong(str(scores.iloc[0]["판정"])), html.Span("공정"), html.Strong(process)],
    )


@app.callback(
    Output("model-drawer", "opened"),
    Input("open-model", "n_clicks"), Input("close-model", "n_clicks"),
    State("model-drawer", "opened"), prevent_initial_call=True,
)
def toggle_model_drawer(open_clicks, close_clicks, opened):
    return ctx.triggered_id == "open-model" if ctx.triggered_id else opened


DECISIONS = {
    "decision-clear": ("clear", "이상 없음", "이상 없음 판정"),
    "decision-retest": ("retest", "재시험 요청", "재시험 요청"),
    "decision-hold": ("hold", "출하 보류", "출하 보류 판정"),
}


@app.callback(
    Output("decision-modal", "opened"), Output("pending-decision", "data"), Output("decision-modal-title", "children"),
    Output("worker-decision", "data"), Output("decision-note", "value"),
    Input("decision-clear", "n_clicks"), Input("decision-retest", "n_clicks"), Input("decision-hold", "n_clicks"),
    Input("cancel-decision", "n_clicks"), Input("confirm-decision", "n_clicks"),
    State("pending-decision", "data"), State("test-select", "value"), State("decision-note", "value"), State("worker-decision", "data"),
    prevent_initial_call=True,
)
def handle_decision(clear, retest, hold, cancel, confirm, pending, test_id, note, current):
    trigger = ctx.triggered_id
    if trigger in DECISIONS:
        code, label, title = DECISIONS[trigger]
        return True, {"code": code, "label": label}, title, current, ""
    if trigger == "cancel-decision":
        return False, None, no_update, current, ""
    if trigger == "confirm-decision" and pending:
        saved = {**pending, "test_id": test_id, "note": note or ""}
        return False, None, no_update, saved, ""
    return no_update, no_update, no_update, current, no_update


@app.callback(
    Output("download-results", "data"),
    Input("export-button", "n_clicks"), Input("export-model", "n_clicks"),
    State("test-select", "value"), prevent_initial_call=True,
)
def export_results(export_clicks, model_clicks, test_id):
    if ctx.triggered_id == "export-model":
        frame = pd.read_csv(BASE / "output" / "models" / "00_모델별_성능비교.csv")
        return dcc.send_data_frame(frame.to_csv, "quality_model_evaluation.csv", index=False)
    return dcc.send_data_frame(test_scores(test_id).to_csv, f"{test_id}_pca_scores.csv", index=False)


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8065)
