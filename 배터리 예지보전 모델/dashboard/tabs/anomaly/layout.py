"""예지 보전 > 이상 탐지 탭 화면 구조 (Callback 로직은 callbacks.py)."""

from dash import dash_table, dcc, html

from common.layout import filter_item, make_card
from common.styles import (
    CARD_STYLE, FONT_BODY, FONT_NUM, GRAY, INK, LIGHT, TEAL_500, WHITE,
)
from tabs.anomaly import data

BUTTON_STYLE = {
    "border": "none",
    "borderRadius": "6px",
    "padding": "6px 12px",
    "fontSize": "12px",
    "fontWeight": "700",
    "cursor": "pointer",
    "fontFamily": FONT_BODY,
}


def make_threshold_control():
    sweep = data.get_threshold_sweep()
    default_q = data.get_default_quantile()
    default_index = int(sweep.index[sweep["calibration_quantile"] == default_q][0])
    last_index = len(sweep) - 1

    # 10단계 눈금 중 양 끝과 기본값에만 글자를 단다
    marks = {}
    for i, q in enumerate(sweep["calibration_quantile"]):
        label = f"{q * 100:g}" if i in (0, default_index, last_index) else " "
        marks[i] = {"label": label, "style": {"fontSize": "10px", "fontFamily": FONT_NUM}}

    return html.Div(
        children=[
            html.Div(
                children=[
                    html.Label("임계값 (정상 보정 분위수)", style={"fontSize": "12px", "color": GRAY}),
                    html.Span(id="anom-threshold-readout",
                              style={"fontFamily": FONT_NUM, "fontSize": "12px", "fontWeight": "600"}),
                ],
                style={"display": "flex", "justifyContent": "space-between", "marginBottom": "2px"},
            ),
            html.Div(
                children=[
                    html.Div(
                        dcc.Slider(
                            id="anom-threshold-slider",
                            min=0, max=last_index, step=None,
                            marks=marks, value=default_index,
                            allow_direct_input=False,
                        ),
                        style={"flex": "1", "minWidth": "0"},
                    ),
                    html.Button("적용", id="anom-apply-btn", n_clicks=0,
                                style={**BUTTON_STYLE, "backgroundColor": TEAL_500, "color": INK}),
                    html.Button("기본값", id="anom-reset-btn", n_clicks=0,
                                style={**BUTTON_STYLE, "backgroundColor": LIGHT, "color": INK}),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "6px"},
            ),
            dcc.Store(id="anom-threshold-store", data=default_q),
        ],
        style={"minWidth": "0", "borderLeft": f"1px solid {LIGHT}", "paddingLeft": "16px"},
    )


def make_filter_bar():
    min_date, max_date = data.get_date_range()
    return html.Div(
        children=[
            filter_item("파일", dcc.Dropdown(
                id="anom-file", options=data.get_file_options(),
                value=data.ALL_FILES, clearable=False, searchable=False,
            )),
            filter_item("기간", dcc.DatePickerRange(
                id="anom-date", start_date=min_date, end_date=max_date,
                min_date_allowed=min_date, max_date_allowed=max_date,
                display_format="YYYY-MM-DD",
            )),
            filter_item("SetPower", dcc.Dropdown(
                id="anom-setpower", options=data.get_setpower_options(),
                value=data.ALL, clearable=False, searchable=False,
            )),
            filter_item("PageNo", dcc.RangeSlider(
                id="anom-pageno", min=data.PAGE_MIN, max=data.PAGE_MAX, step=1,
                value=[data.PAGE_MIN, data.PAGE_MAX],
                marks={p: {"label": str(p), "style": {"fontSize": "10px", "fontFamily": FONT_NUM}}
                       for p in [1, 10, 20, 30, 39]},
                allow_direct_input=False,
                tooltip={"placement": "bottom"},
            )),
            make_threshold_control(),
        ],
        className="filter-bar anom-filter-grid",   # 열 배치는 CSS(화면 폭에 따라 변경)
        style=CARD_STYLE,
    )


def make_module_filter_bar():
    min_date, max_date = data.get_date_range()
    return html.Div(
        children=[
            html.Div(
                children=[
                    html.Span("판정", style={"fontSize": "12px", "color": GRAY, "marginRight": "8px"}),
                    dcc.Checklist(
                        id="anom-judge-chips",
                        options=[],
                        value=data.DEFAULT_JUDGES,
                        inline=True,
                        className="chip-group",
                    ),
                    dcc.Store(id="anom-judge-prev", data=data.DEFAULT_JUDGES),
                ],
                style={"display": "flex", "alignItems": "center"},
            ),
            html.Div(
                dcc.Dropdown(id="anom-module-file", options=data.get_file_options(),
                             value=data.ALL_FILES, clearable=False, searchable=False),
                style={"width": "140px"},
            ),
            html.Div(
                dcc.DatePickerRange(
                    id="anom-module-date", start_date=min_date, end_date=max_date,
                    min_date_allowed=min_date, max_date_allowed=max_date,
                    display_format="YYYY-MM-DD",
                ),
                style={"width": "230px"},
            ),
            dcc.Input(id="anom-module-search", type="text", placeholder="모듈 번호 검색",
                      debounce=True, className="chip-input-text"),
            html.Button("모듈 목록 다운로드 ↓", id="anom-download-btn", n_clicks=0,
                        style={**BUTTON_STYLE, "backgroundColor": INK, "color": WHITE,
                               "padding": "8px 14px", "marginLeft": "auto"}),
            dcc.Download(id="anom-download"),
        ],
        className="filter-bar",
        style={
            **CARD_STYLE,
            "display": "flex",
            "flexWrap": "wrap",
            "alignItems": "center",
            "gap": "12px",
        },
    )


def make_module_table():
    columns = [
        {"name": "판정", "id": "judge_badge", "presentation": "markdown"},
        {"name": "파일", "id": "file_short"},
        {"name": "모듈", "id": "module_label"},
        {"name": "시작 시각", "id": "start_label"},
        {"name": "이상 수", "id": "anomaly_label"},
        {"name": "0W", "id": "zero_power_points"},
        {"name": "최대 점수", "id": "max_score_label"},
        {"name": "39포인트 지도 (PageNo 1 → 39)", "id": "page_map", "presentation": "markdown"},
    ]
    return dash_table.DataTable(
        id="anom-module-table",
        columns=columns,
        data=[],
        page_action="none",          # 페이지 대신 카드 안에서 스크롤 (카드 높이 = 남은 화면 높이)
        markdown_options={"html": True},
        style_as_list_view=True,
        style_header={
            "backgroundColor": WHITE, "color": GRAY, "fontWeight": "600",
            "fontSize": "12px", "borderBottom": f"1px solid {LIGHT}",
            "fontFamily": FONT_BODY, "textAlign": "left",
        },
        style_cell={
            "fontFamily": FONT_BODY, "fontSize": "13px", "color": INK,
            "textAlign": "left", "padding": "6px 10px",
            "borderBottom": f"1px solid {LIGHT}", "cursor": "pointer",
        },
        style_cell_conditional=[
            {"if": {"column_id": c}, "fontFamily": FONT_NUM, "fontSize": "12px"}
            for c in ["module_label", "start_label", "anomaly_label", "zero_power_points", "max_score_label"]
        ] + [{"if": {"column_id": "page_map"}, "width": "360px", "minWidth": "360px"}],
        style_data_conditional=[],
    )


def make_map_legend():
    def item(color, text, border=None):
        return html.Span(
            children=[
                html.Span(style={
                    "display": "inline-block", "width": "10px", "height": "10px",
                    "backgroundColor": color, "borderRadius": "2px", "marginRight": "5px",
                    "border": border or "none", "verticalAlign": "middle",
                }),
                text,
            ],
            style={"marginRight": "14px"},
        )

    return html.Div(
        children=[
            html.Div([item(TEAL_500, "이상 포인트"), item(INK, "미출력(0W)"), item(LIGHT, "정상")]),
            html.Div("정렬: 불량 의심 → 재검사 권고 → 정상"),
        ],
        style={"display": "flex", "justifyContent": "space-between", "fontSize": "11px",
               "color": GRAY, "marginTop": "10px"},
    )


def make_threshold_table():
    return dash_table.DataTable(
        id="anom-threshold-table",
        columns=[
            {"name": "분위수", "id": "quantile_label"},
            {"name": "임계값", "id": "threshold_label"},
            {"name": "FP", "id": "fp"},
            {"name": "오경보 이벤트", "id": "normal_false_alarm_events"},
        ],
        data=[],
        style_as_list_view=True,
        style_header={"backgroundColor": WHITE, "color": GRAY, "fontSize": "11px",
                      "fontWeight": "600", "fontFamily": FONT_BODY, "borderBottom": f"1px solid {LIGHT}"},
        style_cell={"fontFamily": FONT_NUM, "fontSize": "12px", "color": INK, "padding": "4px 8px",
                    "textAlign": "right", "borderBottom": f"1px solid {LIGHT}"},
        style_data_conditional=[],
    )


def create_layout():
    """화면 1: 필터 · 지표 · 모듈 판정(양품/불량 찾기) — 브라우저 한 화면 높이에 맞춤
       화면 2: 선택 모듈 · 혼동행렬 · 조건별 비율 · 임계값 — 스크롤하면 한 화면에 모두 보임
       높이·열 배치는 assets/dashboard.css 의 anom-* 클래스가 화면 크기에 따라 조절한다."""
    section_title = html.H2("모듈(배터리 1개) 판정 · 양품/불량 찾기", className="anom-section-title")

    screen_1 = html.Section(
        children=[
            make_filter_bar(),
            html.Div(
                children=[
                    html.Div(id="anom-metrics", className="anom-metric-grid"),
                    html.Div(id="anom-metric-note",
                             style={"fontSize": "11px", "color": GRAY, "minHeight": "14px",
                                    "marginTop": "4px"}),
                ],
            ),
            section_title,
            make_module_filter_bar(),
            html.Div(id="anom-module-summary", className="anom-summary-grid"),
            make_card(
                children=[
                    html.Div(make_module_table(), className="anom-table-wrap"),
                    make_map_legend(),
                ],
                className="anom-table-card",
            ),
        ],
        className="anom-screen anom-screen-1",
    )

    screen_2 = html.Section(
        children=[
            html.Div(id="anom-selected-module", style=CARD_STYLE),
            make_card("혼동행렬", tag="confusion_matrix", children=html.Div(id="anom-confusion")),
            make_card(
                "공정 조건별 이상 판정 비율", tag="px.bar · 이상 행 ÷ 조건 행",
                children=dcc.Graph(id="anom-condition-graph", className="anom-graph",
                                   responsive=True, config={"displayModeBar": False}),
                className="anom-chart-card",
            ),
            make_card(
                "임계값별 Precision · Recall · F1", tag="go.Scatter · 전체 테스트",
                children=[
                    dcc.Graph(id="anom-threshold-graph", className="anom-graph",
                              responsive=True, config={"displayModeBar": False}),
                    make_threshold_table(),
                ],
                className="anom-chart-card",
            ),
        ],
        className="anom-screen anom-screen-2",
    )

    return html.Div([screen_1, screen_2])
