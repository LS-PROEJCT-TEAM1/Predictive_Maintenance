"""모든 탭이 같이 쓰는 화면 조각 — 사이드바, 탭바, 카드, 지표 카드, 필터 항목."""

from dash import dcc, html

from common.styles import (
    CARD_STYLE, CARD_TITLE_STYLE, FONT_NUM, GRAY, INK, LABEL_STYLE,
    LIGHT, TAG_STYLE, TEAL_500, WHITE,
)

SIDEBAR_MENUS = ["통합 현황", "공급망 예측", "품질 보증", "예지 보전"]


def make_sidebar(active_menu="예지 보전"):
    menu_items = []
    for menu in SIDEBAR_MENUS:
        is_active = menu == active_menu
        menu_items.append(
            html.Div(
                menu,
                style={
                    "padding": "11px 20px",
                    "fontSize": "14px",
                    "color": WHITE if is_active else "#C9C9CE",
                    "fontWeight": "700" if is_active else "400",
                    "backgroundColor": GRAY if is_active else INK,
                    "borderLeft": f"4px solid {WHITE if is_active else INK}",
                },
            )
        )

    return html.Aside(
        children=[
            html.Div(
                children=[
                    html.Div(
                        "BatteryFlow",
                        style={"fontSize": "20px", "fontWeight": "800", "color": WHITE},
                    ),
                    html.Div(
                        "AI CONTROL TOWER",
                        style={
                            "fontSize": "10px",
                            "letterSpacing": "1.5px",
                            "color": GRAY,
                            "marginTop": "2px",
                        },
                    ),
                ],
                style={"padding": "24px 20px 28px"},
            ),
            html.Nav(menu_items),
        ],
        style={
            "width": "210px",
            "flex": "0 0 210px",
            "backgroundColor": INK,
            "minHeight": "100vh",
        },
    )


def make_tabbar(tabs, value, tabs_id="pdm-tabs"):
    """tabs = [(value, label), ...] 목록으로 상단 탭바를 만든다."""
    tab_style = {
        "backgroundColor": INK,
        "color": WHITE,
        "border": "none",
        "borderBottom": f"3px solid {INK}",
        "padding": "0 18px",
        "lineHeight": "41px",
        "fontSize": "14px",
    }
    selected_style = {
        **tab_style,
        "color": TEAL_500,
        "fontWeight": "700",
        "borderBottom": f"3px solid {TEAL_500}",
    }

    return dcc.Tabs(
        id=tabs_id,
        value=value,
        children=[
            dcc.Tab(
                label=label,
                value=tab_value,
                style=tab_style,
                selected_style=selected_style,
            )
            for tab_value, label in tabs
        ],
        parent_style={"marginBottom": "14px"},
        style={
            "backgroundColor": INK,
            "borderRadius": "10px",
            "height": "44px",
            "overflow": "hidden",
            "padding": "0 8px",
        },
    )


def make_card(title=None, children=None, tag=None, extra=None, style=None, className=None):
    """흰 카드. title 옆에 회색 tag, 오른쪽 끝에 extra 를 둘 수 있다."""
    header = []
    if title is not None:
        header_left = [html.H3(title, style=CARD_TITLE_STYLE)]
        if tag:
            header_left.append(html.Span(tag, style=TAG_STYLE))
        header = [
            html.Div(
                children=[
                    html.Div(
                        header_left,
                        style={"display": "flex", "alignItems": "center", "gap": "8px"},
                    ),
                    html.Div(extra),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "marginBottom": "12px",
                },
            )
        ]

    body = children if isinstance(children, list) else [children]
    return html.Div(header + body, className=className, style={**CARD_STYLE, **(style or {})})


def make_metric_card(title, value, sub=None, style=None, value_color=INK):
    return html.Div(
        children=[
            html.P(title, style={"fontSize": "13px", "color": GRAY, "margin": "0 0 8px"}),
            html.Div(
                value,
                className="metric-value",
                style={
                    "fontFamily": FONT_NUM,
                    "fontWeight": "600",
                    "color": value_color,
                    "lineHeight": "1.2",
                },
            ),
            html.P(
                sub,
                style={"fontSize": "11px", "color": GRAY, "margin": "6px 0 0", "minHeight": "14px"},
            ),
        ],
        className="metric-card",
        style={**CARD_STYLE, **(style or {})},
    )


def filter_item(label, component, style=None):
    """필터 바 안의 '라벨 + 컨트롤' 한 칸."""
    return html.Div(
        children=[html.Label(label, style=LABEL_STYLE), component],
        style={"minWidth": "0", **(style or {})},
    )


def make_placeholder(label):
    return make_card(
        title=label,
        children=html.P(
            "이 탭은 아직 준비 중입니다.",
            style={"color": GRAY, "margin": "0", "fontSize": "13px"},
        ),
        style={"backgroundColor": WHITE},
    )


def make_empty_message(message="선택 조건에 해당하는 데이터가 없습니다."):
    return html.Div(
        message,
        style={
            "backgroundColor": LIGHT,
            "borderRadius": "8px",
            "padding": "24px",
            "textAlign": "center",
            "color": GRAY,
            "fontSize": "13px",
        },
    )
