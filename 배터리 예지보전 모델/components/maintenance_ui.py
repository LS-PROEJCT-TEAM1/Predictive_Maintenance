from __future__ import annotations

import dash_mantine_components as dmc
from dash import html
from dash_iconify import DashIconify


def icon(name: str, size: int = 18):
    return DashIconify(icon=name, width=size, height=size)


def app_header() -> html.Header:
    return html.Header(
        html.Div(
            [
                html.Div([html.Span("LS", className="brand-mark"), html.Span("제조 AI 운영센터", className="brand-name")], className="brand"),
                html.Nav(
                    [
                        html.Span("통합 현황", className="nav-item"),
                        html.Span("발주량 예측", className="nav-item"),
                        html.Span("예지보전", className="nav-item active"),
                        html.Span("품질보증", className="nav-item"),
                    ],
                    className="top-nav",
                ),
                html.Div([icon("lucide:bell", 18), html.Span("이현수", className="operator-name")], className="header-user"),
            ],
            className="header-inner",
        ),
        className="app-header",
    )


def kpi_card(label: str, value_id: str, detail_id: str, icon_name: str, tone: str = "default") -> html.Div:
    return html.Div(
        [
            html.Div([html.Span(label, className="kpi-label"), html.Span(icon(icon_name, 20), className="kpi-icon")], className="kpi-head"),
            html.Div(id=value_id, className=f"kpi-value kpi-{tone}"),
            html.Div(id=detail_id, className="kpi-detail"),
        ],
        className=f"kpi-card kpi-card-{tone}",
    )


def modal_header(icon_name: str, title: str, description: str) -> html.Div:
    return html.Div(
        [
            html.Span(icon(icon_name, 20), className="modal-heading-icon"),
            html.Div([html.H3(title), html.P(description)]),
        ],
        className="modal-heading",
    )


def field_pair(label: str, value, value_class: str = "") -> html.Div:
    return html.Div([html.Span(label), html.Strong(value, className=value_class)], className="detail-pair")


def numbered_action(number: int, title: str, description: str) -> html.Div:
    return html.Div(
        [html.Span(str(number), className="action-number"), html.Div([html.Strong(title), html.P(description)])],
        className="action-row",
    )
