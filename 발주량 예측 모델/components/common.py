from __future__ import annotations

from dash import html
import dash_mantine_components as dmc
from dash_iconify import DashIconify


def icon(name: str, size: int = 18) -> DashIconify:
    return DashIconify(icon=name, width=size, height=size)


def status_badge(label: str, tone: str = "normal", badge_id: str | None = None) -> html.Span:
    props = {"className": f"status-badge status-{tone}"}
    if badge_id is not None:
        props["id"] = badge_id
    return html.Span(label, **props)


def app_header() -> html.Header:
    return html.Header(
        html.Div(
            [
                html.Div(
                    [html.Span("LS", className="brand-ls"), html.Span("제조 AI 운영센터", className="brand-name")],
                    className="brand-lockup",
                ),
                html.Nav(
                    [
                        html.Span("통합 현황", className="nav-item"),
                        html.Span("발주량 예측", className="nav-item nav-active"),
                        html.Span("예지보전", className="nav-item"),
                        html.Span("품질보증", className="nav-item"),
                    ],
                    className="global-nav",
                ),
                html.Div(
                    [
                        html.Span("마지막 갱신 2021.11.01", className="header-updated"),
                        status_badge("데이터 정상", "success"),
                        dmc.ActionIcon(icon("lucide:bell", 18), variant="subtle", color="dark", size="sm", **{"aria-label": "알림"}),
                        html.Span("김현우", className="user-name"),
                    ],
                    className="header-actions",
                ),
            ],
            className="header-inner",
        ),
        className="app-header",
    )


def kpi_card(
    label: str,
    value_id: str,
    icon_name: str,
    caption_id: str,
    tone: str = "blue",
) -> html.Div:
    return html.Div(
        [
            html.Div(icon(icon_name, 22), className=f"kpi-icon kpi-icon-{tone}"),
            html.Div(
                [
                    html.Div(label, className="kpi-label"),
                    html.Div(id=value_id, className=f"kpi-value kpi-value-{tone}"),
                    html.Div(id=caption_id, className="kpi-caption"),
                ],
                className="kpi-content",
            ),
        ],
        className="kpi-card",
    )


def decision_button(button_id: str, label: str, count_id: str, direction: str) -> html.Button:
    arrow_icon = "lucide:arrow-up" if direction == "above" else "lucide:arrow-down"
    return html.Button(
        [
            html.Span(icon(arrow_icon, 24), className="decision-icon"),
            html.Span(
                [
                    html.Span(label, className="decision-label"),
                    html.Strong(id=count_id, className="decision-count"),
                ],
                className="decision-copy",
            ),
            html.Span(icon("lucide:chevron-right", 18), className="decision-chevron"),
        ],
        id=button_id,
        className=f"decision-button decision-{direction}",
        n_clicks=0,
    )
