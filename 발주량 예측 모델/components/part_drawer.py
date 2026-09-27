from __future__ import annotations

from dash import dcc, html
import dash_mantine_components as dmc

from .common import icon, status_badge


def metric(label: str, value_id: str, tone: str = "default", caption_id: str | None = None) -> html.Div:
    children = [html.Span(label, className="drawer-metric-label"), html.Strong(id=value_id, className=f"drawer-metric-value drawer-value-{tone}")]
    if caption_id:
        children.append(html.Span(id=caption_id, className="drawer-metric-caption"))
    return html.Div(children, className="drawer-metric")


def part_drawer() -> dmc.Drawer:
    return dmc.Drawer(
        id="part-drawer",
        opened=False,
        position="right",
        size=560,
        padding=0,
        withCloseButton=False,
        overlayProps={"backgroundOpacity": 0.44, "blur": 1},
        children=[
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H2(id="drawer-title", className="drawer-title"),
                                    html.Div(
                                        [
                                            status_badge("검토 필요", "danger", "drawer-status"),
                                            html.Span(id="drawer-target-date", className="drawer-date"),
                                        ],
                                        className="drawer-meta",
                                    ),
                                ]
                            ),
                            dmc.ActionIcon(icon("lucide:x", 20), id="close-drawer", variant="subtle", color="dark", size="lg", **{"aria-label": "상세 화면 닫기"}),
                        ],
                        className="drawer-header",
                    ),
                    html.Div(
                        [
                            html.Section(
                                [
                                    html.H3("핵심 요약", className="drawer-section-title"),
                                    html.Div(
                                        [
                                            metric("현재 D+3 계획", "drawer-plan"),
                                            metric("AI 예측", "drawer-forecast"),
                                            metric("차이", "drawer-gap", "danger"),
                                            metric("참고 상한량", "drawer-recommended", caption_id="drawer-buffer-caption"),
                                        ],
                                        className="drawer-metric-grid",
                                    ),
                                ],
                                className="drawer-section",
                            ),
                            html.Section(
                                [
                                    html.Div(
                                        [
                                            html.H3("최근 실제 발주 추세", className="drawer-section-title"),
                                            html.Span(id="drawer-trend-label", className="trend-label"),
                                        ],
                                        className="drawer-section-heading",
                                    ),
                                    dcc.Graph(id="drawer-trend-chart", config={"displayModeBar": False}, className="drawer-trend-chart"),
                                ],
                                className="drawer-section",
                            ),
                            html.Section(
                                [
                                    html.H3("예측 입력정보", className="drawer-section-title"),
                                    html.Div(id="drawer-input-table"),
                                ],
                                className="drawer-section",
                            ),
                            html.Section(
                                [
                                    html.Div(
                                        [
                                            html.H3("연관 부품 모델", className="drawer-section-title"),
                                            dmc.Switch(id="related-model-toggle", label="검증된 연관 부품 모델 사용", checked=False, color="teal", size="sm"),
                                        ],
                                        className="drawer-section-heading relation-heading",
                                    ),
                                    html.P(id="related-model-note", className="drawer-help"),
                                    html.Div(id="related-parts-list"),
                                    html.Div(id="related-model-comparison"),
                                    html.P("상관관계는 동일 제품/BOM 관계를 의미하지 않습니다.", className="drawer-caveat"),
                                ],
                                className="drawer-section",
                            ),
                        ],
                        className="drawer-body",
                    ),
                    html.Div(
                        [
                            dmc.Button("닫기", id="drawer-close-button", variant="outline", color="dark", className="drawer-action"),
                            dmc.Button("검토 상태 변경", id="review-status-button", color="indigo", className="drawer-action"),
                        ],
                        className="drawer-footer",
                    ),
                ],
                className="drawer-shell",
            )
        ],
    )
