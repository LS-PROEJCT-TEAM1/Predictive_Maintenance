from dash import html
import dash_mantine_components as dmc
from dash_iconify import DashIconify


def icon(name: str, size: int = 18):
    return DashIconify(icon=name, width=size, height=size)


def status_badge(text: str, tone: str = "neutral", badge_id=None):
    props = {"className": f"status-badge status-{tone}"}
    if badge_id is not None:
        props["id"] = badge_id
    return html.Span(text, **props)


def kpi_card(label: str, value_id: str, *, meta_id=None, tone="default"):
    return html.Div(
        [
            html.Div(label, className="kpi-label"),
            html.Div(id=value_id, className=f"kpi-value kpi-{tone}"),
            html.Div(id=meta_id, className="kpi-meta") if meta_id else None,
        ],
        className="kpi-card",
    )


def header():
    return html.Header(
        html.Div(
            [
                html.Div(
                    [html.Span("LS", className="ls-mark"), html.Span("제조 AI 운영센터")],
                    className="brand",
                ),
                html.Nav(
                    [
                        html.A("통합 현황"),
                        html.A("발주량 예측"),
                        html.A("예지보전"),
                        html.A("품질보증", className="active"),
                    ],
                    className="top-nav",
                ),
                html.Div(
                    [
                        html.Span(className="live-dot"),
                        html.Span("분석 데이터 연결됨", className="header-state"),
                        html.Span(icon("lucide:bell", 18), className="header-icon"),
                        html.Span(icon("lucide:circle-user-round", 22), className="avatar-icon"),
                        html.Span("김품질", className="user-name"),
                        icon("lucide:chevron-down", 15),
                    ],
                    className="header-user",
                ),
            ],
            className="header-inner",
        ),
        className="app-header",
    )


def toolbar(test_options):
    return html.Div(
        [
            html.Div(
                [html.Span("시험 ID", className="control-label"),
                 dmc.Select(id="test-select", data=test_options, value="Test07_NG_dchg",
                            allowDeselect=False, size="xs", className="test-select")],
                className="toolbar-field",
            ),
            html.Div([html.Span("공정", className="control-label"), html.Strong(id="process-value")], className="toolbar-field compact-field"),
            html.Div(
                [html.Span("분석 시점", className="control-label"), html.Strong(id="progress-label"),
                 html.Div(html.Div(id="progress-fill", className="progress-fill"), className="progress-track")],
                className="toolbar-field progress-field",
            ),
            html.Div([html.Span("운영 모델", className="control-label"), html.Strong("PCA (T²·SPE)")], className="toolbar-field model-field"),
            html.Div([icon("lucide:circle-check", 17), html.Span("데이터 완전")], className="data-status"),
            dmc.Button("모델 정보", id="open-model", variant="outline", size="xs", leftSection=icon("lucide:info", 16)),
            dmc.Button("검사 결과 내보내기", id="export-button", size="xs", leftSection=icon("lucide:download", 16)),
        ],
        className="filter-toolbar",
    )


def model_drawer():
    return dmc.Drawer(
        id="model-drawer",
        opened=False,
        position="right",
        size=560,
        title=html.Div([html.H2("모델 정보"), html.P("운영 판정 기준과 잠금 테스트 성능")], className="drawer-heading"),
        children=html.Div(
            [
                html.Section(
                    [
                        html.H3("운영 모델"),
                        html.Dl(
                            [
                                html.Div([html.Dt("모델"), html.Dd(["PCA (Hotelling T²·SPE) ", status_badge("운영 채택", "info")])]),
                                html.Div([html.Dt("탐지 규칙"), html.Dd("T² 또는 SPE 관리한계 초과가 10시점 연속")]),
                                html.Div([html.Dt("입력 변수"), html.Dd("셀 전압 176개 + 모듈 온도 32개")]),
                                html.Div([html.Dt("PCA 주성분"), html.Dd("6개")]),
                                html.Div([html.Dt("설명 분산"), html.Dd("97.51%")]),
                                html.Div([html.Dt("판정 임계값"), html.Dd("T² 20.427 · SPE 2.046")]),
                            ],
                            className="model-facts",
                        ),
                    ]
                ),
                html.Section(
                    [html.H3("잠금 테스트 성능"),
                     html.Div([metric("정확도", "99.15%"), metric("정밀도", "99.71%"), metric("재현율", "97.64%"), metric("F1 점수", "98.66%")], className="metric-grid"),
                     html.P("평가셋 기준 · 시험 파일 단위 분리", className="fine-print")]
                ),
                html.Section(
                    [html.H3("모델 비교"),
                     html.Table([html.Thead(html.Tr([html.Th("모델"), html.Th("F1 점수"), html.Th("비고")])),
                                 html.Tbody([html.Tr([html.Td("PCA (T²·SPE)"), html.Td("98.66%"), html.Td(status_badge("운영 채택", "info"))]),
                                             html.Tr([html.Td("MTadGAN"), html.Td("42.75%"), html.Td(status_badge("비교용", "neutral"))])])], className="model-table")]
                ),
                html.Div([icon("lucide:info", 18), html.Div([html.Strong("해석 주의"), html.P("PCA 점수는 시험 시점의 팩 단위 이상 신호입니다. 셀 Z-score와 온도 편차는 위치 추적용 보조 지표이며 원인 확정값이 아닙니다.")])], className="interpretation-note"),
                html.Div([dmc.Button("닫기", id="close-model", variant="outline"), dmc.Button("평가 결과 내보내기", id="export-model", leftSection=icon("lucide:download", 16))], className="drawer-actions"),
            ],
            className="drawer-content",
        ),
    )


def metric(label, value):
    return html.Div([html.Span(label), html.Strong(value)], className="metric-tile")


def decision_modal():
    return dmc.Modal(
        id="decision-modal",
        opened=False,
        title=html.Div([html.H3(id="decision-modal-title"), html.P("이 판정은 작업자 조치 이력에 기록됩니다.")], className="modal-title"),
        centered=True,
        children=[
            dmc.Textarea(id="decision-note", label="판정 메모", placeholder="확인 내용 또는 재시험 사유를 입력하세요.", minRows=3),
            html.Div([dmc.Button("취소", id="cancel-decision", variant="outline"), dmc.Button("판정 기록", id="confirm-decision")], className="modal-actions"),
        ],
    )
