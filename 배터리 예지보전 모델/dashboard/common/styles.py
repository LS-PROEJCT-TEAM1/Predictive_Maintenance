"""공통 스타일 — 색 8개, 글꼴, 카드 스타일, Plotly 기본 레이아웃."""

# 중립색
LIGHT = "#F5F5F5"
GRAY = "#76767E"
INK = "#0A1119"

# 포인트색 (이상·불량 의심·활성 상태에만 사용)
WHITE = "#FFFFFF"
TEAL_050 = "#DDF8F5"   # 연한 강조
TEAL_200 = "#7DE3D9"   # 재검사 권고
TEAL_500 = "#16C7B7"   # 불량 의심 · 활성
TEAL_600 = "#13A89B"   # 강조 숫자

FONT_BODY = "Pretendard, 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif"
FONT_NUM = "'IBM Plex Mono', Consolas, monospace"

GAP = "14px"

PAGE_STYLE = {
    "display": "flex",
    "minHeight": "100vh",
    "backgroundColor": LIGHT,
    "fontFamily": FONT_BODY,
    "color": INK,
}

MAIN_STYLE = {
    "flex": "1",
    "minWidth": "0",
    "padding": "22px 28px",
}

CARD_STYLE = {
    "backgroundColor": WHITE,
    "borderRadius": "10px",
    "padding": "18px 20px",
}

CARD_TITLE_STYLE = {
    "fontSize": "15px",
    "fontWeight": "700",
    "margin": "0",
    "color": INK,
}

TAG_STYLE = {
    "fontFamily": FONT_NUM,
    "fontSize": "11px",
    "color": GRAY,
    "backgroundColor": LIGHT,
    "borderRadius": "4px",
    "padding": "2px 6px",
}

LABEL_STYLE = {
    "fontSize": "12px",
    "color": GRAY,
    "marginBottom": "6px",
    "display": "block",
}

NUM_STYLE = {"fontFamily": FONT_NUM}


def grid_style(columns, margin_bottom=GAP):
    """카드를 가로로 나란히 놓는 grid 스타일."""
    return {
        "display": "grid",
        "gridTemplateColumns": columns,
        "gap": GAP,
        "marginBottom": margin_bottom,
    }


def apply_figure_style(fig, height=None):
    """Plotly 그래프 공통 모양: 흰 배경, 옅은 격자, 본문 글꼴.

    height=None 이면 그래프를 담은 영역 크기에 맞춰 자동으로 늘어난다(autosize).
    """
    fig.update_layout(
        height=height,
        autosize=True,
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font={"family": FONT_BODY, "size": 12, "color": INK},
        margin={"l": 50, "r": 20, "t": 10, "b": 40},
        hoverlabel={"font": {"family": FONT_BODY}},
    )
    fig.update_xaxes(gridcolor=LIGHT, linecolor=LIGHT, zeroline=False)
    fig.update_yaxes(gridcolor=LIGHT, linecolor=LIGHT, zeroline=False)
    return fig
