"""BatteryFlow AI CONTROL TOWER · 예지 보전 대시보드 실행 파일.

실행:  python dashboard/app.py   (또는 dashboard 폴더에서 python app.py)
       → http://127.0.0.1:8050

구조:
    app.py            앱 생성 · 공통 화면(사이드바 + 탭바) · 탭 전환 · 실행
    data_source.py    load(table) — CSV / Firestore 읽기 창구
    common/           공통 스타일 · 공통 화면 조각
    tabs/<탭>/        탭 하나 = 폴더 하나 (layout · callbacks · figures · data)
    assets/           CSS (Dash 가 자동으로 읽음)

새 탭을 만들면 tabs/<탭>/__init__.py 에 create_layout · register_callbacks ·
TAB_VALUE 를 두고 아래 TAB_PAGES 에 추가한다.
"""

from dash import Dash, Input, Output, html

from common.layout import make_placeholder, make_sidebar, make_tabbar
from common.styles import MAIN_STYLE, PAGE_STYLE
from tabs import anomaly

# 탭바 순서 (value, 이름)
TABS = [
    ("anomaly", "이상 탐지"),
    ("forecast", "출력 예측"),
    ("explore", "데이터 탐색"),
    ("manage", "데이터 관리"),
    ("report", "보고서"),
]

# 완성된 탭만 등록한다. 나머지는 '준비 중' 화면을 보여준다.
TAB_PAGES = {
    anomaly.TAB_VALUE: anomaly,
}

app = Dash(
    __name__,
    title="BatteryFlow · 예지 보전",
    suppress_callback_exceptions=True,   # 탭을 바꿀 때 화면을 그리므로 필요
)

app.layout = html.Div(
    children=[
        make_sidebar("예지 보전"),
        html.Main(
            children=[
                make_tabbar(TABS, value="anomaly"),
                html.Div(id="tab-content"),
            ],
            style=MAIN_STYLE,
        ),
    ],
    style=PAGE_STYLE,
)


@app.callback(
    Output("tab-content", "children"),
    Input("pdm-tabs", "value"),
)
def render_tab(tab_value):
    page = TAB_PAGES.get(tab_value)
    if page is None:
        label = dict(TABS).get(tab_value, tab_value)
        return make_placeholder(label)
    return page.create_layout()


for page in TAB_PAGES.values():
    page.register_callbacks(app)


if __name__ == "__main__":
    # dev_tools_ui=False: 오른쪽 아래 디버그 도구 막대가 화면을 가리지 않게 함 (오류는 터미널에 표시)
    app.run(debug=True, dev_tools_ui=False)
