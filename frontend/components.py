from dash import dcc, html
import dash_ag_grid as dag
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import json
from pathlib import Path

_ICON_SET = json.loads((Path(__file__).parent / 'assets' / 'icons.json').read_text(encoding='utf-8'))

BLUE = "#0A1E5A"
RED = "#FA002D"
CYAN = "#009BB4"
LIGHT = "#0569A0"
GRAY = "#7D8282"
TEXT = "#172033"


def icon(name, size=18):
    data = _ICON_SET.get('icons', {}).get(name)
    return DashIconify(icon={**data, 'width': 24, 'height': 24} if data else f"lucide:{name}", width=size, height=size)


def badge(text, tone="neutral"):
    return html.Span(text, className=f"status-badge {tone}")


def settings_disclosure(label, controls):
    return html.Details([html.Summary(label), html.Div(controls, className='settings-panel',
                         role='group', **{'aria-label': label})], className='advanced-controls')


def panel(title, children, subtitle=None, action=None, cls=""):
    return html.Section([html.Div([html.Div([html.H2(title), html.P(subtitle) if subtitle else None]), action], className="panel-head"),
                         html.Div(children, className="panel-body")], className=f"panel {cls}")


def kpi(label, value, detail, tone="", unit=""):
    return html.Div([html.Div(label, className="kpi-label"), html.Div([str(value), html.Small(unit)], className=f"kpi-number {tone}"),
                     html.Div(detail, className="kpi-detail")], className="kpi")


def callout(title, text, tone="info"):
    return html.Div([icon("info" if tone == "info" else "triangle-alert"), html.Div([html.Strong(title), html.P(text)])], className=f"callout {tone}")


def grid(rows, id, columns=None, height=330):
    if columns is None:
        keys = list(dict.fromkeys(k for row in rows for k, v in row.items() if not isinstance(v, (dict, list))))
        columns = [{"field": k, "headerName": k} for k in keys if k not in ["schemaVersion", "dataVersion"]]
    return dag.AgGrid(id=id, rowData=rows, columnDefs=columns,
        defaultColDef={"sortable": True, "filter": True, "resizable": True, "minWidth": 115, "flex": 1},
        dashGridOptions={"rowHeight": 40, "headerHeight": 42, "pagination": True, "paginationPageSize": 10,
                         "localeText": {"page": "페이지", "to": "–", "of": "/", "more": "더 보기", "noRowsToShow": "표시할 항목이 없습니다", "loadingOoo": "불러오는 중…", "filterOoo": "검색…", "equals": "같음", "notEqual": "같지 않음", "contains": "포함", "notContains": "포함하지 않음", "startsWith": "시작", "endsWith": "끝", "blank": "빈 값", "notBlank": "값 있음", "applyFilter": "적용", "resetFilter": "초기화", "clearFilter": "해제", "lessThan": "미만", "greaterThan": "초과", "lessThanOrEqual": "이하", "greaterThanOrEqual": "이상", "inRange": "범위", "andCondition": "그리고", "orCondition": "또는", "firstPage": "첫 페이지", "previousPage": "이전 페이지", "nextPage": "다음 페이지", "lastPage": "마지막 페이지"},
                         "paginationPageSizeSelector": False, "animateRows": False, "tooltipShowDelay": 150,
                         "rowSelection": {"mode": "singleRow", "checkboxes": False, "enableClickSelection": True}},
        style={"height": f"{height}px"}, className="ag-theme-quartz data-grid")


def graph(figure, id=None):
    props = {"figure": figure, "config": {"displayModeBar": False, "responsive": True}}
    if id:
        props["id"] = id
    return dcc.Graph(**props)


def metric_rows(rows):
    return html.Div([html.Div([html.Div([html.Strong(label), html.Small(note)]), html.B(value)], className="metric-row") for label, value, note in rows], className="metric-list")


def select(id, label, data, value, width=None):
    return dmc.Select(id=id, label=label, data=data, value=value, allowDeselect=False, searchable=len(data)>8,
                      persistence=True, persistence_type="session", size="sm", className="context-select")


def num(value, digits=0):
    return "—" if value is None else f"{value:,.{digits}f}"
