from dash import dcc, html
import dash_ag_grid as dag
import dash_mantine_components as dmc
from dash_iconify import DashIconify

BLUE = "#0A1E5A"
RED = "#FA002D"
CYAN = "#009BB4"
LIGHT = "#0569A0"
GRAY = "#7D8282"
TEXT = "#172033"


def icon(name, size=18):
    return DashIconify(icon=f"lucide:{name}", width=size, height=size)


def badge(text, tone="neutral"):
    return html.Span(text, className=f"status-badge {tone}")


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
        dashGridOptions={"rowHeight": 39, "headerHeight": 40, "pagination": True, "paginationPageSize": 10,
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
