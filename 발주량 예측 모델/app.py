from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from inference import predict_records  # noqa: E402


DATA_DIR = ROOT / "outputs" / "dashboard_data" / "csv"
PREDICTION_PATH = DATA_DIR / "predictions_test.csv"


def load_dashboard_data() -> dict[str, pd.DataFrame]:
    required = [
        "predictions_test.csv",
        "model_metrics.csv",
        "model_metrics_per_part.csv",
        "alerts.csv",
        "parts.csv",
    ]
    missing = [name for name in required if not (DATA_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Dashboard data is missing. Run src/build_dashboard_assets.py first: "
            + ", ".join(missing)
        )
    predictions = pd.read_csv(PREDICTION_PATH, parse_dates=["origin_date", "target_date"])
    return {
        "predictions": predictions,
        "metrics": pd.read_csv(DATA_DIR / "model_metrics.csv"),
        "part_metrics": pd.read_csv(DATA_DIR / "model_metrics_per_part.csv"),
        "alerts": pd.read_csv(DATA_DIR / "alerts.csv"),
        "parts": pd.read_csv(DATA_DIR / "parts.csv"),
    }


DATA = load_dashboard_data()
PREDICTIONS = DATA["predictions"]
METRICS = DATA["metrics"]
PART_METRICS = DATA["part_metrics"]
ALERTS = DATA["alerts"]
PARTS = sorted(PREDICTIONS["part_number"].unique())
MODEL_COLUMNS = ["XGBoost", "LightGBM", "CatBoost", "LSTM", "3-day Moving Average"]


CARD_STYLE = {
    "background": "white",
    "border": "1px solid #E2E8F0",
    "borderRadius": "10px",
    "padding": "14px 18px",
    "minWidth": "175px",
    "boxShadow": "0 1px 2px rgba(15, 23, 42, 0.05)",
}


def kpi_card(label: str, value_id: str) -> html.Div:
    return html.Div(
        [
            html.Div(label, style={"fontSize": "13px", "color": "#64748B"}),
            html.Div(id=value_id, style={"fontSize": "25px", "fontWeight": 700, "marginTop": "6px"}),
        ],
        style=CARD_STYLE,
    )


def table_from_frame(frame: pd.DataFrame, max_rows: int = 12) -> html.Table:
    shown = frame.head(max_rows)
    return html.Table(
        [html.Thead(html.Tr([html.Th(column) for column in shown.columns]))]
        + [
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(
                                f"{value:,.2f}" if isinstance(value, float) else str(value)
                            )
                            for value in row
                        ]
                    )
                    for row in shown.itertuples(index=False, name=None)
                ]
            )
        ],
        style={"width": "100%", "borderCollapse": "collapse", "fontSize": "13px"},
    )


app = Dash(__name__)
server = app.server
app.title = "D+3 발주량 예측"

app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("D+3 발주량 예측", style={"margin": 0, "fontSize": "28px"}),
                html.P(
                    "CatBoost 주 모델과 3일 이동평균 보조 모델을 함께 비교합니다. 데이터 기간이 약 50일이므로 장기 계절성은 포함하지 않습니다.",
                    style={"color": "#64748B", "marginBottom": 0},
                ),
            ],
            style={"marginBottom": "22px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("부품"),
                        dcc.Dropdown(
                            id="part-filter",
                            options=[{"label": "전체", "value": "ALL"}]
                            + [{"label": part, "value": part} for part in PARTS],
                            value="ALL",
                            clearable=False,
                        ),
                    ],
                    style={"minWidth": "260px", "flex": 1},
                ),
                html.Div(
                    [
                        html.Label("목표 날짜"),
                        dcc.DatePickerRange(
                            id="date-filter",
                            min_date_allowed=PREDICTIONS["target_date"].min().date(),
                            max_date_allowed=PREDICTIONS["target_date"].max().date(),
                            start_date=PREDICTIONS["target_date"].min().date(),
                            end_date=PREDICTIONS["target_date"].max().date(),
                            display_format="YYYY-MM-DD",
                        ),
                    ],
                    style={"minWidth": "330px"},
                ),
            ],
            style={"display": "flex", "gap": "20px", "alignItems": "end", "marginBottom": "18px"},
        ),
        html.Div(
            [
                kpi_card("실제 수량 합계", "kpi-actual"),
                kpi_card("CatBoost 예측 합계", "kpi-catboost"),
                kpi_card("3일 이동평균 합계", "kpi-moving"),
                kpi_card("CatBoost MAE", "kpi-mae"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "18px"},
        ),
        html.Div(
            [
                dcc.Graph(id="forecast-chart", style={"flex": 2}),
                dcc.Graph(id="metrics-chart", style={"flex": 1}),
            ],
            style={"display": "flex", "gap": "16px", "background": "white", "borderRadius": "10px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H3("선택 부품의 모델 성능"),
                        html.Div(id="part-metrics-table"),
                    ],
                    style={"flex": 1, "background": "white", "padding": "16px", "borderRadius": "10px"},
                ),
                html.Div(
                    [
                        html.H3("계획 대비 확인 대상"),
                        html.Div(id="alerts-table"),
                    ],
                    style={"flex": 1, "background": "white", "padding": "16px", "borderRadius": "10px"},
                ),
            ],
            style={"display": "flex", "gap": "16px", "marginTop": "16px"},
        ),
        html.Div(
            [
                html.H2("새 3일 데이터로 D+3 추론", style={"fontSize": "21px"}),
                html.P(
                    "열 이름이 part_number, date, actual_d, plan_d3, plan_d4, plan_d5인 연속 3일 CSV를 업로드하세요.",
                    style={"color": "#64748B"},
                ),
                dcc.Upload(
                    id="inference-upload",
                    children=html.Div(["CSV 파일을 끌어 놓거나 선택"]),
                    style={
                        "width": "100%",
                        "height": "70px",
                        "lineHeight": "70px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "8px",
                        "textAlign": "center",
                        "background": "#F8FAFC",
                    },
                ),
                html.Button(
                    "예측 실행",
                    id="run-inference",
                    n_clicks=0,
                    style={
                        "marginTop": "12px",
                        "background": "#2563EB",
                        "color": "white",
                        "border": 0,
                        "borderRadius": "7px",
                        "padding": "10px 18px",
                        "cursor": "pointer",
                    },
                ),
                html.Pre(
                    id="inference-result",
                    style={"background": "#0F172A", "color": "#E2E8F0", "padding": "14px", "borderRadius": "8px"},
                ),
            ],
            style={"background": "white", "padding": "20px", "borderRadius": "10px", "marginTop": "16px"},
        ),
    ],
    style={
        "maxWidth": "1450px",
        "margin": "0 auto",
        "padding": "28px",
        "background": "#F1F5F9",
        "minHeight": "100vh",
        "fontFamily": "Arial, sans-serif",
        "color": "#0F172A",
    },
)


@app.callback(
    Output("kpi-actual", "children"),
    Output("kpi-catboost", "children"),
    Output("kpi-moving", "children"),
    Output("kpi-mae", "children"),
    Output("forecast-chart", "figure"),
    Output("metrics-chart", "figure"),
    Output("part-metrics-table", "children"),
    Output("alerts-table", "children"),
    Input("part-filter", "value"),
    Input("date-filter", "start_date"),
    Input("date-filter", "end_date"),
)
def update_dashboard(part_number: str, start_date: str, end_date: str):
    filtered = PREDICTIONS[
        (PREDICTIONS["target_date"] >= pd.Timestamp(start_date))
        & (PREDICTIONS["target_date"] <= pd.Timestamp(end_date))
    ].copy()
    if part_number != "ALL":
        filtered = filtered[filtered["part_number"] == part_number]

    grouped = filtered.groupby("target_date", as_index=False)[
        ["actual", "CatBoost", "3-day Moving Average", "D+3 Plan Reference"]
    ].sum()
    forecast_chart = go.Figure()
    for column, color in [
        ("actual", "#0F172A"),
        ("CatBoost", "#2563EB"),
        ("3-day Moving Average", "#F59E0B"),
        ("D+3 Plan Reference", "#94A3B8"),
    ]:
        forecast_chart.add_trace(
            go.Scatter(x=grouped["target_date"], y=grouped[column], mode="lines+markers", name=column, line={"color": color})
        )
    forecast_chart.update_layout(
        title="실제 수량과 예측 수량",
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 55, "b": 35},
        legend={"orientation": "h", "y": 1.12},
    )

    metric_source = METRICS[METRICS["Eligible_For_Ranking"]].sort_values("MAE")
    metrics_chart = px.bar(
        metric_source,
        x="MAE",
        y="Model",
        orientation="h",
        title="최종 holdout MAE",
        color="MAE",
        color_continuous_scale="Blues_r",
    )
    metrics_chart.update_layout(template="plotly_white", coloraxis_showscale=False, yaxis={"categoryorder": "total descending"})

    if part_number == "ALL":
        selected_metrics = (
            PART_METRICS.groupby("Model", as_index=False)[["MAE", "RMSE", "WAPE_pct", "Forecast_Bias"]]
            .mean()
            .sort_values("MAE")
        )
        selected_alerts = ALERTS[ALERTS["alert_type"] != "within_plan_band"].copy()
    else:
        selected_metrics = PART_METRICS[PART_METRICS["part_number"] == part_number][
            ["Model", "N", "MAE", "RMSE", "WAPE_pct", "Forecast_Bias", "R2"]
        ].sort_values("MAE")
        selected_alerts = ALERTS[ALERTS["part_number"] == part_number].copy()
    selected_alerts = selected_alerts[
        ["part_number", "target_date", "recommended_model", "recommended_forecast", "D+3 Plan Reference", "alert_type"]
    ].sort_values("recommended_forecast", ascending=False)

    cat_mae = (
        (filtered["CatBoost"] - filtered["actual"]).abs().mean() if len(filtered) else float("nan")
    )
    return (
        f"{filtered['actual'].sum():,.0f}",
        f"{filtered['CatBoost'].sum():,.0f}",
        f"{filtered['3-day Moving Average'].sum():,.0f}",
        f"{cat_mae:,.2f}",
        forecast_chart,
        metrics_chart,
        table_from_frame(selected_metrics.round(3)),
        table_from_frame(selected_alerts.round(2)),
    )


@app.callback(
    Output("inference-result", "children"),
    Input("run-inference", "n_clicks"),
    State("inference-upload", "contents"),
    State("inference-upload", "filename"),
    prevent_initial_call=True,
)
def run_inference(_: int, contents: str | None, filename: str | None) -> str:
    if not contents:
        return "CSV 파일을 먼저 선택하세요."
    try:
        _, encoded = contents.split(",", 1)
        decoded = base64.b64decode(encoded)
        records = pd.read_csv(io.BytesIO(decoded))
        result = predict_records(records)
        return (
            f"파일: {filename}\n"
            f"부품: {result['part_number']}\n"
            f"목표일: {result['target_date']}\n"
            f"CatBoost: {result['catboost_prediction']}\n"
            f"3일 이동평균: {result['moving_average_3d']:.2f}\n"
            f"권장 모델: {result['recommended_model']}\n"
            f"권장 예측량: {result['recommended_forecast']:.2f}\n"
            f"Fallback: {result['fallback_reason']}"
        )
    except Exception as error:
        return f"예측 실패: {error}"


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
