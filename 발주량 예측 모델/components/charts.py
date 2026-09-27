from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


LS_BLUE = "#0A1E5A"
LS_RED = "#FA002D"
LS_GREEN = "#009BB4"
LS_LIGHT_BLUE = "#0569A0"
TEXT = "#172033"
MUTED = "#7D8282"
GRID = "#E7EAF0"


def empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font={"color": MUTED})
    figure.update_layout(template="plotly_white", margin={"l": 20, "r": 20, "t": 20, "b": 20}, xaxis={"visible": False}, yaxis={"visible": False})
    return figure


def forecast_figure(frame: pd.DataFrame, selected_part: str, operational_mae: float) -> go.Figure:
    if frame.empty:
        return empty_figure("선택한 조건에 표시할 예측 데이터가 없습니다.")
    grouped = frame.groupby("target_date", as_index=False)[["actual", "ai_forecast", "plan"]].sum()
    figure = go.Figure()
    if selected_part != "ALL":
        lower = grouped["ai_forecast"]
        upper = grouped["ai_forecast"] + operational_mae
        figure.add_trace(go.Scatter(x=grouped["target_date"], y=upper, mode="lines", line={"width": 0}, hoverinfo="skip", showlegend=False))
        figure.add_trace(
            go.Scatter(
                x=grouped["target_date"],
                y=lower,
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(5,105,160,0.10)",
                name="CV 잔차 참고 상한 · 보장 구간 아님",
                hoverinfo="skip",
            )
        )
    figure.add_trace(go.Scatter(x=grouped["target_date"], y=grouped["actual"], mode="lines+markers", name="실제 발주", line={"color": TEXT, "width": 2.2}, marker={"size": 6}))
    figure.add_trace(go.Scatter(x=grouped["target_date"], y=grouped["ai_forecast"], mode="lines+markers", name="AI 예측", line={"color": LS_BLUE, "width": 2.5, "dash": "dot"}, marker={"size": 6}))
    figure.add_trace(go.Scatter(x=grouped["target_date"], y=grouped["plan"], mode="lines+markers", name="D+3 계획", line={"color": MUTED, "width": 1.8, "dash": "dash"}, marker={"size": 5}))
    grouped["gap_pct"] = np.where(grouped["plan"] > 0, (grouped["ai_forecast"] - grouped["plan"]) / grouped["plan"] * 100, np.nan)
    exceptions = grouped.assign(abs_gap=grouped["gap_pct"].abs()).nlargest(2, "abs_gap")
    figure.add_trace(
        go.Scatter(
            x=exceptions["target_date"],
            y=exceptions["ai_forecast"],
            mode="markers+text",
            name="주요 예외",
            text=[f"계획 대비 {value:+.0f}%" for value in exceptions["gap_pct"]],
            textposition="top center",
            textfont={"color": LS_RED, "size": 11},
            marker={"color": "white", "line": {"color": LS_RED, "width": 2}, "size": 11},
            hovertemplate="%{x|%Y-%m-%d}<br>AI 예측 %{y:,.0f}개<extra></extra>",
        )
    )
    figure.update_layout(
        template="plotly_white",
        hovermode="x unified",
        margin={"l": 54, "r": 20, "t": 20, "b": 42},
        legend={"orientation": "h", "y": 1.14, "x": 0.48, "xanchor": "center", "font": {"size": 12}},
        xaxis={"showgrid": True, "gridcolor": GRID, "tickformat": "%m.%d", "title": None, "fixedrange": True},
        yaxis={"showgrid": True, "gridcolor": GRID, "zeroline": False, "title": "수량(개)", "rangemode": "tozero", "fixedrange": True},
        font={"family": "Arial, Noto Sans KR, sans-serif", "color": TEXT, "size": 12},
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=True,
    )
    return figure


def trend_figure(history: pd.DataFrame) -> go.Figure:
    if history.empty:
        return empty_figure("최근 발주 이력이 없습니다.")
    figure = go.Figure(
        go.Scatter(
            x=history["date"],
            y=history["actual_d"],
            mode="lines+markers",
            line={"color": LS_LIGHT_BLUE, "width": 2.2},
            marker={"size": 5, "color": LS_LIGHT_BLUE},
            fill="tozeroy",
            fillcolor="rgba(5,105,160,0.05)",
            hovertemplate="%{x|%m.%d}<br>%{y:,.0f}개<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[history.iloc[-1]["date"]],
            y=[history.iloc[-1]["actual_d"]],
            mode="markers",
            marker={"size": 11, "color": "white", "line": {"color": LS_BLUE, "width": 3}},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.update_layout(
        template="plotly_white",
        margin={"l": 42, "r": 10, "t": 8, "b": 28},
        height=190,
        xaxis={"showgrid": True, "gridcolor": GRID, "tickformat": "%m.%d", "fixedrange": True},
        yaxis={"showgrid": True, "gridcolor": GRID, "rangemode": "tozero", "fixedrange": True},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Arial, Noto Sans KR, sans-serif", "color": TEXT, "size": 10},
        showlegend=False,
    )
    return figure
