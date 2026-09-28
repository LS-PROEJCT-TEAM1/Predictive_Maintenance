from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


LS_BLUE = "#0A1E5A"
LS_RED = "#FA002D"
LS_GREEN = "#009BB4"
LS_LIGHT_BLUE = "#0569A0"
INK = "#172033"
GRID = "#E7EAF0"


def _base_layout(height: int, *, bottom: int = 36) -> dict:
    return {
        "template": "plotly_white",
        "height": height,
        "margin": {"l": 52, "r": 18, "t": 14, "b": bottom},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "#FFFFFF",
        "font": {"family": "Segoe UI, Malgun Gothic, sans-serif", "size": 12, "color": INK},
        "hovermode": "x unified",
        "legend": {"orientation": "h", "y": 1.12, "x": 0, "font": {"size": 11}},
        "xaxis": {"gridcolor": GRID, "zeroline": False, "showline": True, "linecolor": "#D8DDE6"},
        "yaxis": {"gridcolor": GRID, "zeroline": False, "showline": False},
    }


def _event_regions(frame: pd.DataFrame) -> list[tuple[float, float]]:
    values = frame["combined_prediction"].to_numpy(dtype=int)
    changes = np.diff(np.pad(values, (1, 1)))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    x = frame["source_row"].to_numpy()
    return [(float(x[start]), float(x[end])) for start, end in zip(starts, ends)]


def power_figure(frame: pd.DataFrame) -> go.Figure:
    threshold = frame["unsupervised_threshold"].astype(float)
    upper = frame["expected_power"] + threshold * frame["normal_scale"]
    lower = frame["expected_power"] - threshold * frame["normal_scale"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["source_row"], y=upper, mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=frame["source_row"], y=lower, mode="lines", fill="tonexty",
            fillcolor="rgba(0,155,180,.13)", line={"width": 0}, name="정상 허용 범위", hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["source_row"], y=frame["expected_power"], mode="lines",
            line={"color": LS_LIGHT_BLUE, "width": 1.8, "dash": "dash"}, name="PageNo별 정상 기준값",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["source_row"], y=frame["RealPower"], mode="lines",
            line={"color": LS_BLUE, "width": 2}, name="실측 RealPower",
            customdata=frame[["PageNo", "cycle_local", "expected_power"]],
            hovertemplate="행 %{x}<br>실측 %{y:.1f} W<br>정상 기준 %{customdata[2]:.1f} W<br>PageNo %{customdata[0]}<br>반복 Cycle %{customdata[1]}<extra></extra>",
        )
    )
    anomaly = frame[frame["combined_prediction"].eq(1)]
    fig.add_trace(
        go.Scatter(
            x=anomaly["source_row"], y=anomaly["RealPower"], mode="lines",
            line={"color": LS_RED, "width": 2.5}, name="이상 판정 구간",
            hovertemplate="행 %{x}<br>이상 판정 RealPower %{y:.1f} W<extra></extra>",
        )
    )
    for start, end in _event_regions(frame):
        fig.add_vrect(x0=start, x1=end, fillcolor="rgba(250,0,45,.06)", line_width=0, layer="below")
    fig.update_layout(**_base_layout(248, bottom=24))
    fig.update_xaxes(title=None, showticklabels=False)
    fig.update_yaxes(title="RealPower (W)")
    return fig


def score_figure(frame: pd.DataFrame, supervised: str, unsupervised: str) -> go.Figure:
    supervised_ratio = frame["supervised_score"] / frame["supervised_threshold"].clip(lower=1e-12)
    unsupervised_ratio = frame["unsupervised_score"] / frame["unsupervised_threshold"].clip(lower=1e-12)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["source_row"], y=supervised_ratio, mode="lines",
            line={"color": LS_BLUE, "width": 1.8}, name=supervised,
            hovertemplate="행 %{x}<br>임계값 대비 %{y:.2f}×<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["source_row"], y=unsupervised_ratio, mode="lines",
            line={"color": LS_GREEN, "width": 1.8}, name=unsupervised,
            hovertemplate="행 %{x}<br>임계값 대비 %{y:.2f}×<extra></extra>",
        )
    )
    fig.add_hline(y=1, line_dash="dash", line_color=LS_RED, annotation_text="판정 기준 1.0", annotation_position="top right")
    for start, end in _event_regions(frame):
        fig.add_vrect(x0=start, x1=end, fillcolor="rgba(250,0,45,.07)", line_width=0, layer="below")
    fig.update_layout(**_base_layout(132, bottom=30))
    fig.update_xaxes(title="원본 파일 행")
    fig.update_yaxes(title="점수 ÷ 임계값", rangemode="tozero")
    return fig


def heatmap_figure(frame: pd.DataFrame) -> go.Figure:
    matrix = frame.pivot_table(index="cycle_local", columns="PageNo", values="risk_ratio", aggfunc="max").sort_index()
    power = frame.pivot_table(index="cycle_local", columns="PageNo", values="RealPower", aggfunc="first").reindex(index=matrix.index, columns=matrix.columns)
    maximum = max(3.0, float(np.nanpercentile(matrix.to_numpy(), 98)))
    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(), x=matrix.columns, y=matrix.index, customdata=power.to_numpy(),
            colorscale=[[0, "#DDF5EF"], [.32, "#BFECE4"], [.34, "#FFF2CF"], [.62, "#F4B44E"], [1, LS_RED]],
            zmin=0, zmax=maximum, showscale=False,
            hovertemplate="반복 Cycle %{y}<br>PageNo %{x}<br>임계값 대비 %{z:.2f}×<br>RealPower %{customdata:.1f} W<extra></extra>",
            xgap=2, ygap=2,
        )
    )
    fig.update_layout(**_base_layout(158, bottom=32))
    fig.update_xaxes(title="PageNo", dtick=2)
    fig.update_yaxes(title="반복 Cycle", dtick=1)
    return fig


def event_context_figure(context: pd.DataFrame, start_row: int, end_row: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=context["source_row"], y=context["RealPower"], mode="lines", line={"color": LS_BLUE, "width": 1.8}, name="RealPower"))
    fig.add_vrect(x0=start_row, x1=end_row, fillcolor="rgba(250,0,45,.10)", line_color=LS_RED, line_dash="dash")
    fig.update_layout(**_base_layout(170, bottom=28), showlegend=False)
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="W")
    return fig
