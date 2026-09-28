from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LS_BLUE = "#0A1E5A"
LS_RED = "#FA002D"
LS_GREEN = "#009BB4"
LS_LIGHT_BLUE = "#0569A0"
GRID = "#E7EAF0"
TEXT = "#172033"
MUTED = "#667085"


def _regions(mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    starts = np.where(mask & ~np.r_[False, mask[:-1]])[0]
    ends = np.where(mask & ~np.r_[mask[1:], False])[0]
    return list(zip(starts, ends))


def evidence_figure(raw: pd.DataFrame, scores: pd.DataFrame, selected_cell: str):
    cv_cols = [c for c in raw.columns if "CV" in c]
    module = selected_cell[:3]
    module_cols = [c for c in cv_cols if c.startswith(module)]
    pack_median = raw[cv_cols].median(axis=1)
    selected_delta = (raw[selected_cell] - pack_median) * 1000
    module_delta = (raw[module_cols].median(axis=1) - pack_median) * 1000
    x = np.linspace(0, 100, len(raw))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.13,
                        row_heights=[0.58, 0.42], subplot_titles=("선택 셀 전압 편차", "PCA 이상점수 / 관리한계 · 로그 축"))
    sample = max(1, len(raw) // 900)
    for m in range(1, 17):
        cols = [c for c in cv_cols if c.startswith(f"M{m:02d}")]
        delta = (raw[cols].median(axis=1) - pack_median) * 1000
        fig.add_trace(go.Scatter(x=x[::sample], y=delta.iloc[::sample], mode="lines", line=dict(color="#CCD5E3", width=0.6),
                                 hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=x[::sample], y=module_delta.iloc[::sample], mode="lines", name=f"선택 모듈 ({module})",
                             line=dict(color=LS_BLUE, width=1.7), showlegend=False,
                             hovertemplate="%{x:.1f}%<br>%{y:.2f} mV<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x[::sample], y=selected_delta.iloc[::sample], mode="lines", name=f"선택 셀 ({selected_cell})",
                             line=dict(color=LS_RED, width=2.0), showlegend=False,
                             hovertemplate="%{x:.1f}%<br>%{y:.2f} mV<extra></extra>"), row=1, col=1)

    spe_ratio = scores["SPE"].to_numpy() / 2.046
    t2_ratio = scores["Hotelling_T2"].to_numpy() / 20.427
    safe_spe = np.maximum(spe_ratio, 0.01)
    safe_t2 = np.maximum(t2_ratio, 0.01)
    fig.add_trace(go.Scatter(x=x[::sample], y=safe_spe[::sample], mode="lines", name="SPE 비율", line=dict(color=LS_LIGHT_BLUE, width=2),
                             showlegend=False, customdata=spe_ratio[::sample], hovertemplate="%{x:.1f}%<br>SPE %{customdata:.2f}×<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x[::sample], y=safe_t2[::sample], mode="lines", name="T² 비율", line=dict(color=LS_GREEN, width=1.6, dash="dash"),
                             showlegend=False, customdata=t2_ratio[::sample], hovertemplate="%{x:.1f}%<br>T² %{customdata:.2f}×<extra></extra>"), row=2, col=1)
    alert = scores["예측"].to_numpy().astype(bool)
    alert_idx = np.where(alert)[0]
    point_idx = alert_idx[::max(1, len(alert_idx) // 24)] if len(alert_idx) else []
    fig.add_trace(go.Scatter(x=x[point_idx], y=safe_spe[point_idx], mode="markers", name="관리한계 초과", marker=dict(color=LS_RED, size=5),
                             showlegend=False, customdata=spe_ratio[point_idx], hovertemplate="%{x:.1f}%<br>SPE %{customdata:.2f}×<extra></extra>"), row=2, col=1)

    for start, end in _regions(alert):
        fig.add_vrect(x0=x[start], x1=x[end], fillcolor="rgba(250,0,45,0.08)", line_width=0, row="all", col=1)
    fig.add_hline(y=1, line_color="#7D8282", line_dash="dot", line_width=1.2, row=2, col=1,
                  annotation_text="관리한계 1.00×", annotation_position="top left")
    fig.update_yaxes(title_text="팩 중앙값 대비 (mV)", row=1, col=1, zeroline=True, zerolinecolor="#AAB3C2")
    fig.update_yaxes(title_text="관리한계 비율 (×)", type="log", row=2, col=1, range=[-2, 5])
    fig.update_xaxes(title_text="시험 진행률 (%)", row=2, col=1)
    fig.update_layout(template="plotly_white", height=405, margin=dict(l=55, r=18, t=42, b=36), hovermode="x unified",
                      font=dict(family="Arial, Malgun Gothic, sans-serif", size=12, color=TEXT),
                      showlegend=False,
                      paper_bgcolor="white", plot_bgcolor="white")
    fig.update_xaxes(showgrid=True, gridcolor=GRID, fixedrange=True)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, fixedrange=True)
    return fig


def location_figure(ranks: pd.DataFrame, raw: pd.DataFrame, mode: str, selected_cell: str):
    if mode == "temperature":
        cols = [c for c in raw.columns if "T" in c and len(c) == 6 and c.startswith("M")]
        vals = raw[cols]
        center = vals.median(axis=1)
        spread = vals.sub(center, axis=0).abs().mean(axis=0)
        z = np.zeros((16, 2))
        for col, value in spread.items():
            mi, ti = int(col[1:3]) - 1, int(col[-2:]) - 1
            z[mi, ti] = float(value)
        xlabels = ["T01", "T02"]
        title = "모듈 온도 평균 절대편차 (℃) · 위치 추적용"
        colorscale = [[0, "#EAF2FB"], [0.55, "#74A9D8"], [1, LS_RED]]
        zmax = max(0.4, float(z.max()))
        selected = None
    else:
        z = np.zeros((16, 11))
        for row in ranks.to_dict("records"):
            cell = row["셀"]
            z[int(cell[1:3]) - 1, int(cell[-2:]) - 1] = float(row["이상점수"])
        xlabels = [f"{i:02d}" for i in range(1, 12)]
        title = "셀별 Z-score · 위치 추적용"
        colorscale = [[0, "#EAF2FB"], [0.4, "#B7D6F2"], [0.65, "#F7C968"], [1, LS_RED]]
        zmax = max(3.33, float(z.max()))
        selected = selected_cell
    ylabels = [f"M{i:02d}" for i in range(1, 17)]
    fig = go.Figure(go.Heatmap(z=z, x=xlabels, y=ylabels, colorscale=colorscale, zmin=0, zmax=zmax,
                               colorbar=dict(title="점수", thickness=10, len=0.75),
                               xgap=2, ygap=2, hovertemplate="%{y} / %{x}<br>%{z:.2f}<extra></extra>"))
    if selected:
        mi, ci = int(selected[1:3]) - 1, int(selected[-2:]) - 1
        fig.add_shape(type="rect", x0=ci - 0.48, x1=ci + 0.48, y0=mi - 0.48, y1=mi + 0.48,
                      line=dict(color=LS_BLUE, width=2.5))
    fig.update_layout(height=226, margin=dict(l=46, r=66, t=28, b=28), paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(family="Arial, Malgun Gothic, sans-serif", size=11, color=TEXT),
                      annotations=[dict(text=title, x=1, y=1.12, xref="paper", yref="paper", showarrow=False, xanchor="right", font=dict(size=11, color=MUTED))])
    fig.update_yaxes(autorange="reversed", fixedrange=True, tickfont=dict(size=10))
    fig.update_xaxes(side="top", fixedrange=True, tickfont=dict(size=10))
    return fig
