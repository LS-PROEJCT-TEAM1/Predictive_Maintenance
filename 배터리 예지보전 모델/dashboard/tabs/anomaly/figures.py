"""이상 탐지 탭의 Plotly 그래프 생성 함수."""

import plotly.express as px
import plotly.graph_objects as go

from common.styles import (
    FONT_NUM, GRAY, INK, TEAL_200, TEAL_500, TEAL_600, apply_figure_style,
)

FILE_COLORS = {"02_OK": TEAL_200, "03_NG": TEAL_500, "04_NG": INK, "01_OK": GRAY}


def make_empty_figure(message="선택 조건에 해당하는 데이터가 없습니다.", height=None):
    fig = go.Figure()
    fig.add_annotation(
        text=message, showarrow=False, x=0.5, y=0.5,
        xref="paper", yref="paper", font={"color": GRAY, "size": 13},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return apply_figure_style(fig, height)


def make_condition_bar(ratio_df, unseen_setpowers):
    """공정 조건(SetPower)별 이상 판정 비율 — 파일별 묶음 막대."""
    if ratio_df.empty:
        return make_empty_figure("선택 조건에서 이상 판정이 있는 파일이 없습니다.")

    plot_df = ratio_df.copy()
    plot_df["setpower_label"] = plot_df["SetPower"].map(lambda v: f"{v}%")
    plot_df["ratio_pct"] = (plot_df["ratio"] * 100).fillna(0)
    plot_df["bar_text"] = plot_df.apply(
        lambda r: "–" if r["total"] != r["total"] or r["total"] == 0
        else f"{r['ratio_pct']:.1f}%<br>{int(r['anomaly'])}/{int(r['total'])}",
        axis=1,
    )

    fig = px.bar(
        plot_df,
        x="setpower_label",
        y="ratio_pct",
        color="file_short",
        barmode="group",
        text="bar_text",
        color_discrete_map=FILE_COLORS,
        custom_data=["anomaly", "total"],
        labels={"setpower_label": "SetPower (%)", "ratio_pct": "이상 판정 비율 (%)", "file_short": "파일"},
    )
    fig.update_traces(
        textposition="outside",
        textfont={"family": FONT_NUM, "size": 10, "color": INK},
        cliponaxis=False,
        hovertemplate="%{x} · %{y:.1f}%<br>이상 %{customdata[0]} / 전체 %{customdata[1]}<extra>%{fullData.name}</extra>",
    )

    categories = [f"{v}%" for v in sorted(plot_df["SetPower"].unique())]
    tick_text = [
        f"{c}<br><span style='font-size:10px;color:{GRAY}'>학습외</span>"
        if int(c[:-1]) in unseen_setpowers else c
        for c in categories
    ]
    fig.update_xaxes(categoryorder="array", categoryarray=categories,
                     tickvals=categories, ticktext=tick_text, title_text="SetPower (%)")
    fig.update_yaxes(range=[0, 118], ticksuffix="%", title_text="이상 판정 비율 (%)")
    fig.update_layout(
        legend={"title": None, "orientation": "v", "x": 1.01, "y": 1, "font": {"size": 11}},
        bargap=0.25 if plot_df["SetPower"].nunique() > 1 else 0.7,
        uniformtext_minsize=8,
    )
    fig = apply_figure_style(fig)
    fig.update_layout(margin={"l": 50, "r": 70, "t": 16, "b": 50})
    return fig


def make_threshold_curve(sweep_df, applied_quantile):
    """임계값(보정 분위수)별 Precision · Recall · F1."""
    x_index = list(range(len(sweep_df)))
    tick_text = [f"{q * 100:g}%" for q in sweep_df["calibration_quantile"]]

    fig = go.Figure()
    lines = [
        ("precision", "Precision", INK, "solid"),
        ("recall", "Recall", TEAL_600, "solid"),
        ("f1", "F1", GRAY, "dot"),
    ]
    for column, name, color, dash in lines:
        fig.add_trace(
            go.Scatter(
                x=x_index,
                y=sweep_df[column],
                name=name,
                mode="lines+markers",
                line={"color": color, "width": 2, "dash": dash},
                marker={"size": 5},
                customdata=sweep_df["threshold"],
                hovertemplate=f"{name} %{{y:.4f}}<br>임계값 %{{customdata:.2f}}<extra></extra>",
            )
        )

    applied_index = sweep_df.index[sweep_df["calibration_quantile"] == applied_quantile]
    if len(applied_index):
        fig.add_vline(x=int(applied_index[0]), line={"color": TEAL_500, "width": 2, "dash": "dash"})

    fig.update_xaxes(tickvals=x_index, ticktext=tick_text, title_text="정상 보정 분위수",
                     tickfont={"family": FONT_NUM, "size": 10})
    fig.update_yaxes(range=[0.7, 1.02], title_text="값", tickformat=".2f",
                     tickfont={"family": FONT_NUM})
    fig.update_layout(
        legend={"orientation": "h", "x": 0, "y": 1.1, "font": {"size": 11}},
        hovermode="x unified",
    )
    fig = apply_figure_style(fig)
    fig.update_layout(margin={"l": 50, "r": 16, "t": 30, "b": 44})
    return fig
