import numpy as np
import plotly.graph_objects as go
from frontend.components import BLUE, RED, CYAN, LIGHT, GRAY, TEXT


def layout(fig, height=300, ytitle=None):
    fig.update_layout(height=height, margin={"l": 46, "r": 18, "t": 30, "b": 40},
                      paper_bgcolor="white", plot_bgcolor="white", font={"family": "noto_d, sans-serif", "size": 12, "color": TEXT},
                      hovermode="x unified", legend={"orientation": "h", "y": 1.15, "x": 0, "font": {"size": 11}},
                      xaxis={"showgrid": False, "zeroline": False}, yaxis={"gridcolor": "#EDF0F5", "zeroline": False, "title": ytitle},
                      uirevision="keep")
    return fig


def demand_chart(data):
    fig = go.Figure()
    for field, label, color, dash in [("plan", "기존 D+3 계획", LIGHT, "dot"), ("actual", "실제 발주량", GRAY, "solid"), ("forecast", "선택 모델 예측", BLUE, "solid")]:
        fig.add_trace(go.Scatter(x=[r["date"] for r in data], y=[r[field] for r in data], name=label, mode="lines+markers", line={"color": color, "width": 2.5, "dash": dash}, marker={"size": 5}))
    return layout(fig, 290, "수량 (개)")


def maintenance_chart(data, score=False):
    points = data["points"]
    fig = go.Figure()
    fields = [("supRatio", data["supervised"], BLUE), ("unsupRatio", data["unsupervised"], CYAN)] if score else [("power", "실측 출력", BLUE), ("expected", "정상 기준", CYAN)]
    for key, name, color in fields:
        fig.add_trace(go.Scatter(x=[r["row"] for r in points], y=[r[key] for r in points], name=name, mode="lines", line={"color": color, "width": 1.6}))
    if score:
        fig.add_hline(y=1, line_dash="dot", line_color=RED, annotation_text="판정 기준 1.0")
    for event in data["events"]:
        fig.add_vrect(x0=event["start"]-.5, x1=event["end"]+.5, fillcolor=RED, opacity=.075, line_width=0)
    return layout(fig, 275 if not score else 220, "점수 / 임계값" if score else "RealPower")


def maintenance_heat(data):
    cycles = sorted({r["cycle"] for r in data["points"]})
    z = np.zeros((len(cycles), 39))
    lookup = {c: i for i, c in enumerate(cycles)}
    for r in data["points"]:
        z[lookup[r["cycle"]], int(r["page"])-1] = min(r["risk"], 10)
    fig = go.Figure(go.Heatmap(z=z, x=list(range(1, 40)), y=cycles, colorscale=[[0, "#EDF9FA"], [.1, CYAN], [.5, LIGHT], [1, BLUE]], colorbar={"title": "위험비"}, hovertemplate="Cycle %{y} · Page %{x}<br>위험비 %{z:.2f}<extra></extra>"))
    return layout(fig, 255)


def quality_chart(data):
    from plotly.subplots import make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    cell = data["cellSeries"]
    for key, name, color in [("cell", data["selectedCell"], BLUE), ("module", "선택 모듈 평균", CYAN)]:
        fig.add_trace(go.Scatter(x=[p["progressPct"] for p in cell], y=[p[key] for p in cell], name=name, mode="lines", line={"color": color, "width": 2}), secondary_y=False)
    for key, threshold, name, color in [("pcaQ", "spe", "SPE / 관리한계", LIGHT), ("pcaT2", "t2", "T² / 관리한계", GRAY)]:
        fig.add_trace(go.Scatter(x=[p["progressPct"] for p in data["series"]], y=[p[key]/data["pcaThresholds"][threshold] for p in data["series"]], name=name, mode="lines", line={"color": color, "width": 1.4, "dash": "dot"}), secondary_y=True)
    for seg in data["anomalySegments"]:
        a, b = seg["startRow"]/(data["rows"]-1)*100, seg["endRow"]/(data["rows"]-1)*100
        if a <= data["progress"]:
            fig.add_vrect(x0=a, x1=min(b, data["progress"]), fillcolor=RED, opacity=.07, line_width=0)
    layout(fig, 320, "셀 전압 (V)")
    fig.update_xaxes(title="시험 진행률 (%)", range=[0, data["progress"]])
    fig.update_yaxes(title="이상 점수 비율", secondary_y=True, showgrid=False)
    return fig


def quality_heat(data, mode="cell"):
    if mode == "temperature":
        z = np.array([p["temperature"] for p in data["temperatureHeatmap"]]).reshape(16, 2)
        x = ["T01", "T02"]
    else:
        z = np.array([p["zScore"] for p in data["cellHeatmap"]]).reshape(16, 11)
        x = [f"{i:02d}" for i in range(1, 12)]
    fig = go.Figure(go.Heatmap(z=z, x=x, y=[f"M{i:02d}" for i in range(1, 17)], colorscale=[[0, "#EDF9FA"], [.35, CYAN], [.65, LIGHT], [1, RED]], xgap=3, ygap=3, hovertemplate="%{y} / %{x}<br>%{z:.3f}<extra></extra>"))
    layout(fig, 320)
    fig.update_yaxes(autorange="reversed")
    return fig


def model_chart(rows, track):
    if track == "demand":
        names, values, label = [r["Model"] for r in rows], [r["MAE"] for r in rows], "MAE · 낮을수록 우수"
    elif track == "maintenance":
        rows = [r for r in rows if r["split"] == "locked_test"]
        names, values, label = [r["model"] for r in rows], [r["f1"] for r in rows], "잠금 시험 F1"
    else:
        rows = [r for r in rows if r.get("구분") == "테스트"]
        names, values, label = [r["모델"] for r in rows], [r["F1-score"] for r in rows], "잠금 시험 F1"
    fig = go.Figure(go.Bar(y=names, x=values, orientation="h", marker_color=BLUE, text=[f"{v:.3f}" for v in values], textposition="outside", cliponaxis=False))
    layout(fig, max(270, len(names)*36+45), None)
    fig.update_layout(margin={"l": 160, "r": 55, "t": 15, "b": 40}, xaxis_title=label)
    fig.update_yaxes(autorange="reversed")
    return fig
