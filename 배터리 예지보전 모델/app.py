from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dash_table, dcc, html

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from track_b_final_v2 import DATA_ROOT, Detector, add_identity, read_signal

OUTPUT_DIR = ROOT / "outputs" / "track_b_final_v2"
MODEL_DIR = OUTPUT_DIR / "models"
required = [OUTPUT_DIR / x for x in ["metrics.csv", "data_quality.csv", "run_manifest.json"]]
missing = [str(x) for x in required if not x.exists()]
if missing:
    raise FileNotFoundError("Run src/track_b_final_v2.py first. Missing: " + ", ".join(missing))

metrics = pd.read_csv(OUTPUT_DIR / "metrics.csv")
data_quality = pd.read_csv(OUTPUT_DIR / "data_quality.csv")
manifest = json.loads((OUTPUT_DIR / "run_manifest.json").read_text(encoding="utf-8"))
locked_metrics = metrics[metrics["split"].eq("locked_test")].copy()
supervised_models = locked_metrics.loc[locked_metrics["family"].eq("supervised"), "model"].tolist()
unsupervised_models = locked_metrics.loc[locked_metrics["family"].eq("unsupervised"), "model"].tolist()
DEFAULT_SUPERVISED = "LogisticCurrent"
DEFAULT_UNSUPERVISED = manifest["winner_selected_on_validation"]


def load_detector(name: str) -> Detector:
    p = joblib.load(MODEL_DIR / f"{name}.joblib")
    return Detector(
        name=p["name"], family=p["family"], model=p["model"], threshold=float(p["threshold"]),
        feature_names=list(p["feature_names"]), reference=p["reference"],
        threshold_source=p["threshold_source"], training_seconds=0.0,
    )


detectors = {name: load_detector(name) for name in supervised_models + unsupervised_models}
normal_reference = detectors[DEFAULT_UNSUPERVISED].reference


def load_replay_from_firestore_seed(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)["data"]
            common = {
                "source_file": data["sourceFile"],
                "source_row": data["sourceRow"],
                "cycle_local": data["cycle"],
                "group_id": data["groupId"],
                "PageNo": data["pageNo"],
                "WorkingTime": data["workingTime"],
                "RealPower": data["signals"]["realPower"],
                "SetPower": data["signals"]["setPower"],
                "GateOnTime": data["signals"]["gateOnTime"],
                "Speed": data["signals"]["speed"],
                "Length": data["signals"]["length"],
                "label": data["actualLabel"],
                "expected_power": data["normalReference"]["expectedPower"],
                "normal_scale": data["normalReference"]["scale"],
            }
            for model_name, model in data["models"].items():
                rows.append({
                    **common,
                    "model": model_name,
                    "family": model["family"],
                    "score": model["score"],
                    "threshold": model["threshold"],
                    "prediction": model["prediction"],
                })
    if not rows:
        raise ValueError(f"Firestore replay seed is empty: {path}")
    return pd.DataFrame(rows)


def load_replay() -> pd.DataFrame:
    blocks = []
    names = ["WeldingTest_01_OK", "WeldingTest_02_OK", "WeldingTest_03_NG", "WeldingTest_04_NG"]
    source_paths = [DATA_ROOT / "raw_data" / "test" / f"{name}.csv" for name in names]
    seed_path = ROOT / "firestore" / "seed" / "measurements.jsonl"
    if not all(path.exists() for path in source_paths):
        if seed_path.exists():
            return load_replay_from_firestore_seed(seed_path)
        missing_sources = [str(path) for path in source_paths if not path.exists()]
        raise FileNotFoundError(
            "KAMP source data and Firestore replay seed are both unavailable. Missing: "
            + ", ".join(missing_sources)
        )
    for name in names:
        frame = add_identity(read_signal(DATA_ROOT / "raw_data" / "test" / f"{name}.csv"), name)
        if name.endswith("NG"):
            frame["label"] = pd.read_csv(DATA_ROOT / "preprocessed" / "test" / f"{name}_Label.csv")["label"].to_numpy(int)
        else:
            frame["label"] = 0
        frame["expected_power"] = frame["PageNo"].map(normal_reference["median"]).astype(float)
        frame["normal_scale"] = frame["PageNo"].map(normal_reference["scale"]).astype(float)
        base_cols = ["source_file", "source_row", "cycle_local", "group_id", "PageNo", "WorkingTime",
                     "RealPower", "SetPower", "GateOnTime", "Speed", "Length", "label",
                     "expected_power", "normal_scale"]
        for model_name, detector in detectors.items():
            score = detector.score(frame)
            block = frame[base_cols].copy()
            block["model"] = model_name
            block["family"] = detector.family
            block["score"] = score
            block["threshold"] = detector.threshold
            block["prediction"] = (score >= detector.threshold).astype(int)
            blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


replay = load_replay()
source_files = replay["source_file"].drop_duplicates().tolist()


def model_rows(source: str, model: str) -> pd.DataFrame:
    return replay[replay["source_file"].eq(source) & replay["model"].eq(model)].sort_values("source_row").reset_index(drop=True)


def combined_rows(source: str, supervised: str, unsupervised: str) -> pd.DataFrame:
    sup, unsup = model_rows(source, supervised), model_rows(source, unsupervised)
    cols = ["source_file", "source_row", "cycle_local", "group_id", "PageNo", "WorkingTime",
            "RealPower", "SetPower", "GateOnTime", "Speed", "Length", "label",
            "expected_power", "normal_scale"]
    result = sup[cols].copy()
    result["supervised_score"], result["supervised_threshold"] = sup["score"], sup["threshold"]
    result["supervised_prediction"] = sup["prediction"]
    result["unsupervised_score"], result["unsupervised_threshold"] = unsup["score"], unsup["threshold"]
    result["unsupervised_prediction"] = unsup["prediction"]
    result["combined_prediction"] = (result["supervised_prediction"].astype(bool) | result["unsupervised_prediction"].astype(bool)).astype(int)
    result["risk_ratio"] = np.maximum(
        result["supervised_score"] / result["supervised_threshold"].clip(lower=1e-12),
        result["unsupervised_score"] / result["unsupervised_threshold"].clip(lower=1e-12),
    )
    return result


def regions(values: np.ndarray) -> list[tuple[int, int]]:
    changes = np.diff(np.pad(np.asarray(values, dtype=int), (1, 1)))
    return list(zip(np.flatnonzero(changes == 1).tolist(), (np.flatnonzero(changes == -1) - 1).tolist()))


def event_type(event: pd.DataFrame) -> str:
    signed = (event["RealPower"] - event["expected_power"]).median()
    direction = "저출력" if signed < 0 else "고출력"
    if len(event) == 1:
        return f"고립형 {direction} 스파이크"
    if len(event) >= 39 or event["cycle_local"].nunique() >= 2:
        return f"연속 {direction} 이상"
    if event["PageNo"].nunique() <= 2 and event["cycle_local"].nunique() >= 2:
        return f"반복 위치형 {direction} 이상"
    return f"단기 {direction} 이상"


def severity(ratio: float, length: int) -> tuple[str, str]:
    if ratio >= 3 or length >= 39:
        return "위험", "severity-danger"
    if ratio >= 1.5 or length >= 5:
        return "주의", "severity-warning"
    return "관찰", "severity-watch"


def events_for(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for event_id, (start, end) in enumerate(regions(frame["combined_prediction"].to_numpy(int)), 1):
        event = frame.iloc[start:end + 1]
        page_values = sorted(event["PageNo"].unique().tolist())
        page_text = (
            f"{page_values[0]}~{page_values[-1]} ({len(page_values)}곳)"
            if len(page_values) > 10
            else ", ".join(map(str, page_values))
        )
        sup, unsup = bool(event["supervised_prediction"].any()), bool(event["unsupervised_prediction"].any())
        agreement = "지도+비지도" if sup and unsup else "지도만" if sup else "비지도만"
        level, _ = severity(float(event["risk_ratio"].max()), len(event))
        rows.append({
            "event_id": event_id, "severity": level, "event_type": event_type(event),
            "start_row": int(event["source_row"].iloc[0]), "end_row": int(event["source_row"].iloc[-1]),
            "start_time": str(event["WorkingTime"].iloc[0]), "duration_rows": int(len(event)),
            "cycles": int(event["cycle_local"].nunique()),
            "pages": page_text,
            "max_risk_ratio": float(event["risk_ratio"].max()), "model_agreement": agreement,
            "actual_ng_overlap": "예" if event["label"].any() else "아니오",
        })
    return pd.DataFrame(rows)


def recommendations(event: pd.Series | None) -> tuple[str, list[str]]:
    if event is None:
        return "선택 구간에 탐지된 이상 이벤트가 없습니다.", ["정상 운전을 유지하고 다음 cycle을 관찰합니다.", "제품 사양 변경 시 정상 기준을 재확인합니다."]
    kind, agreement = str(event["event_type"]), str(event["model_agreement"])
    if "고립형" in kind:
        actions = ["표시된 PageNo의 치구 정렬과 순간 출력 흔들림을 확인합니다.", "같은 PageNo에서 다음 cycle에도 반복되는지 확인합니다."]
    elif "연속" in kind:
        actions = ["레이저 출력 계통과 광학 전달 경로를 우선 확인합니다.", "SetPower·GateOnTime 변경 이력과 이상 시작 시각을 대조합니다.", "후속 제품의 품질 확인 범위를 확대합니다."]
    else:
        actions = ["표시된 위치의 공정 조건과 직전 cycle을 비교합니다.", "동일 위치 반복 여부와 출력 편차 방향을 확인합니다."]
    if agreement == "비지도만":
        actions.append("미학습 신규 패턴일 수 있으므로 원본 데이터를 보존합니다.")
    elif agreement == "지도만":
        actions.append("기존 불량과 유사하지만 정상 기준 이탈이 약해 현장 판정을 병행합니다.")
    else:
        actions.append("두 계열 모델이 동시에 경보했으므로 우선 점검 대상으로 분류합니다.")
    return "모델은 원인을 확정하지 않으며 다음 항목의 우선 확인을 지원합니다.", actions


def kpi(label: str, value: str, detail: str = "", tone: str = "neutral") -> html.Div:
    return html.Div([html.Div(label, className="kpi-label"), html.Div(value, className=f"kpi-value tone-{tone}"), html.Div(detail, className="kpi-detail")], className="kpi-card")


def heading(title: str, subtitle: str) -> html.Div:
    return html.Div([html.H2(title), html.P(subtitle)], className="section-heading")


def badge(text: str, css: str) -> html.Span:
    return html.Span(text, className=f"status-badge {css}")


def power_figure(frame: pd.DataFrame, title: str) -> go.Figure:
    z = float(detectors[DEFAULT_UNSUPERVISED].threshold)
    upper, lower = frame["expected_power"] + z * frame["normal_scale"], frame["expected_power"] - z * frame["normal_scale"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["source_row"], y=upper, mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=frame["source_row"], y=lower, mode="lines", fill="tonexty", fillcolor="rgba(46,125,50,.12)", line={"width": 0}, name="정상 범위", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=frame["source_row"], y=frame["expected_power"], mode="lines", line={"color": "#2E7D32", "dash": "dot"}, name="정상 예상값"))
    fig.add_trace(go.Scatter(x=frame["source_row"], y=frame["RealPower"], mode="lines", line={"color": "#12304A", "width": 1.6}, name="실제 RealPower",
                             customdata=frame[["PageNo", "cycle_local", "combined_prediction"]], hovertemplate="행 %{x}<br>RealPower %{y:.1f}<br>PageNo %{customdata[0]}<br>Cycle %{customdata[1]}<br>경보 %{customdata[2]}<extra></extra>"))
    a = frame[frame["combined_prediction"].eq(1)]
    fig.add_trace(go.Scatter(x=a["source_row"], y=a["RealPower"], mode="markers", marker={"color": "#D64545", "size": 8, "symbol": "x"}, name="탐지 이상"))
    fig.update_layout(title=title, xaxis_title="원본 파일 행", yaxis_title="RealPower", template="plotly_white", height=430, margin={"l": 55, "r": 25, "t": 55, "b": 45}, legend={"orientation": "h", "y": 1.12})
    return fig


def heatmap(frame: pd.DataFrame) -> go.Figure:
    matrix = frame.pivot_table(index="cycle_local", columns="PageNo", values="risk_ratio", aggfunc="max").sort_index()
    power = frame.pivot_table(index="cycle_local", columns="PageNo", values="RealPower", aggfunc="first").reindex(index=matrix.index, columns=matrix.columns)
    fig = go.Figure(go.Heatmap(z=matrix.to_numpy(), x=matrix.columns, y=matrix.index, customdata=power.to_numpy(),
                               colorscale=[[0, "#E8F5E9"], [.35, "#FFF8E1"], [.6, "#FBC02D"], [1, "#C62828"]],
                               zmin=0, zmax=max(3, float(np.nanpercentile(matrix.to_numpy(), 95))), colorbar={"title": "임계값 대비"},
                               hovertemplate="Cycle %{y}<br>PageNo %{x}<br>위험비 %{z:.2f}<br>RealPower %{customdata:.1f}<extra></extra>"))
    fig.update_layout(title="용접 cycle × PageNo 이상 강도 지도", xaxis_title="용접 위치 PageNo", yaxis_title="Cycle", template="plotly_white", height=max(430, 26 * len(matrix)), margin={"l": 55, "r": 35, "t": 55, "b": 45})
    return fig


def score_figure(frame: pd.DataFrame, supervised: str, unsupervised: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["source_row"], y=frame["supervised_score"] / frame["supervised_threshold"], name=f"지도 · {supervised}", line={"color": "#1976D2"}))
    fig.add_trace(go.Scatter(x=frame["source_row"], y=frame["unsupervised_score"] / frame["unsupervised_threshold"], name=f"비지도 · {unsupervised}", line={"color": "#00897B"}))
    fig.add_hline(y=1, line_dash="dash", line_color="#D64545", annotation_text="경보 임계값")
    fig.update_layout(title="지도·비지도 이상점수 비교", xaxis_title="원본 파일 행", yaxis_title="점수 / 임계값", template="plotly_white", height=400, legend={"orientation": "h", "y": 1.15}, margin={"l": 55, "r": 25, "t": 55, "b": 45})
    return fig


def confusion(row: pd.Series) -> go.Figure:
    matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
    fig = go.Figure(go.Heatmap(z=matrix, x=["예측 정상", "예측 이상"], y=["실제 정상", "실제 이상"], colorscale="Blues", showscale=False, text=matrix, texttemplate="%{text:,}", textfont={"size": 20}))
    fig.update_layout(title=f"{row['model']} 혼동행렬", template="plotly_white", height=350, margin={"l": 70, "r": 25, "t": 55, "b": 45})
    return fig


app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "배터리팩 용접 이상감지 지원"
app.layout = html.Div([
    html.Header([
        html.Div([html.Div("WELDING AI", className="eyebrow"), html.H1("배터리팩 용접 이상감지 지원"), html.P("용접 위치별 이상 징후를 찾고 지도·비지도 모델의 근거와 점검 항목을 함께 제공합니다.")]),
        html.Div([html.Span("운영 재생 모드", className="mode-chip"), html.Div("TRACK B · v2", className="version-text")], className="header-meta"),
    ], className="app-header"),
    html.Div([
        html.Div([html.Label("시험 파일 / 작업 구간"), dcc.Dropdown(source_files, "WeldingTest_04_NG", id="source-file", clearable=False)], className="filter-item wide"),
        html.Div([html.Label([html.Span(className="family-dot supervised"), "지도학습 · 알려진 불량"]), dcc.Dropdown(supervised_models, DEFAULT_SUPERVISED, id="supervised-model", clearable=False)], className="filter-item"),
        html.Div([html.Label([html.Span(className="family-dot unsupervised"), "비지도학습 · 정상 이탈"]), dcc.Dropdown(unsupervised_models, DEFAULT_UNSUPERVISED, id="unsupervised-model", clearable=False)], className="filter-item"),
    ], className="filter-bar"),
    dcc.Tabs(id="main-tabs", value="overview", children=[
        dcc.Tab(label="설비 현황", value="overview"), dcc.Tab(label="용접 위치 지도", value="map"),
        dcc.Tab(label="이벤트·점검", value="events"), dcc.Tab(label="모델 검증실", value="models"),
        dcc.Tab(label="데이터 품질", value="quality"),
    ], className="main-tabs"),
    dcc.Loading(html.Main(id="tab-content", className="main-content"), type="circle"),
    dcc.Download(id="download"),
    html.Footer("AI 판정은 점검 우선순위를 지원하며 물리적 고장 원인을 확정하지 않습니다.", className="app-footer"),
], className="app-shell")


@callback(Output("tab-content", "children"), Input("main-tabs", "value"), Input("source-file", "value"), Input("supervised-model", "value"), Input("unsupervised-model", "value"))
def render_tab(tab: str, source: str, supervised: str, unsupervised: str):
    frame, events = combined_rows(source, supervised, unsupervised), None
    events = events_for(frame)
    anomalous = frame[frame["combined_prediction"].eq(1)]
    pages, max_ratio = sorted(anomalous["PageNo"].unique().tolist()), float(frame["risk_ratio"].max())
    if events.empty:
        status, css, tone = "정상", "severity-normal", "green"
    else:
        status, css = severity(max_ratio, int(events["duration_rows"].max()))
        tone = "red" if status == "위험" else "amber"

    if tab == "overview":
        agreement = int((frame["supervised_prediction"].astype(bool) & frame["unsupervised_prediction"].astype(bool)).sum())
        _, actions = recommendations(None if events.empty else events.iloc[-1])
        return html.Div([
            heading("설비 상태 요약", "모델 점수보다 현재 작업 구간의 이상 위치와 점검 우선순위를 먼저 보여줍니다."),
            html.Div([kpi("종합 상태", status, f"최대 위험비 {max_ratio:.2f}×", tone), kpi("탐지 이벤트", f"{len(events)}건", "연속 경보를 한 이벤트로 집계"),
                      kpi("이상 용접 위치", f"{len(pages)}곳", ", ".join(map(str, pages[:8])) or "없음"), kpi("경보 행", f"{len(anomalous):,}", f"전체 {len(frame):,}행"), kpi("두 모델 동시경보", f"{agreement:,}", "지도+비지도 합의")], className="kpi-grid"),
            html.Div([
                html.Div([html.Div([badge(status, css), html.H3("최근 이벤트 점검 안내")], className="card-title-row"), html.Ul([html.Li(x) for x in actions], className="action-list"),
                          html.Div([html.Span("지도", className="legend-pill supervised-pill"), html.Span("알려진 불량 유사성", className="legend-description"), html.Span("비지도", className="legend-pill unsupervised-pill"), html.Span("정상 공정 이탈 정도", className="legend-description")], className="model-legend")], className="panel"),
                html.Div([html.H3("판정 조합 안내"), html.Div([html.Div([html.B("지도↑ · 비지도↑"), html.Span("우선 점검")]), html.Div([html.B("지도↑ · 비지도↓"), html.Span("기존 불량 유사")]), html.Div([html.B("지도↓ · 비지도↑"), html.Span("신규 이상 가능")]), html.Div([html.B("지도↓ · 비지도↓"), html.Span("정상 운전")])], className="decision-matrix")], className="panel"),
            ], className="two-column compact"),
            html.Div(dcc.Graph(figure=power_figure(frame, f"{source} · 실제 출력과 정상 범위"), config={"displaylogo": False}), className="panel graph-panel"),
        ])

    if tab == "map":
        summary = frame.groupby("PageNo", as_index=False).agg(anomaly_rows=("combined_prediction", "sum"), max_risk=("risk_ratio", "max"), mean_power=("RealPower", "mean"), expected_power=("expected_power", "first")).sort_values(["anomaly_rows", "max_risk"], ascending=False)
        return html.Div([
            heading("용접 위치 이상 지도", "행 번호가 아니라 실제 용접 위치 PageNo 1~39와 cycle을 기준으로 이상 집중 구간을 찾습니다."),
            html.Div([html.Div(dcc.Graph(figure=heatmap(frame), config={"displaylogo": False}), className="panel graph-panel"),
                      html.Div([html.H3("이상 집중 위치"), dash_table.DataTable(data=summary.head(12).round(3).to_dict("records"), columns=[{"name": x, "id": y} for x, y in [("PageNo", "PageNo"), ("경보 행", "anomaly_rows"), ("최대 위험비", "max_risk"), ("평균 출력", "mean_power"), ("정상 예상", "expected_power")]], style_table={"overflowX": "auto"}, style_cell={"textAlign": "right", "padding": "9px"}, style_header={"fontWeight": "700"}), html.P("위험비 1.0 이상은 선택한 모델 중 하나가 임계값을 넘었다는 의미입니다.", className="note")], className="panel side-table")], className="map-layout"),
            html.Div(dcc.Graph(figure=score_figure(frame, supervised, unsupervised), config={"displaylogo": False}), className="panel graph-panel"),
        ])

    if tab == "events":
        options = [] if events.empty else [{"label": f"#{r.event_id} · {r.event_type} · 행 {r.start_row}-{r.end_row}", "value": int(r.event_id)} for r in events.itertuples()]
        value = None if events.empty else int(events.iloc[-1]["event_id"])
        return html.Div([
            heading("이상 이벤트와 점검 지원", "연속 경보를 이벤트로 묶고 이상 형태, 위치, 모델 합의와 권장 확인 항목을 제공합니다."),
            html.Div([html.Div([html.Label("상세 이벤트 선택"), dcc.Dropdown(options, value, id="event-select", clearable=False, placeholder="이상 이벤트 없음")], className="event-selector"), html.Button("이벤트 목록 다운로드", id="download-button", className="primary-button")], className="event-toolbar"),
            html.Div(id="event-detail"),
            html.Div([html.H3("전체 이벤트 관리대장"), dash_table.DataTable(data=events.round({"max_risk_ratio": 2}).to_dict("records"), columns=[{"name": a, "id": b} for a, b in [("번호", "event_id"), ("심각도", "severity"), ("이상 형태", "event_type"), ("시작 행", "start_row"), ("종료 행", "end_row"), ("영향 cycle", "cycles"), ("PageNo", "pages"), ("최대 위험비", "max_risk_ratio"), ("모델 판정", "model_agreement"), ("실제 NG 겹침", "actual_ng_overlap")]], page_size=12, style_table={"overflowX": "auto"}, style_cell={"padding": "9px", "textAlign": "left"}, style_header={"fontWeight": "700"}, style_data_conditional=[{"if": {"filter_query": "{severity} = 위험"}, "backgroundColor": "#FFF0F0", "color": "#A61B1B"}, {"if": {"filter_query": "{severity} = 주의"}, "backgroundColor": "#FFF8E7"}])], className="panel"),
        ])

    if tab == "models":
        selected = locked_metrics[locked_metrics["model"].isin([supervised, unsupervised])]
        cards = []
        for _, row in selected.iterrows():
            family_ko = "지도학습" if row["family"] == "supervised" else "비지도학습"
            cards.append(html.Div([html.Div(family_ko, className=f"model-family {row['family']}"), html.H3(row["model"]), html.Div([html.Span(f"Recall {row['recall']:.4f}"), html.Span(f"F1 {row['f1']:.4f}"), html.Span(f"FN {int(row['fn'])}"), html.Span(f"FP {int(row['fp'])}")], className="model-stat-row"), dcc.Graph(figure=confusion(row), config={"displaylogo": False})], className="panel model-card"))
        cols = [{"name": a, "id": b} for a, b in [("모델", "model"), ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"), ("FN", "fn"), ("FP", "fp"), ("이벤트 Recall", "event_recall")]]
        return html.Div([
            heading("모델 검증실", "운영 판정과 분리된 평가 증빙 화면입니다. 모든 모델은 같은 독립 파일 시험 세트로 비교했습니다."),
            html.Div(cards, className="two-column"),
            html.Div([html.Div([html.Div("지도학습", className="model-family supervised"), html.H3("알려진 불량 패턴 분류"), html.P("라벨이 있는 NG 패턴을 학습하고 불량 확률을 출력합니다."), dash_table.DataTable(data=locked_metrics[locked_metrics["family"].eq("supervised")].round(4).to_dict("records"), columns=cols, style_table={"overflowX": "auto"}, style_cell={"padding": "8px"}, style_header={"fontWeight": "700"})], className="panel"),
                      html.Div([html.Div("비지도학습", className="model-family unsupervised"), html.H3("정상 패턴 이탈 탐지"), html.P("정상 Training_Data만 학습하고 정상 보정 구간으로 임계값을 고정합니다."), dash_table.DataTable(data=locked_metrics[locked_metrics["family"].eq("unsupervised")].round(4).to_dict("records"), columns=cols, style_table={"overflowX": "auto"}, style_cell={"padding": "8px"}, style_header={"fontWeight": "700"})], className="panel")], className="two-column"),
            html.Div([html.H3("해석 주의"), html.P("Current 모델은 현재 RealPower를 사용하므로 현재 이상 감지 성능입니다. 미래 고장 시점을 예측하는 RUL 모델로 해석하지 않습니다.")], className="panel caution-panel"),
        ])

    return html.Div([
        heading("데이터 품질과 재현 정보", "파일 구조, 결측·중복, 용접 cycle, 모델 버전을 함께 확인합니다."),
        html.Div([kpi("파이프라인", manifest["pipeline_version"], "재현 버전"), kpi("정상 학습 cycle", f"{manifest['historical_split']['train_cycles']:,}", "Training_Data"), kpi("정상 보정 cycle", f"{manifest['historical_split']['calibration_cycles']:,}", "임계값 결정"), kpi("선택 모델", manifest["winner_selected_on_validation"], "validation에서 선택")], className="kpi-grid four"),
        html.Div([html.H3("파일별 품질 검사"), dash_table.DataTable(data=data_quality.to_dict("records"), columns=[{"name": c, "id": c} for c in data_quality.columns], page_size=10, style_table={"overflowX": "auto"}, style_cell={"padding": "9px", "textAlign": "right"}, style_header={"fontWeight": "700"})], className="panel"),
        html.Div([html.H3("적용 한계"), html.Ul([html.Li("현재 네 시험 파일은 운영 재생용이며 신규 설비의 완전한 외부 검증은 아닙니다."), html.Li("불량 원인은 공정 데이터만으로 확정하지 않고 제품 검사·정비 이력과 연결해야 합니다."), html.Li("제품 사양이나 recipe가 변경되면 정상 기준과 임계값을 재승인해야 합니다."), html.Li("실시간 적용에는 설비 데이터 수집 인프라와 이벤트 저장소가 필요합니다.")])], className="panel caution-panel"),
    ])


@callback(Output("event-detail", "children"), Input("event-select", "value"), State("source-file", "value"), State("supervised-model", "value"), State("unsupervised-model", "value"), prevent_initial_call=False)
def render_event_detail(event_id, source: str, supervised: str, unsupervised: str):
    frame, events = combined_rows(source, supervised, unsupervised), None
    events = events_for(frame)
    if event_id is None or events.empty:
        message, actions = recommendations(None)
        return html.Div([html.H3("탐지 이벤트 없음"), html.P(message), html.Ul([html.Li(x) for x in actions])], className="panel empty-state")
    selected = events[events["event_id"].eq(int(event_id))].iloc[0]
    event = frame[frame["source_row"].between(int(selected["start_row"]), int(selected["end_row"]))]
    context = frame[frame["source_row"].between(max(0, int(selected["start_row"]) - 20), min(int(frame["source_row"].max()), int(selected["end_row"]) + 20))]
    message, actions = recommendations(selected)
    css = {"위험": "severity-danger", "주의": "severity-warning", "관찰": "severity-watch"}[selected["severity"]]
    return html.Div([
        html.Div([
            html.Div([html.Div([badge(selected["severity"], css), html.H3(f"이벤트 #{int(selected['event_id'])} · {selected['event_type']}")], className="card-title-row"),
                      html.Div([kpi("발생 행", f"{int(selected['start_row'])}-{int(selected['end_row'])}"), kpi("영향 cycle", str(int(selected["cycles"]))), kpi("PageNo", str(selected["pages"])), kpi("최대 위험비", f"{selected['max_risk_ratio']:.2f}×"), kpi("모델 판정", str(selected["model_agreement"]))], className="event-kpis"),
                      html.P(message, className="note"), html.Ol([html.Li(x) for x in actions], className="action-list ordered")], className="panel"),
            html.Div([html.H3("이벤트 신호 요약"), html.Div([html.Div([html.Span("최저 RealPower"), html.B(f"{event['RealPower'].min():.1f}")]), html.Div([html.Span("최고 RealPower"), html.B(f"{event['RealPower'].max():.1f}")]), html.Div([html.Span("중앙 편차"), html.B(f"{(event['RealPower']-event['expected_power']).median():.1f}")]), html.Div([html.Span("실제 NG 겹침"), html.B(str(selected["actual_ng_overlap"]))])], className="signal-summary"), html.P("원인 확정에는 제품 검사 결과, 정비 이력, 비전·변위 센서 등의 추가 정보가 필요합니다.", className="note")], className="panel"),
        ], className="two-column"),
        html.Div(dcc.Graph(figure=power_figure(context, "이벤트 전후 RealPower와 정상 범위"), config={"displaylogo": False}), className="panel graph-panel"),
    ])


@callback(Output("download", "data"), Input("download-button", "n_clicks"), State("source-file", "value"), State("supervised-model", "value"), State("unsupervised-model", "value"), prevent_initial_call=True)
def download_events(_, source: str, supervised: str, unsupervised: str):
    return dcc.send_data_frame(events_for(combined_rows(source, supervised, unsupervised)).to_csv, f"{source}_anomaly_events.csv", index=False)


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=int(os.environ.get("DASH_PORT", "8050")))
