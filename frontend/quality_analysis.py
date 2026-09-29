"""Approved quality inspection workspace; source labels stay separate from AI output."""
from dash import html
import dash_mantine_components as dmc
from frontend.spatial import quality_figure, graph3d
from frontend.components import badge, graph, grid, kpi, metric_rows, num, panel
from frontend.charts import quality_chart


def analysis(data, decision, mode, defect):
    snap = data["snapshot"]
    symptoms = {"capacity": "특정 셀 전압의 급격한 상승·하락", "welding": "셀 전압 누락·저하", "sensor": "측정값의 결측·범위 이탈", "wire": "인접 셀 사이의 전압 차이"}
    cards = [html.Div([html.Strong(item["label"]), html.Span(symptoms[item["id"]]),
        html.Small(item["status"])], className="quality-defect-info") for item in data["defectEvidence"]]
    temperature = mode == "temperature"
    stats, unit = (snap["temperature"], "°C") if temperature else (snap["voltage"], "V")
    basis = "원본" if data["basis"] == "raw" else "보정"
    spread = stats["range"] if temperature or stats["range"] is None else stats["range"] * 1000
    metrics = [("최저", stats["min"], unit), ("평균", stats["mean"], unit), ("최고", stats["max"], unit),
               ("온도 편차" if temperature else "전압 차이 ΔV", spread, "°C" if temperature else "mV")]
    point = next(p for p in snap["cells"] if p["id"] == data["selectedCell"])
    invalid = [p for p in snap["cells"] + snap["temperatures"] if p["invalid"]]
    # Always expose the selected cell and flagged channels, including hidden raw faults.
    rows = [{"채널": p["id"], "원본": p["raw"], "보정": p["clean"], "단위": "V" if "CV" in p["id"] else "°C", "확인": p["reason"]}
            for p in [point] + [p for p in invalid if p["id"] != point["id"]]]
    context = f"{data['testId']} · {data['modeLabel']} · {snap['progressPct']}% · {snap['index']+1:,}/{data['rows']:,}행 · {basis}값"
    return html.Div([
        html.Div([kpi("AI 품질 상태", "검토 필요" if data["abnormalPointCount"] else "이상 미탐지", "시험 전체 · PCA 정상/이상 탐지", "danger" if data["abnormalPointCount"] else "accent"),
            kpi("이상 구간", data["abnormalSegmentCount"], f"시험 전체 · 이상 시점 {data['abnormalPointCount']:,}개", unit="구간"),
            kpi("선택 셀", data["selectedCell"], f"선택 시점 {snap['progressPct']}%"),
            kpi("작업자 판정", decision, "작업자 기록 기준", "accent")], className="kpi-strip"),
        html.Div([
            panel("배터리 셀 검사", [
                html.Div([html.Div([html.Small(label), html.Strong(f"{num(value, 2 if temperature or label.endswith('ΔV') else 3)} {u}")]) for label, value, u in metrics], className="quality-snapshot-stats"),
                graph3d(quality_figure(data,mode),"quality-3d"),
                html.Div(f"{'온도 32채널' if temperature else '전압 176셀'} · 유효값 {stats['available']}개 · 빨강: 원본값 확인 필요", className="section-note")], context),
            panel("선택 셀 · "+data['selectedCell'], [
                html.Div([html.Strong("선택 셀 · 원본 / 보정"), html.B(f"{num(point['raw'],3)} / {num(point['clean'],3)} V"), html.Small(point["reason"])], className="quality-reading"),
                html.P("4대 불량 유형 · 확인 항목",className="inspector-label"),html.Div(cards,className="quality-defect-list"),
                html.P("선택 셀의 확정 진단이 아닙니다.",className="quality-type-note"),
                html.Div([html.Details([html.Summary("출처 확인"),html.Div([html.P([html.Strong(item["label"]),html.Br(),item["evidence"],html.Br(),item["source"]]) for item in data["defectEvidence"]])],className="source-details"),
                    dmc.Button("판정·조치로 이동",id="quality-go-review",className="spaced-button")],className="quality-inspector-actions")])], className="quality-diagnostic-grid"),
        html.P("16모듈 논리 배치 · 실제 팩 형상과 다름 · 진한 테두리: 선택 셀",className="spatial-disclaimer"),
        html.Details([html.Summary('선택 셀 추세·원본 측정 근거'),html.Div([panel("선택 셀과 모듈 평균 추세", graph(quality_chart(data)), f"{basis}값 · 선택 시점까지 표시 · PCA 점수와 이상 구간은 저장된 모델 결과"),
            panel("원본값 확인", grid(rows, "quality-raw-grid", height=285), f"선택 셀 + 선택 시점 원본 결측·범위 이탈 {len(invalid)}개 채널 · 전압 2–5 V / 온도 −20–100 °C", badge("모델 전처리 범위"))], className="quality-diagnostic-grid"),
        panel("시험 전체 우선 확인 셀", grid(data["suspectedCells"], "main-grid", height=280), "이상 구간 전체 기준 · 행 선택 시 해당 셀로 이동")],className='evidence-disclosure')],className="quality-analysis")
