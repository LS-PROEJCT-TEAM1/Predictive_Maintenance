from dash import dcc, html
import dash_mantine_components as dmc
from frontend.components import badge, callout, graph, grid, icon, kpi, metric_rows, num, panel
from frontend.charts import demand_chart, maintenance_chart, maintenance_heat, model_chart, quality_chart, quality_heat

DOMAIN = {"overview": "통합 현황", "demand": "공급망 예측", "maintenance": "예지보전", "quality": "품질 보증"}
TABS = {"overview": [("summary", "통합 요약"), ("project", "프로젝트 개요"), ("data", "데이터 품질"), ("results", "결과 조회"), ("conclusion", "결론·한계")],
        "demand": [("analysis", "예측 분석"), ("review", "검토 목록"), ("validation", "모델 검증")],
        "maintenance": [("analysis", "이상 분석"), ("review", "이벤트 검토"), ("validation", "모델 검증")],
        "quality": [("analysis", "품질 분석"), ("review", "판정·조치"), ("validation", "모델 검증")]}


def section_note(text):
    return html.Div(text, className="section-note")


def overview(data, meta, actions):
    d, m, q = data["demand"], data["maintenance"], data["quality"]
    labels = {'clear': '이상 없음', 'retest': '재시험 요청', 'hold': '출하 보류', 'reviewed': '확인 완료'}
    quality_record = actions.get(f"quality:{q['testId']}")
    quality_status = '판정 조회 불가' if actions.get('__unavailable__') else labels.get((quality_record or {}).get('decision'), '판정 대기')
    cards = []
    for route, name, glyph, value, unit, label, detail, status in [
        ("demand", "공급망 예측", "chart-no-axes-combined", num(d["recommendedForecast"]), "개", "D+3 예상 발주량", f"계획 대비 +{d['planGapPct']}% · 검토 {d['reviewCount']}개", "검토 필요"),
        ("maintenance", "예지보전", "activity", str(m["eventCount"]), "건", "이상 이벤트", "RobustPhaseZ · 39개 Page 영향", "이상 탐지"),
        ("quality", "품질 보증", "shield-check", str(q["abnormalSegmentCount"]), "구간", "품질 이상 구간", f"PCA (T²·SPE) · {q['suspectedCell']} 우선 확인", quality_status)]:
        cards.append(html.Div([html.Div([html.Span(icon(glyph, 20), className="track-symbol"), html.Strong(name), badge(status, "warning")], className="track-head"),
            html.Div([value, html.Small(unit)], className="track-value"), html.Div(label, className="track-label"),
            html.Div([html.Span(detail), dcc.Link(["분석 열기", icon("arrow-up-right", 15)], href=f"/{route}")], className="track-foot")], className="track-card"))
    queue = [{"priority": a["priority"], "track": a["trackLabel"], "target": a["target"], "title": a["title"], "status": a["statusLabel"], "route": a["detailPath"]} for a in data["actions"]]
    for row, original in zip(queue, data['actions']):
        if actions.get('__unavailable__'):
            row['status'] = '조회 불가'
        elif original['track'] == 'demand':
            key = f"demand:{row['target']}:{d['targetDate']}:{d['operatingModel']}"
            row['status'] = '확인 완료' if key in actions else '미확인'
        elif original['track'] == 'quality':
            row['status'] = labels.get(actions.get(f"quality:{row['target']}", {}).get('decision'), '판정 대기')
        else:
            prefix = f"maintenance:{row['target']}:{meta['defaultSupervised']}:{meta['defaultUnsupervised']}:"
            count = sum(key.startswith(prefix) for key in actions)
            row['status'] = f"확인 {count}/{m['eventCount']}"
    return html.Div([
        html.Div(cards, className="track-cards"),
        html.Div([
            panel("발주 계획과 예측 흐름", graph(demand_chart(data["trend"])), "목표일 기준 · 서로 다른 부품을 날짜별 합산", badge("3일 이동평균", "info")),
            panel("운영 모델의 선택 근거", [metric_rows([("3일 이동평균", "36.95", "공급망 · Walk-forward MAE / 개"), ("RobustPhaseZ", "0.9945", "예지보전 · 잠금 시험 F1"), ("PCA (T²·SPE)", "0.9866", "품질 보증 · 잠금 시험 F1")]), section_note("트랙별 평가 단위가 달라 점수를 서로 직접 비교하지 않습니다.")])], className="grid-main"),
        html.Div([panel("우선 검토 목록", grid(queue, "main-grid", [{"field": "priority", "headerName": "순위", "maxWidth": 75, "minWidth": 65}, {"field": "track", "headerName": "영역", "maxWidth": 125}, {"field": "target", "headerName": "대상", "flex": 1.2}, {"field": "title", "headerName": "검토 내용", "flex": 2.5}, {"field": "status", "headerName": "상태", "maxWidth": 125}], 285), "행을 선택해 분석 화면으로 이동하세요. 예지보전 확인 수는 기본 모델 조합 기준입니다.", badge("공식 시드 · 공유 기록")),
            panel("분석 범위", [metric_rows([("공급망", "117 부품", "최신 목표일 예측 109개"), ("레이저 용접", "4 시험", "현재 이상 탐지 · 시험 데이터 재생"), ("배터리 품질", "5 시험", "잠금 시험 · 176셀 / 32온도")]), section_note(("공유 기록 조회 불가" if actions.get("__unavailable__") else f"공유 검토 상태 {len(actions)}건 · Firebase 저장")), callout("독립 데이터셋", "세 트랙의 원본 행을 합치지 않고 상태와 조치만 통합합니다.")])], className="grid-main")])


DEMAND_COLUMNS = [{"field": "part", "headerName": "부품", "maxWidth": 140}, {"field": "forecast", "headerName": "선택 예측 (개)", "type": "numericColumn"},
    {"field": "plan", "headerName": "기존 계획 (개)", "type": "numericColumn"}, {"field": "gap", "headerName": "계획 대비 (개)", "type": "numericColumn"},
    {"field": "gapPct", "headerName": "차이 (%)", "type": "numericColumn"}, {"field": "direction", "headerName": "검토 방향"}, {"field": "status", "headerName": "확인 상태"}]


def demand_view(data, tab, actions, query, direction):
    rows = []
    for r in data["rows"]:
        status = "조회 불가" if actions.get("__unavailable__") else "확인 완료" if f"demand:{r['part']}:{r['date']}:{r['model']}" in actions else "미확인" if r["review"] else "계획 범위"
        row = {**r, "status": status, "direction": r["direction"] if r["review"] else "계획 범위"}
        if tab == "review" and not r["review"]:
            continue
        if query and query.lower() not in r["part"].lower():
            continue
        if direction != "all" and (not r["review"] or (r["gap"] > 0) != (direction == "up")):
            continue
        rows.append(row)
    table = panel("발주 검토 목록" if tab == "review" else "부품별 계획 차이", grid(rows, "main-grid", DEMAND_COLUMNS, 425 if tab == "review" else 310), "행 선택 → 부품 상세 · 헤더에서 정렬/필터 · 현재 표시 목록 CSV 다운로드")
    if tab == "review":
        return html.Div([callout("기준이 같은 검토 목록", "계획 차이가 계획량의 20%와 10개 중 큰 값 이상인 부품을 검토합니다. 확인 상태는 날짜·모델별로 구분됩니다."), table]), rows
    content = [html.Div([kpi("선택 모델 예상량", num(data["forecast"]), data["model"], unit="개"), kpi("기존 D+3 계획", num(data["plan"]), data["date"], unit="개"),
                       kpi("계획 대비 차이", f"{data['gap']:+,.0f}", f"{num(data['gapPct'], 2)}%", "accent", "개"), kpi("검토 필요 부품", data["reviewCount"], f"선택일 예측 {data['count']}개 부품", "danger", "개")], className="kpi-strip"),
        html.Div([panel("D+3 발주량 예측", graph(demand_chart(data["trend"])), "계획·실제·예측을 동일한 목표일로 비교합니다."),
                  panel("발주 판단 가이드", [metric_rows([("운영 기본", "이동평균", "검증 MAE 36.9523"), ("학습형 보조", "XGBoost", "검증 MAE 48.3896")]),
                    callout("차이가 큰 부품부터 검토", "예측은 발주 판단의 근거입니다. 재고·조달 리드타임은 포함되어 있지 않습니다."), section_note("신규 CSV 3일 자료를 업로드하면 D+3 이동평균·XGBoost 추론을 직접 시험할 수 있습니다.")])], className="grid-main"), table]
    if not data["count"]:
        content.insert(0, callout("선택한 조건의 예측 없음", "부품은 등록되어 있지만 해당 목표일의 예측 자료가 없습니다. 날짜 또는 부품을 변경하세요.", "warning"))
    return html.Div(content), rows


def maintenance_view(data, tab, actions, status):
    events = [{**e, "status": "조회 불가" if actions.get("__unavailable__") else "확인 완료" if f"maintenance:{e['id']}" in actions else "미확인"} for e in data["events"]]
    if status != "all":
        events = [e for e in events if e['status'] == ('미확인' if status == 'open' else '확인 완료')]
    table = panel("이벤트 검토 대장", grid(events, "main-grid", [{"field": "event", "headerName": "이벤트"}, {"field": "severity", "headerName": "수준"}, {"field": "type", "headerName": "유형"}, {"field": "start", "headerName": "시작 행"}, {"field": "end", "headerName": "종료 행"}, {"field": "rows", "headerName": "지속 행"}, {"field": "maxRisk", "headerName": "최대 위험비"}, {"field": "status", "headerName": "확인 상태"}], 370), "선택한 지도·비지도 모델의 합집합으로 연속 이벤트를 계산합니다.")
    if tab == "review":
        return html.Div([callout("현재 이상을 확인하는 화면", "미래 고장 시점을 예측한 결과가 아닙니다. 실제 SOP는 연결되어 있지 않습니다."), table]), events
    return html.Div([html.Div([kpi("이상 이벤트", len(data["events"]), "선택 모델 조합", "danger", "건"), kpi("이상 시점", data["anomalyRows"], f"전체 {len(data['points']):,}행", unit="행"), kpi("최대 위험비", num(data["maxRisk"], 2), "점수 ÷ 모델 임계값", "accent", "배"), kpi("용접 사이클", data["cycles"], "사이클당 39개 Page", unit="회")], className="kpi-strip"),
        html.Div([panel("용접 출력과 이상 구간", graph(maintenance_chart(data)), data["run"]),
                  panel("현장 확인 순서", [metric_rows([("01", "출력 신호", "실측과 정상 기준의 차이를 확인"), ("02", "공정 조건", "설정 출력·용접 속도·센싱 상태 점검"), ("03", "이벤트 근거", "전후 신호를 확인하고 검토 기록")]), callout("SOP 미연결", "프로젝트 분석에 근거한 일반 확인 순서입니다. 원인을 확정하지 않습니다.")])], className="grid-main"),
        html.Div([panel("모델별 이상 점수", graph(maintenance_chart(data, True)), "서로 다른 점수를 임계값 비율로 비교"), panel("Cycle × Page 위험 분포", graph(maintenance_heat(data)), "위험비 표시 상한 10 · 상세 값은 신호 차트에서 확인")], className="grid-equal"), table]), events


def quality_view(data, tab, actions, mode):
    record = actions.get(f"quality:{data['testId']}")
    decision_labels = {"clear": "이상 없음", "retest": "재시험 요청", "hold": "출하 보류"}
    decision = "조회 불가" if actions.get("__unavailable__") else decision_labels.get(record["decision"], "미확정") if record else "미확정"
    controls = panel("작업자 판정", [callout("직원 공용 업무 기록", "확인 후 저장하면 작성자와 시각이 함께 기록됩니다."),
        dmc.Select(id="decision-code", label="최종 판정", value="retest", allowDeselect=False, data=[{"label": v, "value": k} for k, v in decision_labels.items()], persistence=data['testId'], persistence_type='memory'),
        dmc.Textarea(id="decision-note", label="검토 메모", placeholder="확인한 근거와 후속 조치를 입력하세요.", autosize=True, minRows=3, inputProps={"maxLength": 2000}, persistence=data['testId'], persistence_type='memory'),
        section_note('작성 중인 판정·메모는 시험별로 탭 이동 동안 유지됩니다. 새로고침하면 저장하지 않은 내용은 사라집니다.'),
        dmc.Button("판정 확인", id="prepare-quality", leftSection=icon("check-check"), className="spaced-button"),
        html.Div([badge(decision, "warning"), html.P('기록 저장소 연결 후 다시 확인하세요.' if actions.get('__unavailable__') else record.get("note", "") if record else "아직 확정한 판정이 없습니다.")], className="decision-current")])
    if tab == "review":
        rows = [{"시험": data["testId"], "판정": decision, "메모": record.get("note", "") if record else "", "시각": record.get("at", "") if record else "", "저장범위": "Firebase"}]
        return html.Div([html.Div([controls, panel("시험 판정 근거", [metric_rows([("운영 모델", "PCA", "T²·SPE 정상 상관 구조"), ("이상 구간", f"{data['abnormalSegmentCount']}개", f"전체 {data['rows']:,}행에서 집계"), ("우선 확인 셀", data["suspectedCells"][0]["셀"] if data["suspectedCells"] else "없음", "예측 이상 구간에서 산정")]), section_note("AI 판단, 시험 정답, 작업자 최종 판정은 서로 다른 정보입니다.")])], className="grid-equal"), panel("최근 저장 판정", grid(rows, "main-grid", height=210))]), rows
    rows = data["suspectedCells"]
    return html.Div([html.Div([kpi("AI 품질 상태", "검토 필요" if data["abnormalPointCount"] else "정상", "시험 전체 결과", "danger" if data["abnormalPointCount"] else "accent"),
        kpi("이상 구간", data["abnormalSegmentCount"], f"이상 시점 {data['abnormalPointCount']:,}개", unit="구간"), kpi("선택 셀", data["selectedCell"], f"{data['rows']:,}개 시점 분석"), kpi("작업자 판정", decision, "Firebase 저장 상태", "accent")], className="kpi-strip"),
        html.Div([panel("시험 진행률별 이상 근거", graph(quality_chart(data)), "셀·모듈 전압과 PCA 점수를 비교합니다. 붉은 구간은 예측 이상 구간입니다."),
                  panel("검사 우선순위", [metric_rows([("운영 모델", "PCA", "T² > 20.4268 또는 SPE > 2.046"), ("지속 조건", "10 시점", "연속 초과 구간의 저장 판정"), ("평가 F1", "0.9866", "잠금 시험 · 시점 단위")]), callout("정답 라벨을 사용하지 않는 위치 점수", "예측 이상 구간의 셀 편차를 이용해 우선 확인 위치를 보여줍니다.")])], className="grid-main"),
        html.Div([panel("기여 위치 지도", graph(quality_heat(data, mode), "quality-heatmap"), "대표 시점 · 셀 지도를 클릭하면 선택 셀의 신호를 확인합니다."),
                  panel("우선 확인 셀", grid(rows, "main-grid", height=365), "정답이 아닌 예측 구간의 평균 절대 Z-score")], className="grid-equal")]), rows


def validation_view(data, track):
    explanations = {
        "demand": ("운영 기본은 3일 이동평균", "3-fold purged walk-forward에서 선정했습니다. 입력과 목표 사이 D+3 간격을 유지하고 최종 7개 목표일은 독립 holdout으로 분리합니다.", "시간순 train → 3일 간격 → validation → 독립 holdout", "MAE·RMSE·WAPE와 과대/과소 예측을 함께 확인하세요. 약 50일 자료로 장기 계절성은 검증되지 않았습니다."),
        "maintenance": ("RobustPhaseZ를 기본 탐지기로 사용", "지도·비지도 모델은 같은 잠금 시험에서 비교합니다. 01_OK·03_NG로 개발하고 02_OK·04_NG를 독립 시험으로 보존합니다.", "39행 사이클 단위 개발 분리 → validation 선정 → 파일 단위 잠금 시험", "현재값 기반 이상 탐지입니다. 새로운 고장 유형에 대한 외부 검증이 필요하며 지도모델은 class_weight로 불균형을 처리합니다."),
        "quality": ("PCA 운영 근거와 지도 분류 성능을 함께 확인", "정상 10개 파일로 정상 구조를 학습하고 개발 Test05·09와 잠금 Test03·04·06·07·08을 구분합니다. Random Forest는 학습에 없던 센서 고장 유형의 미탐 한계를 드러냅니다.", "정상 기준 10파일 → 개발 Test05·09 → 잠금 5파일", "불량 Recall을 함께 확인하세요. 셀별 정답은 없어 시점 단위로 평가합니다. MTadGAN은 윈도우 정렬로 평가 행 수가 다릅니다.")}
    title, intro, split, caveat = explanations[track]
    content = [callout(title, intro), html.Div([panel("동일 평가 기준의 모델 비교", graph(model_chart(data["rows"], track))),
               panel("검증 설계", [html.Div(split, className="split-flow"), callout("평가 해석", caveat), section_note("최종 시험 성능으로 운영 모델을 재선정하지 않습니다.")])], className="grid-main"),
               panel("모델별 지표 · 혼동행렬 값", grid(data["rows"], "main-grid", height=385), "열 제목으로 정렬·필터 · TP/FP/FN/TN 등 원 지표를 확인하세요.")]
    if track == "demand":
        content += [panel("독립 최종 holdout", grid(data["holdout"], "validation-holdout", height=300)), panel("Fold별 반복 검증", grid(data["extra"], "validation-extra", height=315)), panel("부품별 오차", grid(data["partErrors"], "validation-parts", height=315))]
    elif track == "quality":
        content += [panel("파일 그룹 교차검증", grid(data["extra"], "validation-extra", height=300)), panel("불량 유형·시험별 결과", grid(data["fileResults"], "validation-files", height=340)), panel("지도모델 변수 중요도", grid(data["features"], "validation-features", height=290))]
    else:
        content += [panel("지도모델 특징 중요도", grid(data["extra"], "validation-extra", height=315))]
    return html.Div(content), data["rows"]


def project_view():
    rows = [{"영역": "공급망 예측", "질문": "3일 뒤 실제 발주량은 얼마나 필요한가?", "단위": "부품 × 일", "방법": "시계열 기준모델 + 지도 회귀", "성공 기준": "누수 없는 MAE·RMSE 개선 검증"},
            {"영역": "예지보전", "질문": "용접 출력에서 어느 구간을 확인해야 하는가?", "단위": "시험 × 39행 사이클", "방법": "지도 분류 + 비지도 이상 탐지", "성공 기준": "미탐·오경보·탐지 지연"},
            {"영역": "품질 보증", "질문": "어느 시험과 셀을 먼저 검사해야 하는가?", "단위": "시험 × 시점", "방법": "PCA 근거 + 지도 분류 비교", "성공 기준": "불량 Recall·F1·유형 일반화"}]
    return html.Div([callout("제조 데이터에서 검토할 대상을 찾고, 판단의 근거를 연결합니다.", "수요·설비·품질의 독립적인 KAMP 데이터를 공통 업무 화면에서 탐색하는 제조 AI 분석 도구입니다."),
        panel("세 가지 업무 질문", grid(rows, "main-grid", height=240)),
        panel("분석에서 의사결정까지", html.Div([html.Div([html.B(f"0{i+1}"), html.Strong(t), html.P(s)]) for i, (t, s) in enumerate([("데이터 품질", "결측·중복·시간 순서 확인"), ("검증 설계", "시간·파일 그룹 분리"), ("모델 비교", "기준모델과 공정한 비교"), ("근거 탐색", "차트·이벤트·셀 상세"), ("검토와 조치", "사람이 최종 판단")])], className="workflow")),
        html.Div([panel("데이터 출처", [html.P("Korea AI Manufacturing Platform (KAMP)"), html.P("공급망 최적화 / 전자부품(배터리팩) 예지보전 / 전자부품(배터리팩) 품질보증 AI 데이터셋"), html.A("KAMP 출처 확인", href="https://www.kamp-ai.kr", target="_blank", rel="noreferrer")]),
                  panel("통합의 기준", [html.P("서로 다른 데이터셋을 행 단위로 병합하지 않습니다."), html.P("공통으로 묶는 것은 분석 상태, 근거 조회, 검토 대상입니다."), badge("공식 시드 v2", "info")])], className="grid-equal")]), rows


def conclusion_view():
    rows = [{"영역": "공급망", "결론": "이동평균 기본 + XGBoost 보조", "가치": "계획과 예측 차이의 우선 검토", "한계": "짧은 관측 기간·재고/리드타임 미포함", "다음 단계": "운영 변수와 신규 기간 검증"},
            {"영역": "예지보전", "결론": "RobustPhaseZ 기본 탐지", "가치": "이상 구간과 공정 근거 연결", "한계": "현재 이상 탐지·제한된 고장 유형", "다음 단계": "실제 설비 신호와 신규 고장 검증"},
            {"영역": "품질", "결론": "PCA (T²·SPE) 운영 근거", "가치": "검사 우선 시험과 셀 탐색", "한계": "셀별 정답 없음·유형별 일반화 차이", "다음 단계": "검사 결과와 작업자 피드백 확보"}]
    return html.Div([callout("모델의 점수보다, 검토 가능한 근거와 일관된 판단 흐름", "운영 화면에서 데이터·검증·한계를 함께 확인하도록 구성했습니다."), panel("결론과 다음 과제", grid(rows, "main-grid", height=240)),
        html.Div([panel("이번 로컬 버전", metric_rows([("완료 범위", "4개 화면", "FastAPI + Dash + 공식 시드"), ("업무 조치", "공유 이력", "작성자·시각·이전 판정 보존"), ("AI 지원", "Gemini RAG", "로컬 FAISS · 본인 전용 대화")])),
                  panel("팀 기여와 산출물", [html.P("세 트랙의 모델·평가 산출물, 공식 DB 시드, 통합 API와 대시보드로 구성됩니다."), html.P("개인별 최종 기여 내역은 아직 확정되지 않아 임의로 표시하지 않았습니다."), badge("재학습 기능은 범위에서 제외")])], className="grid-equal")]), rows
