from dash import dcc, html
import dash_mantine_components as dmc
from frontend.components import badge, callout, graph, grid, icon, kpi, metric_rows, num, panel
from frontend.charts import demand_chart, maintenance_chart, maintenance_heat, model_chart, quality_chart, quality_heat

DOMAIN = {"overview": "통합 현황", "demand": "공급망 예측", "maintenance": "예지보전", "quality": "품질 보증"}
TABS = {"overview": [("summary", "통합 요약")],
        "demand": [("analysis", "예측·계획"), ("review", "발주 검토")],
        "maintenance": [("analysis", "설비 점검"), ("review", "이벤트 기록")],
        "quality": [("analysis", "검사"), ("review", "판정·조치")]}


def section_note(text):
    return html.Div(text, className="section-note")


def overview(data, meta, actions):
    from frontend.overview_workspace import overview as workspace
    return workspace(data, meta, actions)


DEMAND_COLUMNS = [{"field": "part", "headerName": "부품", "maxWidth": 140}, {"field": "forecast", "headerName": "선택 예측 (개)", "type": "numericColumn"},
    {"field": "plan", "headerName": "기존 계획 (개)", "type": "numericColumn"}, {"field": "gap", "headerName": "계획 대비 (개)", "type": "numericColumn"},
    {"field": "gapPct", "headerName": "차이 (%)", "type": "numericColumn"}, {"field": "direction", "headerName": "검토 방향"}, {"field": "status", "headerName": "확인 상태"}]


def demand_view(data, tab, actions, query, direction):
    from frontend.demand_workspace import analysis
    return analysis(data,tab,actions,query,direction,DEMAND_COLUMNS)


def maintenance_view(data, tab, actions, status, linked_event=None):
    if tab == 'analysis':
        from frontend.maintenance_workspace import analysis
        for cycle in data['cycleDetails']:
            ids = [e['id'] for e in data['events'] if e['event'] in cycle['관련 이벤트'].split(', ')]
            cycle['확인 상태'] = ('조회 불가' if actions.get('__unavailable__') else '해당 없음' if not ids else
                '확인 완료' if all(f'maintenance:{i}' in actions for i in ids) else '미확인')
        return analysis(data), data['cycleDetails']
    events = [{**e, "status": "조회 불가" if actions.get("__unavailable__") else "확인 완료" if f"maintenance:{e['id']}" in actions else "미확인"} for e in data["events"]]
    if status != "all":
        events = [e for e in events if e['status'] == ('미확인' if status == 'open' else '확인 완료')]
    table = panel("이벤트 검토 대장", grid(events, "main-grid", [{"field": "event", "headerName": "이벤트"}, {"field": "severity", "headerName": "수준"}, {"field": "type", "headerName": "유형"}, {"field": "start", "headerName": "시작 행"}, {"field": "end", "headerName": "종료 행"}, {"field": "rows", "headerName": "지속 행"}, {"field": "maxRisk", "headerName": "최대 위험비"}, {"field": "status", "headerName": "확인 상태"}], 370), "선택 지도·비지도 모델의 OR 경보 · 정상 행이나 120초 초과 수집 공백에서 종료합니다.")
    if linked_event:
        table.children[1].children.selectedRows = [e for e in events if e['id']==linked_event]
    if tab == "review":
        return html.Div([callout("현재 이상을 확인하는 화면", "미래 고장 시점을 예측한 결과가 아닙니다. 실제 SOP는 연결되어 있지 않습니다."), table]), events
    return html.Div([callout("현재 이상 탐지 · 회고 데이터 재생", f"{data['config']['pipelineVersion']} · 입력 결측·무한대·불완전 사이클은 추론 전에 차단합니다. 외부 설비 검증과 현장 SOP는 미완료입니다."),
        html.Div([kpi("점검 경보", len(data["events"]), "수집 공백을 구분한 경보", "danger", "건"), kpi("이상 시점", data["anomalyRows"], f"전체 {len(data['points']):,}행", unit="행"), kpi("최대 위험비", num(data["maxRisk"], 2), "점수 ÷ 모델 임계값", "accent", "배"), kpi("점검 대상 사이클", data['alertCycles'], f"전체 {data['cycles']}회 · 한 행 이상 경보", unit="회")], className="kpi-strip"),
        html.Div([panel("용접 출력과 이상 구간", graph(maintenance_chart(data)), data["run"]),
                  panel("현장 확인 순서", [metric_rows([("01", "출력 신호", "실측과 정상 기준의 차이를 확인"), ("02", "공정 조건", "설정 출력·용접 속도·센싱 상태 점검"), ("03", "이벤트 근거", "전후 신호를 확인하고 검토 기록")]), callout("SOP 미연결", "프로젝트 분석에 근거한 일반 확인 순서입니다. 원인을 확정하지 않습니다.")])], className="grid-main"),
        html.Div([panel("모델별 이상 점수", graph(maintenance_chart(data, True)), "서로 다른 점수를 임계값 비율로 비교"), panel("Cycle × Page 위험 분포", graph(maintenance_heat(data)), "위험비 표시 상한 10 · 상세 값은 신호 차트에서 확인")], className="grid-equal"),
        panel("선택 조합의 사이클 평가", maintenance_policy_evidence(data), "02_OK + 04_NG 회고 평가 전체 기준 · 현재 선택 파일의 지표와 구분합니다."), table]), events


def maintenance_policy_evidence(data):
    score = data.get('operatingEvaluation')
    if not score:
        return section_note('이 조합의 평가 결과가 없습니다.')
    return html.Div([metric_rows([('사이클 정밀도', f"{score['cycle_precision']:.1%}", '경보 사이클 중 실제 이상 비율'),
                                 ('정상 사이클 오경보율', f"{score['cycle_fpr']:.1%}", f"정상 {score['cycle_fp']+score['cycle_tn']}회 중 {score['cycle_fp']}회"),
                                 ('사이클 미탐', str(score['cycle_fn']), '독립 고장 사건 수와 다릅니다.')]),
                     section_note('현장 승인 기준은 미정입니다. 모델 검증 탭에서 시간 검증 실패와 분할 민감도를 함께 확인하세요.')])


def quality_view(data, tab, actions, mode, defect="capacity"):
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
        from frontend.quality_workspace import history_panel
        rows = data.get('history', {}).get('rows', [])
        return html.Div([html.Div([controls, panel("시험 판정 근거", [metric_rows([("운영 모델", "PCA", "T²·SPE 정상 상관 구조"), ("이상 구간", f"{data['abnormalSegmentCount']}개", f"전체 {data['rows']:,}행에서 집계")]), section_note("AI 판단, 시험 정답, 작업자 최종 판정은 서로 다른 정보입니다. 기록은 자동 재학습에 사용하지 않습니다.")])], className="grid-equal"), history_panel(data)]), rows
    from frontend.quality_analysis import analysis
    return analysis(data, decision, mode, defect), data["suspectedCells"]


def validation_view(data, track):
    explanations = {
        "demand": (f"운영 기본: {data.get('config', {}).get('primaryModel', '')}", "내부 5일 검증과 외부 3-fold 시간순 검증으로 선정했습니다. 마지막 7개 목표일은 이전 실험에서 본 기간이므로 회고 평가로 표시합니다.", "학습 → D+3 간격 → 내부 검증 → 시간순 반복검증 → 회고 평가", "MAE·RMSE·WAPE와 과대/과소 예측을 함께 확인하세요. 약 50일 자료로 장기 계절성은 검증되지 않았습니다."),
        "maintenance": (f"기준 탐지기: {data.get('config',{}).get('selectedModel','')}", "01_OK·03_NG로 개발하고 02_OK·04_NG는 선택에 쓰지 않습니다. 네 파일 모두 과거에 본 자료여서 재사용 파일 회고 평가로 표시합니다.", "입력 검사 → 시간 오류 격리 → 사이클 분리 → validation 선택 고정 → 회고 평가", "지도 validation 이상 3건 · 회고 시험 이상 1개 연속 사건. 높은 행 F1이 신규 고장이나 고장 사전 예측 성능을 보장하지 않습니다."),
        "quality": ("PCA 운영 근거와 지도 분류 성능을 함께 확인", "정상 10개 파일로 정상 구조를 학습하고 개발 Test05·09와 잠금 Test03·04·06·07·08을 구분합니다. Random Forest는 학습에 없던 센서 고장 유형의 미탐 한계를 드러냅니다.", "정상 기준 10파일 → 개발 Test05·09 → 잠금 5파일", "불량 Recall을 함께 확인하세요. 셀별 정답은 없어 시점 단위로 평가합니다. MTadGAN은 윈도우 정렬로 평가 행 수가 다릅니다.")}
    title, intro, split, caveat = explanations[track]
    display_rows, display_columns = data['rows'], None
    if track == 'maintenance':
        display_rows = [{**r, 'splitLabel': '개발 검증' if r['split']=='validation' else '회고 시험',
                         'familyLabel':'지도' if r['family']=='supervised' else '비지도'} for r in data['rows']]
        display_columns = [{'field':'model','headerName':'모델','minWidth':230}, {'field':'familyLabel','headerName':'계열'},
                           {'field':'splitLabel','headerName':'평가 구간'}, {'field':'rows','headerName':'행 수'},
                           *[{'field':key,'headerName':label,'valueFormatter':{'function':"params.value == null ? '—' : Number(params.value).toFixed(4)"}}
                             for key,label in [('precision','정밀도'),('recall','Recall'),('f1','F1'),('false_positive_rate','오경보율')]],
                           *[{'field':key,'headerName':label} for key,label in [('tp','정탐 TP'),('fp','오경보 FP'),('fn','미탐 FN'),('tn','정상 TN'),('event_total','실제 이상 사건'),('mean_detection_delay_rows','평균 탐지 지연 (행)')]]]
    content = [callout(title, intro), html.Div([panel("동일 평가 기준의 모델 비교", graph(model_chart(data["rows"], track))),
               panel("검증 설계", [html.Div(split, className="split-flow"), callout("평가 해석", caveat), section_note("최종 시험 성능으로 운영 모델을 재선정하지 않습니다.")])], className="grid-main"),
               panel("모델별 지표 · 혼동행렬 값", grid(display_rows, "main-grid", display_columns, height=385), "열 제목으로 정렬·필터 · TP/FP/FN/TN 등 원 지표를 확인하세요.")]
    if track == "demand":
        content += [panel("회고 평가 · 이전에 관측한 기간", grid(data["holdout"], "validation-holdout", height=300)), panel("Fold별 반복 검증", grid(data["extra"], "validation-extra", height=315)), panel("부품별 오차", grid(data["partErrors"], "validation-parts", height=315))]
    elif track == "quality":
        content += [panel("파일 그룹 교차검증", grid(data["extra"], "validation-extra", height=300)), panel("불량 유형·시험별 결과", grid(data["fileResults"], "validation-files", height=340)), panel("지도모델 변수 중요도", grid(data["features"], "validation-features", height=290))]
    else:
        content.insert(1, callout('시간 검증에서 정상 신호 오경보 확인', '회고 F1과 현장 적용 가능성은 다릅니다. 아래 정상 전용 시간 검증에서 초기 기준모델의 일반화 실패를 확인하세요.', 'warning'))
        operational = [r for r in data['operational'] if r['split'] == 'locked_test']
        op_columns = [{'field':'model','headerName':'모델 / OR 조합','minWidth':330,'flex':2},
                      {'field':'cycle_precision','headerName':'사이클 정밀도'}, {'field':'cycle_recall','headerName':'사이클 Recall'},
                      {'field':'cycle_f1','headerName':'사이클 F1'}, {'field':'cycle_fpr','headerName':'사이클 오경보율'},
                      {'field':'cycle_fp','headerName':'오경보 사이클'}, {'field':'cycle_fn','headerName':'미탐 사이클'},
                      {'field':'false_alarm_count','headerName':'오경보 알림'},
                      {'field':'false_alarms_per_normal_observed_hour','headerName':'정상 관측시간당 오경보','minWidth':210}]
        forward_columns = [{'field':'source_file','headerName':'시험 파일','minWidth':240},
                           {'field':'calibration_end','headerName':'학습·보정 마지막 시점','minWidth':245},
                           {'field':'test_start','headerName':'시험 시작','minWidth':245},
                           {'field':'false_positive_rate','headerName':'정상 행 오경보율'},
                           {'field':'fp','headerName':'오경보 행'}, {'field':'fn','headerName':'미탐 행'}, {'field':'f1','headerName':'행 F1'}]
        for col in op_columns + forward_columns:
            if col['field'] in ['cycle_precision','cycle_recall','cycle_fpr','false_positive_rate']:
                col['valueFormatter'] = {'function':"params.value == null ? '—' : (Number(params.value)*100).toFixed(1)+'%'"}
            elif col['field'] in ['cycle_f1','f1','false_alarms_per_normal_observed_hour']:
                col['valueFormatter'] = {'function':"params.value == null ? '—' : Number(params.value).toFixed(3)"}
        content += [callout('과거 자료만 사용한 시간 검증: 현장 적용 보류', '초기 정상 자료로 적합한 별도 RobustPhaseZ는 01_OK·02_OK 정상 행 모두에 경보를 냈습니다. 이 결과를 숨기거나 주 모델 선택에 재사용하지 않았습니다. 새 시점·설비 자료와 정상 기준 변화 검증이 필요합니다.', 'warning'),
                    panel('사이클 · 운영 조합별 평가', grid(operational,'validation-operational',op_columns,height=360),
                          '한 행 이상 경보면 점검 사이클 · 7개 이상 사이클은 같은 고장 사건입니다. 비율 0.02 = 2%. 가동시간은 관측 간격 기준의 보조 지표입니다.'),
                    panel('정상 전용 시간 검증 · 시험 이전 자료로만 적합', grid(data['forward'],'validation-forward',forward_columns,height=255),
                          '주 모델과 별도로 적합한 기준모델 · 모델 선택에 미사용 · 모든 시험보다 과거의 정상 자료만 사용'),
                    panel('분할 · 정상 분위수 민감도', grid(data['sensitivity'],'validation-sensitivity',height=300), '개발 비율 50/60%와 주 평가 70% 비교 · 보정 분위수 99.5/99.9/99.95% · 결과로 모델을 재선정하지 않음'),
                    panel('반대 파일 방향 스트레스 평가',grid(data['reverse'],'validation-reverse',height=290), '02_OK·04_NG로 개발 → 01_OK·03_NG 평가. 단일 연속 사건을 나눈 개발 분할이므로 독립 사건 검증이 아닙니다.'),
                    panel('데이터 격리와 재현성', metric_rows([('시간 오류 격리',str(data['config']['historicalSplit']['excluded_time_cycles'])+' 사이클','원본 유지 · 사이클 내부 시간 역전'),
                        ('0 출력 격리',str(data['config']['historicalSplit']['excluded_zero_power_cycles'])+' 사이클','정상 기준 학습에서만 제외'),
                        ('모델 버전',data['config']['pipelineVersion'],'계획·원본·코드·저장 모델 해시 기록')])),
                    panel("지도모델 특징 중요도", grid(data["extra"], "validation-extra", height=315))]
    return html.Div(content), data["rows"]


def project_view():
    rows = [{"영역": "공급망 예측", "질문": "3일 뒤 실제 발주량은 얼마나 필요한가?", "단위": "부품 × 일", "방법": "시계열 기준모델 + 지도 회귀", "성공 기준": "누수 없는 MAE·RMSE 개선 검증"},
            {"영역": "예지보전", "질문": "용접 출력에서 어느 구간을 확인해야 하는가?", "단위": "시험 × 39행 사이클", "방법": "지도 분류 + 비지도 이상 탐지", "성공 기준": "미탐·오경보·탐지 지연"},
            {"영역": "품질 보증", "질문": "어느 시험과 셀을 먼저 검사해야 하는가?", "단위": "시험 × 시점", "방법": "PCA 근거 + 지도 분류 비교", "성공 기준": "불량 Recall·F1·유형 일반화"}]
    return html.Div([callout("제조 데이터에서 검토할 대상을 찾고, 판단의 근거를 연결합니다.", "수요·설비·품질의 독립적인 KAMP 데이터를 공통 업무 화면에서 탐색하는 제조 AI 분석 도구입니다."),
        panel("세 가지 업무 질문", grid(rows, "main-grid", height=240)),
        panel("분석에서 의사결정까지", html.Div([html.Div([html.B(f"0{i+1}"), html.Strong(t), html.P(s)]) for i, (t, s) in enumerate([("데이터 품질", "결측·중복·시간 순서 확인"), ("검증 설계", "시간·파일 그룹 분리"), ("모델 비교", "기준모델과 공정한 비교"), ("근거 탐색", "차트·이벤트·셀 상세"), ("검토와 조치", "사람이 최종 판단")])], className="workflow")),
        html.Div([panel("데이터 출처", [html.P("Korea AI Manufacturing Platform (KAMP)"), html.P("공급망 최적화 / 전자부품(배터리팩) 예지보전 / 전자부품(배터리팩) 품질보증 AI 데이터셋"), html.A("KAMP 출처 확인", href="https://www.kamp-ai.kr", target="_blank", rel="noreferrer")]),
                  panel("통합의 기준", [html.P("서로 다른 데이터셋을 행 단위로 병합하지 않습니다."), html.P("공통으로 묶는 것은 분석 상태, 근거 조회, 검토 대상입니다."), badge("공식 시드 v3", "info")])], className="grid-equal")]), rows


def conclusion_view(meta):
    rows = [{"영역": "공급망", "결론": f"{meta['demandPrimary']} 기본 + {meta['demandAuxiliary']} 보조", "가치": "계획과 예측 차이의 우선 검토", "한계": "짧은 관측 기간·재고/리드타임 미포함", "다음 단계": "운영 변수와 신규 기간 검증"},
            {"영역": "예지보전", "결론": "RobustPhaseZ 기본 탐지", "가치": "이상 구간과 공정 근거 연결", "한계": "현재 이상 탐지·제한된 고장 유형", "다음 단계": "실제 설비 신호와 신규 고장 검증"},
            {"영역": "품질", "결론": "PCA (T²·SPE) 운영 근거", "가치": "검사 우선 시험과 셀 탐색", "한계": "셀별 정답 없음·유형별 일반화 차이", "다음 단계": "검사 결과와 작업자 피드백 확보"}]
    return html.Div([callout("모델의 점수보다, 검토 가능한 근거와 일관된 판단 흐름", "운영 화면에서 데이터·검증·한계를 함께 확인하도록 구성했습니다."), panel("결론과 다음 과제", grid(rows, "main-grid", height=240)),
        html.Div([panel("이번 로컬 버전", metric_rows([("완료 범위", "4개 화면", "FastAPI + Dash + 공식 시드"), ("업무 조치", "공유 이력", "작성자·시각·이전 판정 보존"), ("AI 지원", "Gemini RAG", "로컬 FAISS · 본인 전용 대화")])),
                  panel("팀 기여와 산출물", [html.P("세 트랙의 모델·평가 산출물, 공식 DB 시드, 통합 API와 대시보드로 구성됩니다."), html.P("개인별 최종 기여 내역은 아직 확정되지 않아 임의로 표시하지 않았습니다."), badge("재학습 기능은 범위에서 제외")])], className="grid-equal")]), rows
