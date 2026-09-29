"""Demand workflow: operational comparison first, research evidence on validation."""
from dash import html, Input, Output, State, ctx, no_update
import dash_mantine_components as dmc
import plotly.graph_objects as go
from frontend.components import panel, graph, grid, kpi, metric_rows, callout, num, BLUE, CYAN, RED
from frontend.palette import LIGHT, BORDER
from frontend.charts import demand_chart, layout


def analysis(data, tab, actions, query, direction, columns):
    if (data.get('partInfo') or {}).get('quarantined'):
        return callout('원본 기록 충돌 · 예측 제외','원본 정정 전까지 Part 21·26은 예측과 발주 검토에서 제외합니다.','warning'), []
    unavailable = actions.get('__unavailable__')
    rows = []
    for r in data['rows']:
        saved = actions.get(f"demand:{r['part']}:{r['date']}:{r['model']}")
        status = ('조회 불가' if unavailable else '재확인 필요' if saved and saved.get('dataVersion')!=data['version'] else
                  '확인 완료' if saved and saved.get('decision')=='reviewed' else '미확인' if r['review'] else '계획 범위')
        rows.append({**r,'status':status,'direction':r['direction'] if r['review'] else '계획 범위'})
    candidates = [r for r in rows if r['review'] and (not query or query.lower() in r['part'].lower())]
    counts = {'all':len(candidates),'up':sum(r['gap']>0 for r in candidates),'down':sum(r['gap']<0 for r in candidates),
              'open':None if unavailable else sum(r['status']!='확인 완료' for r in candidates)}
    if tab=='review':
        rows = [r for r in candidates if direction=='all' or direction=='open' and r['status'] in ['미확인','재확인 필요'] or direction=='up' and r['gap']>0 or direction=='down' and r['gap']<0]
    table = panel('발주 검토 목록' if tab=='review' else '부품별 계획 차이',grid(rows,'main-grid',columns,425 if tab=='review' else 310),
                  '행 선택 → 부품 상세·확인 처리 · 현재 목록 CSV 내보내기')
    context = (callout('선택 조건의 예측 없음','날짜 또는 부품을 변경하세요.','warning') if not rows else
               html.Div([html.Strong(f"{data.get('originDate','')} → {data['date']}"),html.Span('D+3 일별 최종 ERP 발주 계획량 예측 · 실측 소비량 아님')],className='context-line'))
    if tab=='review':
        buttons = dmc.Group([dmc.Button(f'{label} {num(counts[key])}',id='d-count-'+key,variant='filled' if direction==key else 'outline',
                    className='review-direction-'+key, **{'aria-pressed':str(direction==key).lower()},
                    disabled=key=='open' and unavailable) for key,label in [('all','전체'),('up','↑ 상향'),('down','↓ 하향'),('open','미확인')]],gap='xs')
        return html.Div([context,panel('검토 방향과 확인 상태',buttons,'검색된 검토 대상 기준 · 계획 차이가 max(10개, 계획량의 20%) 이상'),table]),rows
    chart = demand_chart(data['trend'])
    if data['model']!=data['config']['auxiliaryModel']:
        chart.add_trace(go.Scatter(x=[r['date'] for r in data['trend']],y=[r['auxiliary'] for r in data['trend']],
                       name=data['config']['auxiliaryModel']+' 보조 · 클릭하여 보기',visible='legendonly',mode='lines',line={'color':CYAN,'dash':'dash'}))
    ranked = sorted(rows,key=lambda r:abs(r['gap']),reverse=True)[:8]
    gaps = go.Figure(go.Bar(y=[r['part'] for r in ranked],x=[r['gap'] for r in ranked],orientation='h',
        marker_color=[RED if r['gap']>0 else LIGHT for r in ranked],width=.52,
        text=[f"{r['gap']:+,.0f}" for r in ranked],textposition='outside',cliponaxis=False,
        hovertemplate='%{y}<br>예측 − 계획 %{x:+,.0f}개<extra></extra>'))
    layout(gaps,310,'부품')
    gaps.update_layout(showlegend=False,margin={'l':75,'r':48,'t':12,'b':40})
    gap_values = [0, *[r['gap'] for r in ranked]]
    gap_span = max(max(gap_values)-min(gap_values), 1)
    gaps.update_xaxes(title='예측 − 계획 (개)',showgrid=True,gridcolor=BORDER,zeroline=True,zerolinecolor=BLUE,
                     range=[min(gap_values)-gap_span*.18,max(gap_values)+gap_span*.18],nticks=5)
    gaps.update_yaxes(autorange='reversed',title=None,showgrid=False)
    return html.Div([context,html.Div([
        kpi('선택 모델 예상량',num(data['forecast']),data['model'],unit='개'),
        kpi('기존 D+3 계획',num(data['plan']),data['date'],unit='개'),
        kpi('계획 대비 차이',f"{data['gap']:+,.0f}",num(data['gapPct'],2)+'%','danger' if data['gap']>0 else 'accent',unit='개'),
        kpi('검토 필요 부품',data['reviewCount'],f"상향 {counts['up']} · 하향 {counts['down']}",unit='개')],className='kpi-strip'),
        html.Div([panel('D+3 계획·예측 비교',graph(chart),'범례를 눌러 보조 예측 표시 · 최종 기록량은 사후 비교값'),
                  panel('검토 방향',[metric_rows([('상향 검토',f"{counts['up']}개",'예측이 계획보다 큰 부품'),('하향 검토',f"{counts['down']}개",'예측이 계획보다 작은 부품'),('확인 대기',num(counts['open'])+'개' if counts['open'] is not None else '조회 불가','작업자 확인 기준')]),
                       html.Details([html.Summary('예측 해석 기준'),html.P(f"운영 기본 {data['config']['primaryModel']} · 보조 {data['config']['auxiliaryModel']}. 재고·리드타임은 포함되지 않습니다.")],className='source-details')])],className='grid-main'),
        html.Div([panel('계획 차이가 큰 부품',graph(gaps),'절대 차이 상위 8개 · 빨강 상향 / 파랑 하향'),table],className='demand-comparison-grid'),
        html.Details([html.Summary('계획 변경 이력'),panel('같은 목표일의 계획 변경 이력',grid(data['planHistory'],'d-plan-history',height=275),
              '5·4·3일 전은 예측 당시 확인 가능 · 2·1일 전은 사후 정보')],className='demand-details')],className='demand-workspace'),rows


def validation_panels(data):
    labels = ['기준일 최종 기록','기준일 D+3 계획','기준일 D+4 계획','기준일 D+5 계획','D+3 최종 기록']
    heat = go.Figure(go.Heatmap(z=data['correlation'],x=labels,y=labels,zmin=-1,zmax=1,zmid=0,
                    colorscale=[[0,RED],[.5,'#FFFFFF'],[1,BLUE]],texttemplate='%{z:.2f}',hoverongaps=False,
                    hovertemplate='%{y} × %{x}<br>r=%{z:.3f}<extra></extra>'))
    layout(heat,365);heat.update_layout(margin={'l':130,'r':20,'t':15,'b':95});heat.update_xaxes(tickangle=-25)
    top = data['importance'][:12]
    remaining = sum(r['percent'] for r in data['importance'][12:])
    if remaining:top=top+[{'feature':'기타 변수 합계','percent':remaining}]
    def feature_label(name):
        for suffix,prefix in [('_origin','기준일'),('_lag1','1일 전'),('_lag2','2일 전')]:
            if name.endswith(suffix):
                base=name[:-len(suffix)]
                return prefix+' '+('최종 기록' if base=='actual_d' else base.replace('plan_d','D+')+' 계획')
        return {'dow_sin':'요일 주기 (sin)','dow_cos':'요일 주기 (cos)','month':'월','days_since_start':'관측 경과일',
                'is_weekend':'주말 여부','part_number (전체 인코딩)':'부품 구분 (전체)'}.get(name,name)
    fig = go.Figure(go.Bar(x=[r['percent'] for r in top][::-1],y=[feature_label(r['feature']) for r in top][::-1],orientation='h',marker_color=BLUE,
                         text=[f"{r['percent']:.2f}%" for r in top][::-1],textposition='auto'))
    layout(fig,365,'변수');fig.update_layout(margin={'l':190,'r':30,'t':10,'b':35});fig.update_xaxes(title='전체 total gain 비율 (%)')
    q=data['quality']
    quality = [{'단계':'원본 기록','건수':q['source_rows'],'근거':f"{q['source_parts']}부품 · {q['source_columns']}열"},
        {'단계':'동일 시각 수량 충돌','건수':q['conflict_rows'],'근거':f"{q['conflicting_timestamp_groups']}그룹 · Part 21·26 전체 격리"},
        {'단계':'일별 최종 기록','건수':q['daily_rows'],'근거':f"{q['daily_parts']}부품 · 관측 {q['observed_dates']}일 · 0 유지"},
        {'단계':'반복 검증 상관분석','건수':data['correlationRows'],'근거':'선택 부품의 외부 CV 목표일 · 신규 독립 시험 아님'}]
    columns = [{'field':k,'headerName':k,**({'valueFormatter':{'function':"params.value == null ? '—' : Number(params.value).toFixed(3)"}} if k in ['MAE','Bias','상관계수'] else {})} for k in data['metrics'][0]]
    return html.Div([callout('검증 근거 · '+data['part'],
        f"{data['period']['start'] or '자료 없음'} ~ {data['period']['end'] or '—'} 외부 시간순 CV 목표일 · 날짜 필터와 별개의 고정 검증 범위 · {data['version']}"),
        panel('계획 발표 시점별 오차와 편향',grid(data['metrics'],'d-plan-metrics',columns,height=280),
              '모든 시점의 계획이 있는 동일 부품·목표일만 비교 · Bias=계획−최종 기록량, 음수는 과소 · 2·1일 전은 사후 참고'),
        html.Div([panel('입력 변수와 목표값 상관관계',graph(heat),'Pearson · CV 표본 전체의 기술 통계 · 부품 규모 영향 포함 · 인과·성능 아님 · 상수/부족 표본은 공란'),
                  panel('XGBoost 비교 모델의 변수 중요도',graph(fig),data['importanceMethod']+' · 전체 부품 고정')],className='grid-equal'),
        callout('중요도의 해석 범위','XGBoost는 비교 모델입니다. 7일 이동평균·LSTM의 설명이나 개별 예측 기여도가 아닙니다. D+n은 각 기록일 기준 계획입니다. 상관된 변수는 중요도가 분산될 수 있습니다.'),
        html.Details([html.Summary('데이터 처리 근거 · 원본 충돌과 수량 정의'),grid(quality,'d-quality-evidence',height=250),
            html.P(f"Total과 시간대 합계 불일치: 당일 {q['total_slot_mismatch_rows']['actual_d']}행, D+3 {q['total_slot_mismatch_rows']['plan_d3']}행, D+4 {q['total_slot_mismatch_rows']['plan_d4']}행. 현재 기준은 원본 Total이며 수량 정의를 임의 전환하지 않습니다.",className='section-note')],className='demand-details')])


def register(app):
    @app.callback(Output('direction-filter','value',allow_duplicate=True),
        *[Input('d-count-'+key,'n_clicks',allow_optional=True) for key in ['all','up','down','open']],prevent_initial_call=True)
    def filter_counts(*clicks):
        return ctx.triggered_id.replace('d-count-','') if any(clicks) else no_update
