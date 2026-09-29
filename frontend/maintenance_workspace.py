"""A compact maintenance review workflow within the existing three tabs."""
import base64
from urllib.parse import urlencode

import httpx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
import dash_mantine_components as dmc

from frontend.charts import layout, maintenance_heat
from frontend.components import BLUE, CYAN, GRAY, RED, TEXT, badge, callout, grid, kpi, metric_rows, num, panel
from frontend.quality_workspace import confusion_figure
from frontend.spatial import maintenance_figure, graph3d, selection


def plot(fig, id=None):
    props = {'figure':fig,'config':{'displayModeBar':True,'displaylogo':False,'modeBarButtonsToRemove':['select2d','lasso2d','autoScale2d'],'toImageButtonOptions':{'format':'png','filename':'maintenance_evidence'},'responsive':True}}
    if id:
        props['id'] = id
    return dcc.Graph(**props)


def power_chart(data, points, score=False):
    fig = go.Figure()
    fields = [('supRatio',data['supervised'],BLUE),('unsupRatio',data['unsupervised'],CYAN)] if score else [('power','실측 출력',TEXT),('expected','정상 기준',CYAN)]
    for key,name,color in fields:
        fig.add_trace(go.Scatter(x=[p['row'] for p in points],y=[p[key] for p in points],mode='lines',name=name,
            line={'color':color,'width':1.7}, customdata=[[p['time'],p['cycle'],p['page']] for p in points],
            hovertemplate='%{y:.3f}<br>%{customdata[0]}<br>사이클 %{customdata[1]} · 지점 %{customdata[2]}<extra>'+name+'</extra>'))
    if score:
        fig.add_hline(y=1,line_color=RED,line_dash='dot',annotation_text='경보 기준 1배')
    else:
        alarms = [p for p in points if p['prediction']]
        fig.add_trace(go.Scatter(x=[p['row'] for p in alarms],y=[p['power'] for p in alarms],name='경보 지점',mode='markers',marker={'color':RED,'size':5}))
    layout(fig,230 if score else 270,'점수 / 모델 임계값' if score else '출력 (W)')
    fig.update_xaxes(title='원본 행 ID (0부터)', rangeslider={'visible':not score,'thickness':.09})
    return fig


def cycle_columns():
    return [{'field':'cycle','headerName':'사이클','minWidth':85,'maxWidth':100},
            {'field':'상태','minWidth':120,'cellClassRules':{'maintenance-alert':"x === '점검 대상'"}},
            {'field':'시작 시각','minWidth':195,'flex':1.6},
            *[{'field':s,'minWidth':100,'maxWidth':120} for s in ['경보 지점','출력 0','학습 밖','모델 불일치','최대 위험비']],
            {'field':'확인 상태','minWidth':115},{'field':'관련 이벤트','minWidth':140}]


def analysis(data):
    return html.Div([
        html.Div([kpi('점검 대상 사이클',data['alertCycles'],f"전체 {data['cycles']}회 · 한 지점 이상 경보",'danger','회'),
            kpi('경보 이벤트',len(data['events']),'정상 복귀·수집 공백 기준',unit='건'),
            kpi('경보 지점',data['anomalyRows'],f"전체 {len(data['points']):,}행",unit='행'),
            kpi('학습 밖 레시피',data['unseenRows'],'정상 기준 학습 조건과 비교','accent',unit='행')],className='kpi-strip'),
        html.Div([dmc.Select(id='m-cycle',label='상세 조회 사이클',data=[{'label':'전체 사이클','value':'all'}]+[{'label':f"사이클 {c['cycle']} · {c['상태']}",'value':str(c['cycle'])} for c in sorted(data['cycleDetails'],key=lambda c:c['cycle'])],value=str(next((c['cycle'] for c in data['cycleDetails'] if c['경보 지점']), data['cycleDetails'][0]['cycle'])),allowDeselect=False,searchable=True,className='context-select',persistence=data['run'],persistence_type='session'),
                  html.Span('사이클 선택 → 지점 확인 → 이벤트 기록',className='workflow-hint'),
                  html.P(f"공정 조건 확인 · 학습 밖 레시피 {data['unseenRows']}행 · 설정값 확인 필요",className='maintenance-condition-warning') if data['unseenRows'] else None],className='filter-toolbar compact'),
        html.Div(id='m-cycle-chart'),
        html.Details([html.Summary('출력 신호·이벤트 근거'),html.Div(id='m-cycle-detail')],className='evidence-disclosure'),
        html.Details([html.Summary('사이클별 점검 목록'),panel('사이클별 점검 목록',grid(data['cycleDetails'],'m-cycle-grid',cycle_columns(),300),
              '점검 대상·출력 0·위험비 순 · 행 선택 시 상세 조회',
              dmc.Button('목록 CSV',id='m-cycle-export',variant='outline',size='xs'))],className='evidence-disclosure'),
        html.Details([html.Summary('전체 사이클 위험 분포'),plot(maintenance_heat(data),'m-risk-map')],className='maintenance-explore'),
        html.P(f"{data['config']['pipelineVersion']} · 현재 이상 탐지 / 과거 시험 재생 · 현장 SOP와 미래 고장 예측은 미검증",className='section-note')],className='maintenance-analysis')


def cycle_detail(data, selected, selected_point=None):
    available = {str(p['cycle']) for p in data['points']}
    chosen = selected if selected in available else 'all'
    points = [p for p in data['points'] if chosen=='all' or str(p['cycle'])==chosen]
    events = [e for e in data['events'] if e['start']<=points[-1]['row'] and e['end']>=points[0]['row']]
    overview = html.Div([
        panel('출력 기준·편차 · '+('전체 사이클' if chosen=='all' else '사이클 '+chosen),plot(power_chart(data,points),'m-power-chart'),data['referenceNotice']),
        panel('확인할 근거',metric_rows([('경보 지점',f"{sum(p['prediction'] for p in points)} / {len(points)}",'선택한 사이클·구간'),
            ('출력 0',str(sum(p['power']==0 for p in points))+'행','출력·센싱 상태와 설정값 확인'),
            ('모델 판정 불일치',str(sum(p['agreement']=='판정 불일치' for p in points))+'행','지도·비지도 중 한 모델만 경보'),
            ('학습 밖 레시피',str(sum(p['unseen'] for p in points))+'행','분포 밖 조건 · 고장 원인 확정 아님')]))],className='grid-main')
    content = []
    if chosen!='all':
        initial=next((p for p in points if str(p['row'])==selected_point),next((p for p in points if p['prediction']),points[0]))
        content += [panel('용접 지점 검사 · 사이클 '+chosen,html.Div([html.Div([
            graph3d(maintenance_figure(data,points,initial['row']),'maintenance-3d'),
            html.P('논리 지점판 · 빨강: 경보 / 검정: 출력 0 / 진한 테두리: 선택',className='spatial-disclaimer')]),html.Div([
            dmc.Select(id='m-point-select',label='지점 직접 선택',data=[{'label':f"지점 {p['page']:02d} · {p['power']:,.1f} W",'value':str(p['row'])} for p in points],value=str(initial['row']),allowDeselect=False,searchable=True,className='context-select'),
            html.Div(id='m-point-detail')],className='spatial-inspector')],className='spatial-inspection-grid'), '지점을 눌러 출력·설정·경보 근거를 확인하세요.')]
    else:
        content += [callout('전체 사이클 비교 중','지점 검사에는 사이클을 선택하세요. 아래 목록에서도 선택할 수 있습니다.')]
    content += [html.Details([html.Summary('상세 경보 점수'),panel('선택 구간의 모델별 이상 점수',plot(power_chart(data,points,True)), '임계값 대비 배수 · 고장 확률이 아닙니다.')],className='maintenance-explore'),
        panel('관련 이벤트 · 선택해서 검토 기록',grid(events,'main-grid',[
            {'field':'event','headerName':'이벤트'},{'field':'type','headerName':'유형'},{'field':'start','headerName':'시작 행'},
            {'field':'end','headerName':'종료 행'},{'field':'rows','headerName':'지속 행'},{'field':'maxRisk','headerName':'최대 위험비'}],235),
            '이벤트 선택 시 전후 신호·확인 기록 조회' if events else '선택한 사이클에 연결된 경보 이벤트가 없습니다.')]
    return content[:1], [overview,*content[1:]]


def exploration(data):
    frame = pd.DataFrame(data['points'])
    box, histogram, corr = go.Figure(),go.Figure(),go.Figure()
    rows = []
    for value, group in frame.groupby('setPower'):
        box.add_trace(go.Box(y=group.power,name=f'{value:g}% 선택 시험',boxpoints=False,marker_color=BLUE))
        rows.append({'설정 출력 (%)':value,'측정 행':len(group),'경보 행':int(group.prediction.sum()),
            '경보 비율 (%)':round(group.prediction.mean()*100,2),'학습 밖 행':int(group.unseen.sum()),'평균 실측 (W)':round(group.power.mean(),2)})
    for entry in data['normalSummary']:
        low,q1,median,q3,high = entry['q']
        box.add_trace(go.Box(q1=[q1],median=[median],q3=[q3],lowerfence=[low],upperfence=[high],
            name=f"{entry['setPower']:g}% 정상 학습",marker_color=GRAY))
    for flag,name,color in [(0,'경보 없음',GRAY),(1,'경보 있음',RED)]:
        histogram.add_trace(go.Histogram(x=frame.loc[frame.prediction.eq(flag),'deviation'],name=name,marker_color=color,opacity=.7,nbinsx=40))
    layout(box,260,'실측 출력 (W)')
    layout(histogram,260,'행 수')
    histogram.update_layout(barmode='overlay')
    histogram.update_xaxes(title='실측 − PageNo 정상 기준 (W)')
    names = {'speed':'속도','length':'길이','setPower':'설정 출력','gateOnTime':'GateOnTime','power':'실측 출력'}
    matrix = frame[list(names)].rename(columns=names).corr()
    corr.add_trace(go.Heatmap(z=matrix.values,x=list(matrix.columns),y=list(matrix.index),zmin=-1,zmax=1,
        colorscale=[[0,GRAY],[.5,'#F0F3FA'],[1,BLUE]],hovertemplate='%{y} × %{x}: %{z:.3f}<extra></extra>'))
    layout(corr,280)
    return html.Div([html.Div([panel('설정 출력별 실측 분포',plot(box),data['run']+f" · 정상 학습 {data['normalRows']:,}행 비교 · 정상 학습 수염은 최소/최대"),
                              panel('출력 편차 분포',plot(histogram),'경보 유무 기준 · 정답의 정상/불량 분포와 다릅니다.')],className='grid-equal'),
        html.Div([panel('공정 조건별 확인',grid(rows,'m-condition-grid',height=250),'비율 분모는 각 조건의 전체 행 수입니다.'),
                  panel('변수 상관관계',plot(corr),'선택 파일 전체 · 인과관계가 아닙니다. 일정한 변수의 상관계수는 빈칸입니다.')],className='grid-equal')])


def validation_panels(data):
    from backend.quality_engine import confusion
    metric = confusion([p['label'] for p in data['points']],[p['prediction'] for p in data['points']])
    scope = '회고 시험 파일' if data['run'] in ('WeldingTest_02_OK','WeldingTest_04_NG') else '개발에 사용한 파일 · 독립 시험 아님'
    return html.Div([panel('선택 모델 조합 · '+data['run'],html.Div([
        plot(confusion_figure(metric)),metric_rows([('정밀도',num(metric['Precision'],4),'경보 중 실제 이상'),
            ('Recall',num(metric['Recall'],4),'실제 이상 없으면 산출 불가'),('오경보',str(metric['FP'])+'행','실제 정상 중 경보'),
            ('미탐',str(metric['FN'])+'행','실제 이상 중 미탐')])],className='grid-equal'),scope+' · 선택 파일 전체 행 · '+data['supervised']+' OR '+data['unsupervised']),
        callout('정상 기준과 경보 기준의 의미','기본 RobustPhaseZ는 용접 지점별 정상 중앙값으로부터의 편차를 비교합니다. 임계값은 정상 보정 자료로 정한 통계적 기준이며 설비의 합격 허용범위가 아닙니다. 변수 중요도는 아래 표에 명시된 모델에만 해당합니다.')])


def sensitivity_panel(validation):
    rows = [r for r in validation['sensitivity'] if r['experiment']=='normal_quantile' and r['model']=='RobustPhaseZ']
    rows.sort(key=lambda r:float(r['setting']))
    fig = go.Figure()
    for key,label,color in [('precision','정밀도',BLUE),('recall','Recall',CYAN),('f1','F1',GRAY)]:
        fig.add_trace(go.Scatter(x=[f"{float(r['setting'])*100:g}%" for r in rows],y=[r[key] for r in rows],name=label,
            mode='lines+markers',line={'color':color},customdata=[[r['fp'],r['fn'],r['threshold']] for r in rows],
            hovertemplate='%{y:.4f}<br>오경보 %{customdata[0]}행 · 미탐 %{customdata[1]}행<br>임계값 %{customdata[2]:.3f}<extra>'+label+'</extra>'))
    layout(fig,240,'평가 지표')
    fig.update_yaxes(range=[0,1.05])
    return html.Details([html.Summary('고정된 임계값 민감도 결과 보기'),panel('RobustPhaseZ · 기존 보조 검증',plot(fig),
        '02_OK + 04_NG 회고 시험 · 정상 보정 분위수 비교 · 주 모델/임계값 선택에 미사용 · 화면에서 운영 기준을 변경하지 않습니다.')],className='maintenance-explore')


def modals():
    return html.Div([dcc.Download(id='m-download'),
        dmc.Modal(id='m-report-modal',title='예지보전 결과 보고서',size='lg',centered=True,closeButtonProps={'aria-label':'닫기'},children=html.Div(id='m-report-content')),
        dmc.Modal(id='m-upload-modal',title='용접 공정 CSV 검사·분석',size='90%',centered=True,closeButtonProps={'aria-label':'닫기'},children=[
            dcc.Store(id='m-upload-token'),dcc.Store(id='m-upload-result'),
            html.Div([html.Span(s) for s in ['① 파일 선택','② 입력 검사','③ 처리 확인','④ 저장 모델 분석']],className='quality-upload-steps'),
            callout('완료된 용접 사이클 분석','필수 9열 · UTF-8/CP949 · 5MB/19,500행 이하. 기본 LogisticCurrent OR RobustPhaseZ 고정 모델을 사용합니다. 업로드는 본인만 조회하며 15분 후 만료됩니다.'),
            html.A('정상 예제 CSV 다운로드',href='/api/maintenance/sample',className='template-link'),
            html.Div([dcc.Upload(id='m-upload-file',children=html.Div([html.Strong('공정 CSV 선택 또는 끌어놓기'),html.Small(id='m-upload-filename')]),accept='.csv',max_size=5_000_000,className='upload-zone'),
                dcc.Upload(id='m-upload-label',children=html.Div([html.Strong('정답 CSV 선택 · 선택 사항'),html.Small(id='m-upload-labelname')]),accept='.csv',max_size=5_000_000,className='upload-zone')],className='grid-equal'),
            html.P('정답은 검사 후 제공되는 템플릿의 sourceRow·sourceSha256·label(0/1)로 연결합니다. 정답 없이도 분석할 수 있습니다.',className='section-note'),
            dmc.Group([dmc.Button('입력 검사',id='m-upload-inspect'),dmc.Button('검사한 자료 분석',id='m-upload-analyze',disabled=True)]),
            dcc.Loading(html.Div(id='m-upload-checks')),dcc.Loading(html.Div(id='m-upload-output'))])])


def register(app, request):
    @app.callback(Output('m-cycle','value'),Input('m-cycle-grid','selectedRows',allow_optional=True),Input('m-risk-map','clickData',allow_optional=True),prevent_initial_call=True)
    def choose_cycle(rows,click):
        if ctx.triggered_id=='m-risk-map' and click:
            return str(int(click['points'][0]['y']))
        return str(rows[0]['cycle']) if rows else no_update

    @app.callback(Output('m-cycle-chart','children'),Output('m-cycle-detail','children'),Input('m-cycle','value',allow_optional=True),Input('view-data','data'))
    def detail(selected,view):
        if not view or view['domain']!='maintenance' or view['tab']!='analysis' or not view['payload'].get('points'):
            return no_update,no_update
        return cycle_detail(view['payload'],selected)

    @app.callback(Output('m-point-select','value'),Input('maintenance-3d','clickData',allow_optional=True),State('view-data','data'),prevent_initial_call=True)
    def select_point(click,view):
        if not view or view['domain']!='maintenance':return no_update
        return selection(click,{str(p['row']) for p in view['payload']['points']}) or no_update

    @app.callback(Output('m-point-detail','children'),Input('m-point-select','value',allow_optional=True),State('view-data','data'))
    def point(selected,view):
        if not view or selected is None:return []
        p = next((p for p in view['payload'].get('points',[]) if str(p['row'])==selected),None)
        if not p:return []
        return html.Div([badge('출력 0' if p['power']==0 else '경보' if p['prediction'] else '경보 없음','danger' if p['prediction'] else 'info'),
            metric_rows([('실측 출력',f"{p['power']:,.1f} W",p['time']),('정상 기준',f"{p['expected']:,.1f} W",f"편차 {p['deviation']:+,.1f} W"),
                ('설정 출력',f"{p['setPower']:g}%",f"속도 {p['speed']:g} · 길이 {p['length']:g}"),('모델 판정',p['agreement'],f"지도 {p['supRatio']:.3f}배 / 비지도 {p['unsupRatio']:.3f}배")])],className='point-inspector')

    @app.callback(Output('maintenance-3d','figure'),Input('m-point-select','value',allow_optional=True),State('view-data','data'),State('m-cycle','value',allow_optional=True),State('maintenance-3d','figure',allow_optional=True),prevent_initial_call=True)
    def highlight_point(selected,view,cycle,current):
        if selected is None or not current or not view or view['domain']!='maintenance':return no_update
        points=[p for p in view['payload']['points'] if str(p['cycle'])==cycle]
        if not points or selected not in {str(p['row']) for p in points}:return no_update
        fig=maintenance_figure(view['payload'],points,int(selected))
        fig.update_layout(scene_uirevision=current['layout']['scene'].get('uirevision'))
        return fig

    @app.callback(Output('m-download','data'),Input('m-cycle-export','n_clicks',allow_optional=True),State('m-cycle-grid','virtualRowData',allow_optional=True),State('view-data','data'),prevent_initial_call=True)
    def export(_,rows,view):
        if not _:
            return no_update
        from backend.maintenance_routes import csv_bytes
        rows = rows if rows is not None else view['payload']['cycleDetails']
        return dcc.send_bytes(csv_bytes(rows),'maintenance_cycles.csv')

    @app.callback(Output('m-report-modal','opened'),Output('m-report-content','children'),Input('open-maintenance-report','n_clicks'),State('m-run','value'),State('m-sup','value'),State('m-unsup','value'),prevent_initial_call=True)
    def report(_,run,sup,unsup):
        query = urlencode({'run':run,'supervised':sup,'unsupervised':unsup})
        return True,[callout('선택 파일 전체를 보고서로 정리',f'{run} · {sup} OR {unsup}. 표의 검색·필터와 별개로 파일 전체를 포함합니다. 업무 기록은 기존 이벤트 검토에서 조회하세요.'),
            html.Div([html.A(label,href=f'/api/maintenance/report?{query}&format={kind}',target='_blank' if kind=='print' else None,className='template-link') for label,kind in [('분석 결과 묶음 · CSV + 보고서','zip'),('Markdown 보고서','markdown'),('인쇄 / PDF로 저장','print')]],className='maintenance-report-links'),
            html.P('측정·사이클·이벤트·조건별 요약·평가 지표를 내보냅니다. 보고서에 모델·데이터·정책 버전과 한계가 포함됩니다.',className='section-note')]

    @app.callback(Output('m-upload-modal','opened'),Input('open-maintenance-upload','n_clicks'),prevent_initial_call=True)
    def open_upload(_):
        return True

    @app.callback(Output('m-upload-checks','children'),Output('m-upload-token','data'),Output('m-upload-analyze','disabled'),Output('m-upload-filename','children'),Output('m-upload-labelname','children'),
        Input('m-upload-inspect','n_clicks'),Input('m-upload-file','contents'),Input('m-upload-label','contents'),State('m-upload-file','filename'),State('m-upload-label','filename'),prevent_initial_call=True,
        running=[(Output('m-upload-inspect','loading'),True,False)])
    def inspect_upload(_,contents,labels,name,labelname):
        if ctx.triggered_id!='m-upload-inspect':
            return html.P('파일이 변경되었습니다. 입력 검사를 실행하세요.'),None,True,name or '',labelname or ''
        try:
            if not contents:
                raise ValueError('공정 CSV를 선택하세요.')
            files = {'file':(name or 'input.csv',base64.b64decode(contents.split(',',1)[1],validate=True),'text/csv')}
            if labels:
                files['labels'] = (labelname or 'labels.csv',base64.b64decode(labels.split(',',1)[1],validate=True),'text/csv')
            result = request('/api/maintenance/inspect','POST',files=files)
            return [callout('입력 검사 완료','주의 항목과 처리 내역을 확인한 뒤 분석하세요. 업로드 자료는 공식 시드·Firestore에 등록되지 않습니다.'),
                html.Div([panel('검사 결과',grid(result['checks'],'m-upload-check-grid',height=330)),panel('처리 전·후',grid(result['beforeAfter'],'m-upload-before-grid',height=260),action=html.A('정답 템플릿',href=f"/api/maintenance/uploads/{result['token']}/labels",className='template-link'))],className='grid-equal')],result['token'],False,name or '',labelname or ''
        except (ValueError,httpx.HTTPError) as exc:
            return callout('입력 검사 실패',str(exc) if isinstance(exc,ValueError) else '검사 서비스 연결을 확인하세요.','warning'),None,True,name or '',labelname or ''

    @app.callback(Output('m-upload-output','children'),Output('m-upload-result','data'),Input('m-upload-analyze','n_clicks'),Input('m-upload-token','data'),prevent_initial_call=True,
        running=[(Output('m-upload-analyze','loading'),True,False)])
    def analyze_upload(_,token):
        if ctx.triggered_id!='m-upload-analyze' or not token:
            return [],None
        try:
            data = request(f'/api/maintenance/uploads/{token}/analyze','POST')
            children = [callout('분석 완료 · 임시 결과',f"{len(data['points']):,}행 · 경보 {data['anomalyRows']}행 · 점검 {data['alertCycles']}사이클 · 학습 밖 {data['unseenRows']}행. 기록 저장과 공식 데이터 등록은 별도입니다."),
                html.A('분석 결과·보고서 다운로드',href=f'/api/maintenance/uploads/{token}/export',className='template-link'),
                panel('업로드 출력·경보',plot(power_chart(data,data['points'])),data['referenceNotice']),
                panel('업로드 사이클 점검 목록',grid(data['cycleDetails'],'m-upload-cycles',cycle_columns(),300)),
                dmc.Select(id='m-upload-cycle',label='39지점 상세 조회',data=[{'label':f"사이클 {r['cycle']} · {r['상태']}",'value':str(r['cycle'])} for r in data['cycleDetails']],value=str(data['cycleDetails'][0]['cycle']),allowDeselect=False),
                html.Div(id='m-upload-cycle-detail')]
            if all(p['label'] is not None for p in data['points']):
                children.append(validation_panels(data))
            else:
                children.append(html.P('정답 없음 · 탐지 결과만 표시하며 성능 수치는 계산하지 않습니다.',className='section-note'))
            return children,data
        except (ValueError,httpx.HTTPError) as exc:
            return callout('분석 실패',str(exc) if isinstance(exc,ValueError) else '분석 서비스 연결을 확인하세요.','warning'),None

    @app.callback(Output('m-upload-cycle-detail','children'),Input('m-upload-cycle','value',allow_optional=True),Input('m-upload-result','data'))
    def uploaded_detail(selected,data):
        if not selected or not data:
            return no_update
        points = [p for p in data['points'] if str(p['cycle'])==selected]
        return grid([{'지점':p['page'],'시각':p['time'],'실측 (W)':p['power'],'기준 (W)':p['expected'],'편차 (W)':round(p['deviation'],2),'판정':p['agreement'],'학습 밖':p['unseen']} for p in points],'m-upload-point-grid',height=300)
