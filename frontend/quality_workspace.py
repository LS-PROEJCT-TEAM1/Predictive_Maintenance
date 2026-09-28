"""Quality detection, exploration, evaluation and upload surfaces."""
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, no_update, ctx
import dash_mantine_components as dmc

from frontend.components import BLUE, CYAN, GRAY, RED, badge, callout, graph, grid, kpi, metric_rows, num, panel
from frontend.charts import layout


def score_figure(points, thresholds):
    fig = go.Figure()
    for key, label, color in [('spe', 'SPE / 관리한계', BLUE), ('t2', 'T² / 관리한계', CYAN)]:
        fig.add_trace(go.Scatter(x=[p['index'] for p in points], y=[p[key]/thresholds[key] for p in points], name=label, mode='lines', line={'color': color, 'width': 1.5}))
    fig.add_hline(y=1, line_color=RED, line_dash='dot', annotation_text='관리한계 1배')
    layout(fig, 295, '관리한계 대비 배수')
    if max((p[k]/thresholds[k] for p in points for k in ('t2', 'spe')), default=0) > 100:
        fig.update_yaxes(type='log', title='관리한계 대비 배수 (로그축 · 0 제외)')
    fig.update_xaxes(title='원본 행 ID (0부터)')
    return fig


def confusion_figure(metrics):
    z = [[metrics['TN'], metrics['FP']], [metrics['FN'], metrics['TP']]]
    text = [[f"정상 TN · {metrics['TN']:,}", f"오경보 FP · {metrics['FP']:,}"],
            [f"미탐 FN · {metrics['FN']:,}", f"정탐 TP · {metrics['TP']:,}"]]
    fig = go.Figure(go.Heatmap(z=z, x=['예측 정상', '예측 이상'], y=['실제 정상', '실제 이상'],
                    text=text, texttemplate='%{text}', colorscale=[[0, '#F0F3FA'], [1, BLUE]], showscale=False, xgap=6, ygap=6,
                    hovertemplate='%{text}<extra></extra>'))
    layout(fig, 245)
    fig.update_yaxes(autorange='reversed')
    return fig


def evaluation_panels(data, scope):
    metrics = data['all' if scope == 'all' else 'selected']
    label = '잠금 시험 5개 전체' if scope == 'all' else data['test']
    fig = go.Figure()
    ratio = np.array(data['variance'])*100
    pc = [f'PC{i+1}' for i in range(len(ratio))]
    fig.add_trace(go.Bar(x=pc, y=ratio, name='개별 설명분산', marker_color=GRAY))
    fig.add_trace(go.Scatter(x=pc, y=ratio.cumsum(), name='누적 설명분산', mode='lines+markers', line={'color': BLUE}))
    layout(fig, 245, '설명분산 (%)')
    return html.Div([
        html.Div([panel('혼동행렬 · '+label, graph(confusion_figure(metrics)), '정답 라벨과 비교 · 시점 단위 · 정답은 모델 입력에 사용하지 않음'),
                  panel('미탐·오경보를 함께 확인', metric_rows([
                      ('불량 Recall', num(metrics['Recall'], 3), '실제 이상 중 탐지 비율 · 실제 이상 없음은 —'),
                      ('정밀도', num(metrics['Precision'], 3), '탐지한 시점 중 실제 이상 비율'),
                      ('오경보율', num(metrics['FPR']*100 if metrics['FPR'] is not None else None, 2)+'%', '실제 정상 중 오경보 비율'),
                      ('F1', num(metrics['F1'], 4), '시점 평가 · 불량 원인 분류 성능 아님')]))], className='grid-equal'),
        html.Div([panel('PCA 주성분 설명분산', graph(fig), f"정상 기준 {data['normalRows']:,}행 · 원본 208채널을 요약한 9개 입력 지표"),
                  panel('파일 그룹 교차검증 · 기존 평가', grid([{**r, 'F1-score': r['F1-score'] if r['실제 이상'] else None} for r in data['folds']], 'quality-fold-grid', height=290), '정상 전용 fold의 F1은 — · 교차검증과 잠금 시험은 별도 평가입니다.')], className='grid-equal'),
        panel('시험별 오경보·미탐', grid(data['rows'], 'quality-file-grid', height=285), '행을 선택하면 해당 시험으로 이동합니다. 표의 열 필터로 공정·AI 상태를 좁힐 수 있습니다.')])


def detection(data, scope):
    fig = go.Figure()
    for flag, name, color in [(0, '모델 정상', GRAY), (1, '모델 이상', RED)]:
        points = [p for p in data['points'] if p['prediction'] == flag and p['t2'] > 0 and p['spe'] > 0]
        fig.add_trace(go.Scatter(x=[p['t2']/data['thresholds']['t2'] for p in points],
            y=[p['spe']/data['thresholds']['spe'] for p in points], customdata=[p['index'] for p in points],
            mode='markers', name=name, marker={'color': color, 'size': 5, 'opacity': .55},
            hovertemplate='행 %{customdata}<br>T² %{x:.3f}배 · SPE %{y:.3f}배<extra></extra>'))
    fig.add_vline(x=1, line_dash='dot', line_color=BLUE)
    fig.add_hline(y=1, line_dash='dot', line_color=BLUE)
    layout(fig, 295, 'SPE / 관리한계 (로그축)')
    fig.update_xaxes(type='log', title='T² / 관리한계 (로그축)')
    fig.update_yaxes(type='log')
    return html.Div([callout('고정 관리한계로 시험 전체를 확인', f"T² {data['thresholds']['t2']:.4f} / SPE {data['thresholds']['spe']:.4f} · 연속 {data['thresholds']['consecutivePoints']}시점. 그래프는 최대 502시점 표본이며 평가는 전체 행으로 계산합니다."),
        html.Div([panel('이상 점수와 관리한계', graph(score_figure(data['points'], data['thresholds']))),
                  panel('T²–SPE 분포', graph(fig), '1배 초과는 관리한계 이탈 · 색은 연속 조건을 적용한 저장 판정 · 로그축에서 0은 제외')], className='grid-main'),
        html.P('관리한계 이탈은 불량 원인 확정이 아닙니다. 셀 검사에서 원본값과 추세를 확인한 뒤 판정을 기록하세요.',className='section-note')])


def exploration(data):
    fig = go.Figure()
    for row in data['curves']:
        normal = row['grade'] == 'normal-reference'
        fig.add_trace(go.Scatter(x=[p['progress'] for p in row['points']], y=[p['dv'] for p in row['points']],
            name=row['file'], mode='lines', line={'color': GRAY if normal else BLUE, 'width': 1 if normal else 2.5}, opacity=.5 if normal else 1))
    layout(fig, 330, '셀 전압 차이 ΔV (mV)')
    fig.update_xaxes(title='각 시험의 상대 진행률 (%)')
    fig.update_layout(legend={'font': {'size': 10}, 'y': 1.2})
    box = go.Figure()
    for mode, mode_label in [('chg', '충전'), ('dchg', '방전')]:
        for row in [r for r in data['boxes'] if r['mode'] == mode]:
            q = row['q']
            box.add_trace(go.Box(name=row['file'], x=[mode_label+' · '+row['file']], q1=[q[1]], median=[q[2]], q3=[q[3]], lowerfence=[q[0]], upperfence=[q[4]],
                marker_color=BLUE if row['file'] == data['test'] else CYAN if row['grade'] == 'NG' else GRAY, showlegend=False))
    layout(box, 330, '모듈 온도 편차 (°C)')
    box.update_layout(margin={'l': 50, 'r': 15, 't': 10, 'b': 120})
    box.update_xaxes(tickangle=-45, tickfont={'size': 9})
    return html.Div([callout('같은 공정의 정상 기준과 선택 시험 비교', '서로 다른 팩을 각 시험의 상대 진행률로 정렬한 참고 비교입니다. 같은 팩의 충·방전 쌍이나 합격 판정 기준을 뜻하지 않습니다.'),
        html.Div([panel(('충전' if data['mode'] == 'chg' else '방전')+' · 전압 편차 추세', graph(fig), '정상 기준 5파일 + 선택 시험 · 보정값 · 파일별 최대 201시점'),
                  panel('충전·방전별 온도 편차 분포', graph(box), '파일별 중앙값·사분위수 · 수염은 최솟값/최댓값 · NG는 파일 단위 자료 구분')], className='grid-equal'),
        panel('원본 처리 근거 · 공식 자료', grid(data['quality'], 'quality-data-grid', height=350), '모델 기준 파일별 행 수·범위 이탈·이상 시점 · 기록에 없는 검사는 정상값으로 채우지 않습니다.')])


def upload_modal():
    return dmc.Modal(id='quality-upload-modal', title='배터리 시험 CSV 검사·분석', size='90%', centered=True, closeButtonProps={'aria-label': '닫기'}, children=[
        dcc.Store(id='quality-upload-job'), dcc.Store(id='quality-upload-result'),
        html.Div([html.Span(s) for s in ['① 파일 선택', '② 입력 검사', '③ 보정 확인', '④ 고정 모델 분석']], className='quality-upload-steps'),
        callout('완료된 시험 파일만 업로드', 'UTF-8 · 30MB 이하 · 10~20,000행 · Date/Time/SerialNumber + 전압 176·온도 32채널. 업로드 자료는 임시 메모리에만 보관하며 15분 후 만료됩니다.'),
        html.A('예제 시험 CSV 다운로드 · Test03', href='/api/quality/sample', className='template-link'),
        dmc.Select(id='quality-upload-mode', label='공정 직접 선택', data=[{'label': '충전', 'value': 'chg'}, {'label': '방전', 'value': 'dchg'}], value='chg', allowDeselect=False),
        html.Div([
            dcc.Upload(id='quality-upload-file', children=html.Div([html.Strong('시험 CSV 선택'), html.Small(id='quality-upload-filename')]), accept='.csv', max_size=30_000_000, className='upload-zone'),
            dcc.Upload(id='quality-upload-label', children=html.Div([html.Strong('정답 CSV 선택 · 선택 사항'), html.Small(id='quality-upload-labelname')]), accept='.csv', max_size=30_000_000, className='upload-zone')], className='grid-equal'),
        html.P('선택 정답: sourceRow·sourceSha256·label(0/1)로 원본과 연결합니다. 먼저 검사한 뒤 정답 템플릿을 내려받을 수 있습니다.', className='section-note'),
        dmc.Group([dmc.Button('입력 검사', id='quality-upload-inspect'), dmc.Button('검사한 자료 분석', id='quality-upload-analyze', disabled=True),
                   dmc.Button('분석 결과 CSV', id='quality-upload-export', disabled=True, variant='outline')]),
        dcc.Download(id='quality-upload-download'),
        dcc.Loading(html.Div(id='quality-upload-checks'), type='circle'), dcc.Loading(html.Div(id='quality-upload-output'), type='circle')])


def history_panel(data):
    history = data.get('history', {})
    if history.get('unavailable'):
        return panel('점검 이력', callout('기록 조회 불가', 'Firestore 연결·할당량을 확인하세요. 기록이 없다는 뜻은 아닙니다.', 'warning'))
    rows = history.get('rows', [])
    columns = [{'field': key, 'minWidth': size, 'flex': 2 if key == '메모' else 1} for key, size in
               [('시각', 210), ('시험', 180), ('공정', 90), ('AI 상태', 125), ('작업자 판정', 135), ('작성자', 130), ('메모', 230), ('이전 판정', 130), ('수정 차수', 100), ('데이터 버전', 150)]]
    return panel('점검 이력 · 실제 저장 기록', [
        html.P('조회 범위에 저장된 기록이 없습니다.', className='section-note') if not rows else None,
        grid(rows, 'quality-history-grid', columns, height=365)],
        history.get('scope', '')+' · 열 필터: 시험·공정·AI 상태·판정·작성자 · 시각은 저장된 시간대 포함',
        dmc.Button('필터 결과 CSV', id='quality-history-export', disabled=not bool(rows), variant='outline'))


def register(app, request):
    import base64
    import httpx

    @app.callback(Output('quality-upload-modal', 'opened'), Input('open-quality-upload', 'n_clicks'), prevent_initial_call=True)
    def opened(_):
        return True

    @app.callback(Output('quality-upload-checks', 'children'), Output('quality-upload-job', 'data'), Output('quality-upload-analyze', 'disabled'),
                  Output('quality-upload-filename', 'children'), Output('quality-upload-labelname', 'children'),
                  Input('quality-upload-inspect', 'n_clicks'), Input('quality-upload-file', 'contents'), Input('quality-upload-label', 'contents'), Input('quality-upload-mode', 'value'),
                  State('quality-upload-file', 'filename'), State('quality-upload-label', 'filename'), prevent_initial_call=True,
                  running=[(Output('quality-upload-inspect', 'disabled'), True, False)])
    def inspect_upload(_, contents, labels, mode, name, labelname):
        if ctx.triggered_id != 'quality-upload-inspect':
            return html.P('파일 또는 공정이 변경되었습니다. 입력 검사를 실행하세요.'), None, True, name or '', labelname or ''
        try:
            if not contents:
                raise ValueError('시험 CSV를 선택하세요.')
            files = {'file': (name or 'test.csv', base64.b64decode(contents.split(',', 1)[1], validate=True), 'text/csv')}
            if labels:
                files['labels'] = (labelname or 'labels.csv', base64.b64decode(labels.split(',', 1)[1], validate=True), 'text/csv')
            result = request('/api/quality/inspect', 'POST', files=files, data={'mode': mode})
            content = [callout('검사 완료 · 보정 내역을 확인한 뒤 분석하세요', result['notice']),
                html.Div([panel('입력 검사 결과', grid(result['checks'], 'quality-upload-check-grid', height=350)),
                          panel('처리 전·후', grid(result['beforeAfter'], 'quality-upload-before-grid', height=260), action=html.A('정답 템플릿 다운로드', href=f"/api/quality/uploads/{result['token']}/labels", className='template-link'))], className='grid-equal'),
                panel('원본·보정값 예시', grid(result['changes'], 'quality-upload-change-grid', height=260), '대표 시점 및 범위 이탈 일부 채널 · 원본값은 수정하지 않음'),
                html.P('PCA 입력 9개: '+' · '.join(result['features']), className='section-note')]
            return content, result['token'], False, name or '', labelname or ''
        except (ValueError, httpx.HTTPError) as exc:
            return callout('입력 검사 실패', str(exc) if isinstance(exc, ValueError) else '검사 서비스 연결을 확인하세요.', 'warning'), None, True, name or '', labelname or ''

    @app.callback(Output('quality-upload-output', 'children'), Output('quality-upload-result', 'data'), Output('quality-upload-export', 'disabled'),
                  Input('quality-upload-analyze', 'n_clicks'), Input('quality-upload-job', 'data'), prevent_initial_call=True,
                  running=[(Output('quality-upload-analyze', 'loading'), True, False)])
    def infer(_, token):
        if ctx.triggered_id != 'quality-upload-analyze' or not token:
            return [], None, True
        try:
            result = request(f'/api/quality/uploads/{token}/analyze', 'POST')
            children = [callout('분석 완료', result['notice']), html.Div([
                kpi('분석 시점', f"{result['rows']:,}", result['modelVersion']), kpi('이상 시점', f"{result['abnormalRows']:,}", '고정된 연속 10시점 규칙'),
                kpi('정답 비교', '제공됨' if result['confusion'] else '미제공', '정답 미제공 시 성능을 계산하지 않음'), kpi('저장 상태', '임시 분석', '공식 시드·Firestore 미저장')], className='kpi-strip'),
                panel('업로드 시험 이상 점수', graph(score_figure(result['points'], result['thresholds'])), '차트 최대 500시점 표본 · CSV는 전체 행')]
            if result['confusion']:
                children.append(panel('업로드 정답과 비교', graph(confusion_figure(result['confusion']))))
            return children, result['results'], False
        except (ValueError, httpx.HTTPError) as exc:
            return callout('분석 실패', str(exc) if isinstance(exc, ValueError) else '연결을 확인하세요.', 'warning'), None, True

    @app.callback(Output('quality-upload-download', 'data'), Input('quality-upload-export', 'n_clicks'), State('quality-upload-result', 'data'), prevent_initial_call=True)
    def export(_, rows):
        import pandas as pd
        return dcc.send_data_frame(pd.DataFrame(rows or []).to_csv, 'quality_analysis.csv', index=False, encoding='utf-8-sig')

    @app.callback(Output('quality-history-download', 'data'), Input('quality-history-export', 'n_clicks', allow_optional=True),
                  State('quality-history-grid', 'virtualRowData', allow_optional=True), prevent_initial_call=True)
    def export_history(clicks, rows):
        import pandas as pd
        if not clicks:
            return no_update
        frame = pd.DataFrame(rows or [])
        # Spreadsheet text from staff notes must not become executable formulas.
        for c in frame:
            frame[c] = frame[c].map(lambda x: "'"+x if isinstance(x, str) and x.lstrip().startswith(('=', '+', '-', '@')) else x)
        return dcc.send_data_frame(frame.to_csv, 'quality_history_filtered.csv', index=False, encoding='utf-8-sig')
