import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import httpx
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
import dash_mantine_components as dmc
from frontend.components import badge, callout, icon


def drawer():
    return dmc.Drawer(id='copilot-drawer', title='AI Copilot', position='right', size=560,
        classNames={'body':'copilot-body','content':'copilot-content'},
        closeButtonProps={'aria-label': '닫기'}, children=[
        dcc.Store(id='chat-state', data={}),
        dcc.Store(id='chat-scroll-state'),
        dcc.Store(id='chat-history'), dcc.Store(id='chat-threads', data=[]), dcc.Store(id='chat-busy', data=False),
        html.Div([html.Span('나만의 대화',className='chat-private'),html.Div([
            dmc.Button('대화 목록',id='chat-list-toggle',variant='subtle',leftSection=icon('history',16)),
            dmc.Button('새 대화',id='chat-new',variant='default',leftSection=icon('plus',16))],className='chat-toolbar-actions')],className='chat-toolbar'),
        html.Div(id='chat-status', role='status'),
        html.Div([
            html.Div([html.H3('대화 목록'),html.Div(id='chat-history-list')],className='chat-history-pane'),
            html.Div([
                html.Details([html.Summary([html.Span('현재 화면',className='chat-context-label'),html.Span(id='copilot-context')]),
                    html.P('질문을 보낼 때 선택한 화면·부품·시험 조건을 함께 참조합니다. 다른 대상은 질문에 번호를 적어 주세요.')],className='chat-context-details'),
                html.Div([html.Div(id='chat-messages',children=messages_view([]),className='chat-messages'),
                    html.Div([dmc.Button(id='chat-suggestion-'+str(i),variant='default',justify='flex-start',rightSection=icon('chevron-right',15)) for i in range(3)],className='chat-suggestions'),
                    dmc.Button('답변으로 검토 초안 만들기',id='chat-draft',variant='subtle',leftSection=icon('check-check',16),className='chat-draft-action',style={'display':'none'})],className='chat-feed',id='chat-feed'),
                html.Div('근거를 검색하고 답변을 준비하고 있습니다…',id='chat-progress',className='chat-progress',role='status',style={'display':'none'}),
                html.Div([dmc.Textarea(id='chat-question',placeholder='부품·시험 번호 또는 궁금한 내용을 입력하세요',minRows=2,maxRows=5,autosize=True,inputProps={'maxLength':2000,'aria-label':'Copilot에게 질문'}),
                    dmc.Button(icon('arrow-up',18),id='chat-send',className='chat-send',buttonProps={'title':'질문 보내기'},**{'aria-label':'질문 보내기'})],className='chat-composer'),
                html.Details([html.Summary('본인만 조회 · 답변과 저장 안내'),html.P('질문과 검색 근거는 Gemini API로 전송됩니다. 대화는 본인만 조회합니다. 현장 SOP는 연결되어 있지 않으며 최종 판정과 업무 기록 저장은 직원이 직접 확인합니다.')],className='chat-notice')
            ],className='chat-conversation')],id='copilot-workspace',className='copilot-workspace')])


def snapshot_view(snapshot):
    labels={'dataVersion':'자료 버전','dataSource':'자료 기준','date':'예측 목표일','part':'부품','model':'예측 모델',
            'forecast':'예측량','plan':'기존 계획','gap':'계획과의 차이','run':'용접 시험','testId':'품질 시험',
            'selectedCell':'선택 셀','zeroPowerRows':'출력 0W 행 수','anomalyRows':'이상 행 수',
            'abnormalPointCount':'이상 시점 수','basis':'측정 기준','recordScope':'기록 조회 범위'}
    rows=[html.Div([html.Dt(label),html.Dd(str(snapshot[key]))]) for key,label in labels.items() if key in snapshot]
    return html.Div([html.Dl(rows,className='chat-facts'),html.P(snapshot.get('interpretation') or snapshot.get('targetDefinition',''),className='section-note'),
        html.Details([html.Summary('전체 근거 데이터'),html.Pre(json.dumps(snapshot,ensure_ascii=False,indent=2),className='source-snapshot')])])


def messages_view(messages):
    items = []
    for message in messages:
        sources = []
        for source in message.get('citations', []):
            if source['id'] == 'SCREEN':
                detail = snapshot_view(source.get('snapshot', {}))
            else:
                detail = html.Div([html.P(source.get('excerpt', ''), className='source-excerpt'),
                    html.A('문서 원문 열기', href=f"/api/copilot/sources/{source['source']}", target='_blank', rel='noopener')])
            sources.append(html.Details([html.Summary(f"[{source['id']}] {source['title']}"), detail], className='chat-source'))
        items.append(html.Article([html.Div('나' if message['role'] == 'user' else 'AI Copilot', className='chat-speaker'),
            dcc.Markdown(message.get('text', ''), className='chat-text', dangerously_allow_html=False),
            html.Div([dcc.Link(link['label']+' →',href=link['url'],className='template-link') for link in message.get('links',[]) if link.get('url','').startswith(('/demand?','/maintenance?','/quality?'))],className='chat-result-links'),*sources,
            html.Small(local_time(message.get('at', '')))], className='chat-message '+message['role']))
        if sources:
            items[-1].children[3:3+len(sources)] = [html.Details([html.Summary(f'답변 근거 {len(sources)}개'),*sources],className='chat-sources')]
    return items or html.Div([icon('message-square',24),html.H3('어떤 내용을 확인할까요?'),html.P('현재 분석의 근거를 확인하거나, 찾고 싶은 부품·시험을 질문하세요.')],className='chat-empty')


def local_time(value):
    try:
        stamp = datetime.fromisoformat(value.replace('Z','+00:00'))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp.astimezone(timezone(timedelta(hours=9))).strftime('%m월 %d일 %H:%M')
    except (ValueError, AttributeError):
        return ''


def register(app, request):
    app.clientside_callback("""function(children) {
        window.setTimeout(function() {
            const feed=document.getElementById('chat-feed');
            if(feed) feed.scrollTop=feed.scrollHeight;
        }, 80);
        return Date.now();
    }""",Output('chat-scroll-state','data'),Input('chat-messages','children'))

    @app.callback(Output('chat-history-list','children'),Input('chat-threads','data'),Input('chat-state','data'))
    def history_list(threads,current):
        return [html.Button([icon('message-square',16),html.Span(t['title'])],id={'type':'chat-thread','id':t['id']},n_clicks=0,
            title=t['title'],className='chat-history-row'+(' selected' if t['id']==(current or {}).get('id') else ''),
            **{'aria-pressed':'true' if t['id']==(current or {}).get('id') else 'false'}) for t in threads or []] or html.P('저장된 대화가 없습니다. 새 대화를 시작해 보세요.',className='chat-history-empty')

    @app.callback(Output('chat-history','data'),Input({'type':'chat-thread','id':ALL},'n_clicks'),State('chat-busy','data'),prevent_initial_call=True)
    def select_history(clicks,busy):
        if busy or not any(clicks) or not ctx.triggered or not ctx.triggered[0].get('value'):
            return no_update
        return {'id':ctx.triggered_id['id'], 'selection':uuid4().hex}

    @app.callback(Output('copilot-workspace','className'),Output('chat-list-toggle','children'),
        Input('chat-list-toggle','n_clicks'),Input('chat-new','n_clicks'),Input('chat-history','data'),State('copilot-workspace','className'),prevent_initial_call=True)
    def toggle_history(_,new,selected,current):
        opened=ctx.triggered_id=='chat-list-toggle' and 'show-history' not in (current or '')
        return ('copilot-workspace show-history' if opened else 'copilot-workspace'),('대화로 돌아가기' if opened else '대화 목록')

    @app.callback(Output('chat-draft','style'),Input('chat-state','data'))
    def draft_visibility(state):
        return {} if any(m.get('role')=='assistant' for m in (state or {}).get('messages',[])) else {'display':'none'}

    @app.callback(*[Output('chat-suggestion-'+str(i),'children') for i in range(3)],Input('url','pathname'))
    def suggestions(path):
        return {'/demand':['계획과 차이가 큰 부품 3개를 찾아줘','미확인 검토 부품을 찾아줘','왜 기본 모델이 7일 이동평균이야?'],
                '/maintenance':['현재 시험에서 점검할 이벤트를 찾아줘','출력 0W 구간을 찾아줘','현재 예지보전 모델의 검증 한계는?'],
                '/quality':['현재 시험의 불량 유형 근거를 설명해줘','선택 셀에서 확인할 점을 알려줘','작업자 판정 기록을 찾아줘']}.get(path,['각 업무에서 검토할 대상을 찾아줘','계획과 차이가 큰 부품 3개를 찾아줘','현재 예지보전 모델의 검증 한계는?'])

    @app.callback(Output('chat-question','value',allow_duplicate=True),
        *[Input('chat-suggestion-'+str(i),'n_clicks') for i in range(3)],
        *[State('chat-suggestion-'+str(i),'children') for i in range(3)],prevent_initial_call=True)
    def choose_suggestion(*values):
        return values[3+int(ctx.triggered_id.rsplit('-',1)[-1])] if any(values[:3]) else no_update

    @app.callback(Output('copilot-context', 'children'), Input('url', 'pathname'), Input('d-part', 'value'), Input('d-date', 'value'), Input('d-model', 'value'),
        Input('m-run', 'value'), Input('m-sup', 'value'), Input('m-unsup', 'value'), Input('q-test', 'value'), Input('q-cell', 'value'))
    def context_label(path, part, date, model, run, sup, unsup, test, cell):
        key = (path or '/').strip('/')
        return {'demand': f'공급망 · {date} · {part} · {model}', 'maintenance': f'예지보전 · {run} · {sup} / {unsup}',
                'quality': f'품질 · {test} · {cell}'}.get(key, '통합 현황 · 세 트랙의 현재 요약')

    @app.callback(Output('chat-state', 'data'), Output('chat-messages', 'children'), Output('chat-threads', 'data'),
        Output('chat-question', 'value'), Output('chat-status', 'children'),
        Input('chat-send', 'n_clicks'), Input('chat-new', 'n_clicks'), Input('chat-history', 'data'), Input('open-copilot', 'n_clicks'),
        State('chat-question', 'value'), State('chat-state', 'data'), State('url', 'pathname'),
        State('d-date', 'value'), State('d-part', 'value'), State('d-model', 'value'),
        State('m-run', 'value'), State('m-sup', 'value'), State('m-unsup', 'value'),
        State('q-test', 'value'), State('q-cell', 'value'), State('q-progress', 'value'),State('q-basis','value'),
        prevent_initial_call=True, running=[(Output('chat-send', 'disabled'), True, False), (Output('chat-new', 'disabled'), True, False), (Output('chat-list-toggle', 'disabled'), True, False), (Output('chat-question','disabled'),True,False), (Output('chat-busy','data'),True,False), (Output('chat-progress','style'),{}, {'display':'none'})])
    def chat(send, new, selected, opened, question, current, path, date, part, model, run, sup, unsup, test, cell, progress,basis):
        current = current or {}
        try:
            trigger = ctx.triggered_id
            if trigger == 'chat-send':
                if not question or not question.strip():
                    return no_update, no_update, no_update, no_update, callout('질문을 입력하세요', '', 'warning')
                context = {'track': (path or '/').strip('/') or 'overview', 'date': date, 'part': part, 'model': model,
                    'run': run, 'supervised': sup, 'unsupervised': unsup, 'test': test, 'cell': cell, 'progress': progress,'basis':basis}
                current = request('/api/copilot/ask', 'POST', json={'question': question, 'threadId': current.get('id'), 'revision': current.get('revision', 0), 'context': context})
                selected = current['id']
            elif trigger == 'chat-new':
                current, selected = {}, None
            elif trigger == 'chat-history' and selected:
                current = request(f"/api/copilot/conversations/{selected['id']}")
            status = ''
            try:
                threads = request('/api/copilot/conversations')['threads']
                options = [{'id': t['id'], 'title': t['title']} for t in threads]
            except (ValueError, httpx.HTTPError) as exc:
                options = no_update
                status = callout('대화 목록 조회 불가', str(exc) if isinstance(exc, ValueError) else '연결을 확인하세요.', 'warning')
            return current, messages_view(current.get('messages', [])), options, '' if trigger in ('chat-send', 'chat-new','chat-history') else no_update, status
        except (ValueError, httpx.HTTPError) as exc:
            detail = str(exc) if isinstance(exc, ValueError) else '서버 연결을 확인하세요.'
            return no_update, no_update, no_update, no_update, callout('응답을 완료하지 못했습니다', detail, 'warning')
