import json

import httpx
from dash import Input, Output, State, ctx, dcc, html, no_update
import dash_mantine_components as dmc
from frontend.components import badge, callout, icon


def drawer():
    return dmc.Drawer(id='copilot-drawer', title='AI Copilot', position='right', size=560,
        closeButtonProps={'aria-label': '닫기'}, children=[
        dcc.Store(id='chat-state', data={}),
        html.Div([badge('본인만 조회', 'info'), badge('Gemini · 문서 근거', 'success')], className='copilot-badges'),
        html.P('선택한 화면 데이터와 프로젝트 문서로 답변합니다. 질문과 검색 근거는 Gemini API로 전송됩니다.', className='section-note'),
        html.Div(id='copilot-context', className='copilot-context'),
        dmc.Group([dmc.Select(id='chat-history', label='내 대화 이력', placeholder='저장된 대화 선택', data=[], searchable=True, className='chat-select'),
                   dmc.Button('새 대화', id='chat-new', variant='default', size='sm')], align='flex-end'),
        html.Div(id='chat-status', role='status'),
        dcc.Loading(html.Div(id='chat-messages', children=callout('근거부터 확인하세요', '예: 현재 품질 시험에서 먼저 확인할 점과 한계를 설명해 줘.'), className='chat-messages'), color='#0A1E5A', delay_show=200),
        dmc.Textarea(id='chat-question', label='Copilot에게 질문', placeholder='현재 결과의 근거와 확인할 점을 알려줘', minRows=3, maxRows=6, autosize=True, inputProps={'maxLength': 2000}),
        dmc.Button('질문 보내기', id='chat-send', leftSection=icon('send', 16), fullWidth=True, className='spaced-button'),
        dmc.Button('답변으로 검토 초안 만들기', id='chat-draft', variant='outline', fullWidth=True, className='spaced-button'),
        html.P('현장 SOP 미연결 · 최종 판정과 업무 기록 저장은 직원이 직접 확인합니다.', className='section-note')])


def messages_view(messages):
    items = []
    for message in messages:
        sources = []
        for source in message.get('citations', []):
            if source['id'] == 'SCREEN':
                detail = html.Pre(json.dumps(source.get('snapshot', {}), ensure_ascii=False, indent=2), className='source-snapshot')
            else:
                detail = html.Div([html.P(source.get('excerpt', ''), className='source-excerpt'),
                    html.A('문서 원문 열기', href=f"/api/copilot/sources/{source['source']}", target='_blank', rel='noopener')])
            sources.append(html.Details([html.Summary(f"[{source['id']}] {source['title']}"), detail], className='chat-source'))
        items.append(html.Article([html.Div('나' if message['role'] == 'user' else 'AI Copilot', className='chat-speaker'),
            html.Div(message.get('text', ''), className='chat-text'), *sources,
            html.Small(message.get('at', '')[:19].replace('T', ' ')+' UTC')], className='chat-message '+message['role']))
    return items or callout('새 대화', '선택한 화면에 대해 질문해 보세요.')


def register(app, request):
    @app.callback(Output('copilot-context', 'children'), Input('url', 'pathname'), Input('d-part', 'value'), Input('d-date', 'value'), Input('d-model', 'value'),
        Input('m-run', 'value'), Input('m-sup', 'value'), Input('m-unsup', 'value'), Input('q-test', 'value'), Input('q-cell', 'value'))
    def context_label(path, part, date, model, run, sup, unsup, test, cell):
        key = (path or '/').strip('/')
        return {'demand': f'공급망 · {date} · {part} · {model}', 'maintenance': f'예지보전 · {run} · {sup} / {unsup}',
                'quality': f'품질 · {test} · {cell}'}.get(key, '통합 현황 · 세 트랙의 현재 요약')

    @app.callback(Output('chat-state', 'data'), Output('chat-messages', 'children'), Output('chat-history', 'data'), Output('chat-history', 'value'),
        Output('chat-question', 'value'), Output('chat-status', 'children'),
        Input('chat-send', 'n_clicks'), Input('chat-new', 'n_clicks'), Input('chat-history', 'value'), Input('open-copilot', 'n_clicks'),
        State('chat-question', 'value'), State('chat-state', 'data'), State('url', 'pathname'),
        State('d-date', 'value'), State('d-part', 'value'), State('d-model', 'value'),
        State('m-run', 'value'), State('m-sup', 'value'), State('m-unsup', 'value'),
        State('q-test', 'value'), State('q-cell', 'value'), State('q-progress', 'value'),
        prevent_initial_call=True, running=[(Output('chat-send', 'disabled'), True, False), (Output('chat-new', 'disabled'), True, False), (Output('chat-history', 'disabled'), True, False)])
    def chat(send, new, selected, opened, question, current, path, date, part, model, run, sup, unsup, test, cell, progress):
        current = current or {}
        try:
            trigger = ctx.triggered_id
            if trigger == 'chat-send':
                if not question or not question.strip():
                    return no_update, no_update, no_update, no_update, no_update, callout('질문을 입력하세요', '', 'warning')
                context = {'track': (path or '/').strip('/') or 'overview', 'date': date, 'part': part, 'model': model,
                    'run': run, 'supervised': sup, 'unsupervised': unsup, 'test': test, 'cell': cell, 'progress': progress}
                current = request('/api/copilot/ask', 'POST', json={'question': question, 'threadId': current.get('id'), 'revision': current.get('revision', 0), 'context': context})
                selected = current['id']
            elif trigger == 'chat-new':
                current, selected = {}, None
            elif trigger == 'chat-history' and selected:
                current = request(f'/api/copilot/conversations/{selected}')
            status = ''
            try:
                threads = request('/api/copilot/conversations')['threads']
                options = [{'value': t['id'], 'label': t['title']} for t in threads]
            except (ValueError, httpx.HTTPError) as exc:
                options = no_update
                status = callout('대화 목록 조회 불가', str(exc) if isinstance(exc, ValueError) else '연결을 확인하세요.', 'warning')
            return current, messages_view(current.get('messages', [])), options, selected, '' if trigger in ('chat-send', 'chat-new') else no_update, status
        except (ValueError, httpx.HTTPError) as exc:
            detail = str(exc) if isinstance(exc, ValueError) else '서버 연결을 확인하세요.'
            return no_update, no_update, no_update, no_update, no_update, callout('응답을 완료하지 못했습니다', detail, 'warning')
