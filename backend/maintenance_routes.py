"""Maintenance uploads and reproducible exports, scoped to the signed-in user."""
import html
import io
import secrets
import time
import zipfile
from collections import OrderedDict
from datetime import datetime, timezone
from threading import Lock

import pandas as pd
from fastapi import File, HTTPException, Request, UploadFile
from fastapi.responses import Response, HTMLResponse
from starlette.concurrency import run_in_threadpool

from backend.maintenance_engine import MAX_BYTES, analyze, assets, inspect
from backend.quality_engine import confusion
from backend.runtime_assets import RUNTIME


def csv_bytes(rows):
    frame = pd.DataFrame(rows)
    for col in frame.select_dtypes(include='object'):
        frame[col] = frame[col].map(lambda v: "'"+v if isinstance(v,str) and v.lstrip().startswith(('=','+','-','@')) else v)
    return frame.to_csv(index=False).encode('utf-8-sig')


def report(data):
    points = data['points']
    scope = '재사용 파일 회고 평가 · 모델 선택에 미사용' if data['run'] in ('WeldingTest_02_OK','WeldingTest_04_NG') else '개발에 사용한 파일 · 독립 시험 아님' if data['run'].startswith('WeldingTest_') else '사용자 업로드 · 외부 성능 검증을 의미하지 않음'
    metric = confusion([p['label'] for p in points],[p['prediction'] for p in points]) if all(p.get('label') is not None for p in points) else None
    lines = ['# BatteryFlow AI · 예지보전 분석 보고서', '',
        f"생성: {datetime.now(timezone.utc).isoformat()}", f"시험: {data['run']}",
        f"모델: {data['supervised']} OR {data['unsupervised']}",
        f"모델 버전: {data['config']['pipelineVersion']}",f"데이터: {data['version']}",
        f"경보 정책: {data['config']['alarmPolicy']['version']}",f'평가 범위: {scope} · 선택 파일 전체 (화면의 표 필터와 별도)', '',
        '## 점검 요약', f"- 측정 {len(points):,}행 / 용접 사이클 {data['cycles']}회",
        f"- 경보 {data['anomalyRows']}행 / 점검 대상 {data['alertCycles']}사이클 / 이벤트 {len(data['events'])}건",
        f"- 출력 0: {sum(p['power']==0 for p in points)}행 / 정상 기준 학습 밖 레시피: {data['unseenRows']}행",
        '- 사이클은 경보가 한 지점 이상인 경우 점검 대상으로 분류합니다. 제품 불량 판정이 아닙니다.',
        '- 정상 행 또는 120초 초과 수집 공백에서 경보 이벤트를 분리합니다.', '', '## 선택 파일의 행 단위 평가']
    if metric:
        lines += [f'- {k}: {"산출 불가" if v is None else f"{v:.4f}" if isinstance(v,float) else v}' for k,v in metric.items()]
    else:
        lines += ['정답 라벨 없음 · 정확도·Recall·혼동행렬은 산출하지 않습니다.']
    lines += ['', '## 해석과 한계', '- 정상 기준은 PageNo별 정상 학습 중앙값이며 미래 출력 예측·공정 합격 범위가 아닙니다.',
        '- 고장 확률·고장 시점·잔여 수명을 추정하지 않습니다. 이상 점수는 모델 임계값 대비 비율입니다.',
        '- 재사용 회고 시험의 연속 고장 사건은 1건입니다. 신규 설비·고장 유형 일반화는 미검증입니다.',
        '- 별도 과거 정상 전용 시간 검증에서 정상 시험 오경보가 발생했습니다. 이동 기준선을 자동 적용하지 않습니다.',
        '- 학습 밖 공정 조건은 성능 검증 범위 밖입니다. 출력 신호와 공정 설정을 함께 확인하세요.',
        '- 현장 SOP·허용범위는 미연결입니다. 작업자 확인 및 설비 담당자의 판단이 필요합니다.']
    return '\n\n'.join(lines), metric


def bundle(data):
    text, metrics = report(data)
    memory = io.BytesIO()
    with zipfile.ZipFile(memory,'w',zipfile.ZIP_DEFLATED) as z:
        for name, rows in [('measurements',data['points']),('cycles',data['cycleDetails']),('events',data['events']),('metrics',[metrics] if metrics else [])]:
            z.writestr(name+'.csv',csv_bytes(rows))
        frame = pd.DataFrame(data['points'])
        conditions = frame.groupby('setPower').agg(rows=('prediction','size'),alarmRows=('prediction','sum'),meanPower=('power','mean'),unseenRows=('unseen','sum')).reset_index()
        conditions['alarmRate'] = conditions.alarmRows/conditions.rows
        z.writestr('conditions.csv',csv_bytes(conditions.to_dict('records')))
        z.writestr('report.md',text.encode('utf-8-sig'))
    return memory.getvalue()


def install_maintenance_routes(app, repo):
    jobs, lock = OrderedDict(), Lock()

    def get_job(token, uid):
        with lock:
            for k in [k for k,v in jobs.items() if v['expires'] < time.monotonic()]:
                del jobs[k]
            item = jobs.get(token)
            if not item or item['uid'] != uid:
                raise HTTPException(404,'업로드가 만료되었거나 접근할 수 없습니다. 다시 검사하세요.')
            return item

    @app.get('/api/maintenance/sample', tags=['예지보전'])
    def sample():
        return Response((RUNTIME/'maintenance/sample.csv').read_bytes(), media_type='text/csv',headers={'Content-Disposition':'attachment; filename="WeldingTest_02_OK.csv"'})

    @app.post('/api/maintenance/inspect', tags=['예지보전'])
    async def upload(request: Request, file: UploadFile = File(...), labels: UploadFile | None = File(None)):
        payload = await file.read(MAX_BYTES+1)
        label = await labels.read(MAX_BYTES+1) if labels else None
        item = await run_in_threadpool(inspect,payload,label)
        token = secrets.token_hex(16)
        with lock:
            while len(jobs) >= 4:
                jobs.popitem(last=False)
            jobs[token] = {'uid':request.state.user['uid'],'expires':time.monotonic()+900,'item':item,'lock':Lock(),'result':None}
        return {**item['public'],'token':token}

    @app.get('/api/maintenance/uploads/{token}/labels', tags=['예지보전'])
    def labels(token: str, request: Request):
        public = get_job(token,request.state.user['uid'])['item']['public']
        rows = [{'sourceRow':i,'sourceSha256':public['sha256'],'label':''} for i in range(public['rows'])]
        return Response(csv_bytes(rows),media_type='text/csv',headers={'Content-Disposition':'attachment; filename="maintenance_labels.csv"'})

    @app.post('/api/maintenance/uploads/{token}/analyze', tags=['예지보전'])
    def infer(token: str, request: Request):
        job = get_job(token,request.state.user['uid'])
        with job['lock']:
            if job['result'] is None:
                job['result'] = analyze(job['item'])
            return job['result']

    @app.get('/api/maintenance/uploads/{token}/export', tags=['예지보전'])
    def export_upload(token: str, request: Request):
        job = get_job(token,request.state.user['uid'])
        if job['result'] is None:
            raise ValueError('분석을 먼저 실행하세요.')
        return Response(bundle(job['result']),media_type='application/zip',headers={'Content-Disposition':'attachment; filename="maintenance_upload.zip"'})

    @app.get('/api/maintenance/reference', tags=['예지보전'])
    def reference():
        a = assets()
        return {k:a[k] for k in ['normalSummary','normalRows','fitEnd','recipes','version']}

    @app.get('/api/maintenance/report', tags=['예지보전'])
    def download_report(run: str = 'WeldingTest_04_NG', supervised: str | None = None, unsupervised: str | None = None, format: str = 'zip'):
        data = repo.maintenance(run,supervised,unsupervised)
        text,_ = report(data)
        if format == 'zip':
            return Response(bundle(data),media_type='application/zip',headers={'Content-Disposition':'attachment; filename="maintenance_report.zip"'})
        if format == 'markdown':
            return Response(text.encode('utf-8-sig'),media_type='text/markdown',headers={'Content-Disposition':'attachment; filename="maintenance_report.md"'})
        if format != 'print':
            raise ValueError('지원하지 않는 보고서 형식입니다.')
        blocks = []
        for line in text.split('\n\n'):
            tag,body = ('h1',line[2:]) if line.startswith('# ') else ('h2',line[3:]) if line.startswith('## ') else ('p',line)
            blocks.append(f'<{tag}>{html.escape(body)}</{tag}>')
        return HTMLResponse('<!doctype html><html lang="ko"><meta charset="utf-8"><title>예지보전 보고서</title>'
            '<style>@font-face{font-family:SUIT Variable;src:url(/auth/fonts/SUIT-Variable.woff2)}body{font-family:SUIT Variable,sans-serif;color:#172033;max-width:1000px;margin:32px auto;font-size:13px}h1{font-size:24px;color:#0A1E5A}h2{font-size:17px;border-bottom:1px solid #E7EAF0;padding-top:16px}p{line-height:1.5;margin:6px 0;overflow-wrap:anywhere}button{padding:10px 20px} @media print{button{display:none}body{margin:0}h2{break-after:avoid}p{break-inside:avoid}}</style>'
            '<button onclick="window.print()">인쇄 / PDF로 저장</button>'+''.join(blocks)+'</html>')
