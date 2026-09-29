"""Quality workspace read models and bounded, owner-scoped upload jobs."""
import io
import json
import secrets
import time
from collections import OrderedDict
from functools import lru_cache
from threading import Lock

import numpy as np
import pandas as pd
from fastapi import File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from backend.quality_engine import MAX_BYTES, analyze, confusion, inspect, model
from backend.runtime_assets import RUNTIME, quality_path


@lru_cache(maxsize=1)
def exploration_assets():
    return json.loads((RUNTIME/'quality/exploration.json').read_text(encoding='utf-8'))


def diagnostics(repo, test):
    if test not in repo.meta()['tests']:
        raise ValueError('시험 ID를 확인하세요.')
    frame = pd.DataFrame(repo.artifact('quality_scores'))
    rows, selected = [], frame[frame['파일'] == test]
    for name, group in frame.groupby('파일', sort=False):
        doc = repo.get('qualityTests', name)
        metric = confusion(group['label'], group['예측'])
        rows.append({'시험': name, '공정': doc['modeLabel'], '행 수': len(group), '실제 이상': int(group['label'].sum()),
                     '예측 이상': int(group['예측'].sum()), 'AI 상태': '검토 필요' if group['예측'].any() else '이상 미탐지', **metric})
    ids = np.unique(np.r_[np.linspace(0, len(selected)-1, min(500, len(selected)), dtype=int),
                          selected['SPE'].to_numpy().argmax(), selected['Hotelling_T2'].to_numpy().argmax()])
    points = [{'index': int(r['시점']), 't2': float(r['Hotelling_T2']), 'spe': float(r['SPE']),
               'prediction': int(r['예측']), 'label': int(r['label'])} for _, r in selected.iloc[ids].iterrows()]
    doc = repo.get('qualityTests', test)
    return {'test': test, 'points': points, 'rows': rows, 'selected': confusion(selected['label'], selected['예측']),
            'all': confusion(frame['label'], frame['예측']), 'thresholds': doc['pcaThresholds'],
            'variance': model()['varianceRatio'], 'normalRows': model()['normalRows'],
            'folds': repo.artifact('quality_folds'), 'version': model()['version']}


def install_quality_routes(app, repo, service):
    jobs, lock = OrderedDict(), Lock()

    def get_job(token, uid):
        with lock:
            now = time.monotonic()
            for key in [k for k, v in jobs.items() if v['expires'] < now]:
                del jobs[key]
            job = jobs.get(token)
            if job is None or job['uid'] != uid:
                raise HTTPException(404, '업로드가 만료되었거나 접근할 수 없습니다. 다시 검사하세요.')
            return job

    @app.get('/api/quality/diagnostics', tags=['품질'])
    def quality_diagnostics(test: str = 'Test07_NG_dchg'):
        return diagnostics(repo, test)

    @app.get('/api/quality/sample', tags=['품질'])
    def sample():
        import gzip
        path = quality_path('Test03_OK_chg')
        raw = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
        return Response(raw, media_type='text/csv', headers={'Content-Disposition': 'attachment; filename="Test03_OK_chg.csv"'})

    @app.get('/api/quality/exploration', tags=['품질'])
    def quality_exploration(test: str = 'Test07_NG_dchg'):
        doc = repo.get('qualityTests', test)
        mode = 'dchg' if doc['modeLabel'] == '방전' else 'chg'
        assets = exploration_assets()
        return {'test': test, 'mode': mode, 'curves': [p for p in assets['curves'] if p['mode'] == mode and (p['grade'] == 'normal-reference' or p['file'] == test)],
                'boxes': assets['temperatureBoxes'], 'quality': repo.data_quality('quality')}

    @app.get('/api/quality/history', tags=['품질'])
    def quality_history(test: str = 'all'):
        if test != 'all' and test not in repo.meta()['tests']:
            raise ValueError('시험 ID를 확인하세요.')
        saved = service.history(None if test == 'all' else 'quality:'+test)
        result = []
        names = {'clear': '이상 없음', 'retest': '재시험 요청', 'hold': '출하 보류'}
        for row in saved:
            if row.get('track') != 'quality' or row.get('target') not in repo.meta()['tests'] or (test != 'all' and row.get('target') != test):
                continue
            doc = repo.get('qualityTests', row['target'])
            result.append({'시각': row.get('at'), '시험': row['target'], '공정': doc['modeLabel'],
                           'AI 상태': '검토 필요' if doc['abnormalPointCount'] else '이상 미탐지',
                           '작업자 판정': names.get(row.get('decision'), row.get('decision')), '작성자': row.get('actor') or row.get('actorUid'),
                           '메모': row.get('note', ''), '이전 판정': names.get(row.get('previousDecision'), '—'),
                           '수정 차수': row.get('revision'), '데이터 버전': row.get('dataVersion')})
        return {'rows': result, 'scope': '최근 업무 기록 100건 중 품질 기록' if test == 'all' else '선택 시험의 조회 기록 최대 100건'}

    @app.post('/api/quality/inspect', tags=['품질'])
    async def quality_inspect(request: Request, file: UploadFile = File(...), mode: str = Form(...), labels: UploadFile | None = File(None)):
        payload = await file.read(MAX_BYTES+1)
        label_payload = await labels.read(MAX_BYTES+1) if labels else None
        item = await run_in_threadpool(inspect, payload, mode, label_payload)
        token = secrets.token_hex(16)
        with lock:
            while len(jobs) >= 4:
                jobs.popitem(last=False)
            jobs[token] = {'uid': request.state.user['uid'], 'expires': time.monotonic()+900, 'item': item}
        return {**item['public'], 'token': token, 'expiresSeconds': 900}

    @app.get('/api/quality/uploads/{token}/labels', tags=['품질'])
    def label_template(token: str, request: Request):
        job = get_job(token, request.state.user['uid'])['item']
        frame = pd.DataFrame({'sourceRow': range(job['public']['rows']), 'sourceSha256': job['public']['sha256'], 'label': ''})
        return Response(frame.to_csv(index=False).encode('utf-8-sig'), media_type='text/csv', headers={'Content-Disposition': 'attachment; filename="quality_labels_template.csv"'})

    @app.post('/api/quality/uploads/{token}/analyze', tags=['품질'])
    def quality_analyze(token: str, request: Request):
        item = get_job(token, request.state.user['uid'])['item']
        return analyze(item)
