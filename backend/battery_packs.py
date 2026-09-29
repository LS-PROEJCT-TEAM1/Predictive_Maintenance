"""Authenticated, read-only access to saved pack analysis. No inference here."""
import gzip
import hashlib
import json
import math
import re
import time
from pathlib import Path
from threading import RLock

from fastapi import HTTPException, Query
from backend.defect_policy import load_policy, apply_policy

DEMO_FILE = Path(__file__).resolve().parents[1] / 'runtime/quality/battery_pack_demo.json.gz'


def number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def cell_status(value, z):
    if number(value) is None or number(z) is None:
        return 'missing'
    return 'danger' if abs(z) >= 3 else 'warning' if abs(z) >= 2 else 'normal'


class BatteryPacks:
    def __init__(self, service, demo=False):
        self.service, self.demo = service, demo
        self.lock, self.cache = RLock(), {}
        self.local = None

    def _read(self, key, reader, force=False):
        with self.lock:
            cached = self.cache.get(key)
            if not force and cached and cached[0] > time.monotonic():
                return cached[1]
            try:
                result = reader()
            except HTTPException:
                raise
            except Exception:
                raise HTTPException(503, '팩 분석 결과를 조회하지 못했습니다. 연결을 확인하고 새로고침하세요.')
            self.cache[key] = (time.monotonic()+60, result)
            return result

    def _demo(self):
        if self.local is None:
            self.local = json.loads(gzip.decompress(DEMO_FILE.read_bytes()))
        return self.local

    def index(self, force=False):
        def read():
            policy = load_policy()
            if self.demo:
                packs, config = self._demo()['packs'], self._demo()['config']
                dashboard = self._demo().get('dashboard') or {}
            else:
                db = self.service.db
                packs = {s.id: s.to_dict() for s in db.collection('battery_packs').stream(timeout=20, retry=None)}
                config = db.document('battery_meta/config').get(timeout=20, retry=None).to_dict() or {}
                dashboard = db.document('battery_meta/dashboard').get(timeout=20, retry=None).to_dict() or {}
            rows = []
            for ident, doc in packs.items():
                if not re.fullmatch(r'\d+_(chg|dchg)', ident) or not doc:
                    continue
                ratio = number(doc.get('anomaly_ratio'))
                rows.append(apply_policy({**doc, 'pack_id': ident, 'pack_no': ident.rsplit('_', 1)[0],
                             'process': ident.rsplit('_', 1)[1],
                             'ai_verdict': doc.get('ai_verdict') if doc.get('ai_verdict') in ('OK', 'NG') else '미확인',
                             'anomalyPercent': ratio*100 if ratio is not None else None}, policy))
            rows.sort(key=lambda r: (int(r['pack_no']), r['process']))
            version = hashlib.sha256(json.dumps({'rows': rows, 'config': config, 'dashboard': dashboard, 'defectPolicy':policy}, sort_keys=True, default=str).encode()).hexdigest()[:20]
            return {'rows': rows, 'packNumbers': sorted({r['pack_no'] for r in rows}, key=int),
                    'count': len(rows), 'ngCount': sum(r['ai_verdict'] == 'NG' for r in rows),
                    'dataVersion': 'battery-packs-'+version, 'config': config,
                    'defectThresholds': policy['thresholdsByProcess'],
                    'defectPolicyVersion': policy['version']}
        return self._read('index', read, force)

    def detail(self, ident, snapshot='last', force=False):
        index = self.index(force)
        row = next((r for r in index['rows'] if r['pack_id'] == ident), None)
        if row is None:
            raise HTTPException(404, '선택한 팩·공정의 결과가 없습니다.')
        def read():
            if self.demo:
                return self._demo()['details'].get(ident, {})
            base = self.service.db.collection('battery_packs').document(ident)
            return {kind: base.collection(kind).document(doc).get(timeout=20, retry=None).to_dict() or {}
                    for kind, doc in [('heatmap', 'cells'), ('series', 'score')]}
        detail = self._read('detail:'+ident, read, force)
        heatmap = detail.get('heatmap') or {}
        snap = (heatmap.get('snapshots') or {}).get(snapshot) or {}
        ids = index['config'].get('cell_ids') or []
        values, zs = snap.get('v') or [], snap.get('z') or []
        cells, counts = [], dict.fromkeys(['normal', 'warning', 'danger', 'missing'], 0)
        # A shape mismatch is unavailable evidence, not 176 fabricated normal cells.
        if len(ids) == 176 and len(values) == len(zs) == len(ids):
            for ident_cell, value, z in zip(ids, values, zs):
                status = cell_status(value, z)
                cells.append({'id': ident_cell, 'value': number(value), 'z': number(z), 'status': status, 'invalid': False})
                counts[status] += 1
        series = detail.get('series') or {}
        return {'summary': row, 'snapshot': {'cells': cells, 't': snap.get('t'), 'kind': snapshot},
                'counts': counts, 'series': series, 'dataVersion': index['dataVersion'],
                'testId': 'pack:'+row['pack_id'], 'thresholds': index['config'].get('thresholds') or {},
                'defectThresholds': index['defectThresholds'][row['process']],
                'defectPolicyVersion': index['defectPolicyVersion']}

    def validate_decision(self, body):
        index = self.index(force=True)
        ident = body.target.removeprefix('pack:')
        if ident not in {r['pack_id'] for r in index['rows']}:
            raise ValueError('팩·공정 선택을 확인하세요.')
        if not body.dataVersion or body.dataVersion != index['dataVersion']:
            raise HTTPException(409, '팩 분석 결과가 변경되었습니다. 새로고침 후 다시 검토하세요.')
        if body.decision not in ('clear', 'retest', 'hold'):
            raise ValueError('작업자 판정을 선택하세요.')
        return {**body.model_dump(), 'key': 'quality:'+body.target,
                'dataVersion': index['dataVersion'], 'persistence': 'preview-only'}


def install_pack_routes(app, store, service):
    @app.get('/api/battery-packs')
    def index(refresh: bool = False):
        data = store.index(refresh)
        return {k: v for k, v in data.items() if k != 'config'}

    @app.get('/api/battery-packs/{ident}')
    def detail(ident: str, snapshot: str = Query('last', pattern='^(last|peak)$'), refresh: bool = False):
        return store.detail(ident, snapshot, refresh)

    @app.get('/api/battery-packs/{ident}/history')
    def history(ident: str):
        if ident not in {r['pack_id'] for r in store.index()['rows']}:
            raise HTTPException(404, '팩·공정 선택을 확인하세요.')
        return {'records': service.history('quality:pack:'+ident)}
