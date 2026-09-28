"""Opt-in live verification against the existing Firebase project.

Launch: python scripts/verify_connected_workflow.py serve --scope verification-...
Verify: python scripts/verify_connected_workflow.py check --scope verification-...
Uses existing employees, real session verification, real transactions, and an isolated
manufacturingVerification/{scope} namespace. Never changes official/operational docs.
No passwords, keys, cookies or user IDs are written to the report.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta, datetime, timezone
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def session(service, email, base):
    import httpx
    from firebase_admin import auth
    account = auth.get_user_by_email(email, app=service.app)
    service.employee(account.uid)
    custom = auth.create_custom_token(account.uid, app=service.app).decode()
    response = httpx.post('https://identitytoolkit.googleapis.com/v1/accounts:signInWithCustomToken',
        params={'key': service.config['firebase_web_key']},
        json={'token': custom, 'returnSecureToken': True}, timeout=20)
    if response.status_code != 200:
        raise RuntimeError('Test sign-in failed')
    cookie = auth.create_session_cookie(response.json()['idToken'], expires_in=timedelta(hours=1), app=service.app)
    client = httpx.Client(base_url=base, timeout=90, trust_env=False)
    client.get('/login')
    client.cookies.set('manufacturing_session', cookie.decode() if isinstance(cookie, bytes) else cookie, domain='127.0.0.1', path='/')
    client.headers.update({'Origin': base, 'X-CSRF-Token': client.cookies.get('manufacturing_csrf')})
    assert client.get('/api/me').status_code == 200
    return client


def check(service, args):
    import httpx
    base = f'http://127.0.0.1:{args.port}'
    employee = session(service, 'test@gmail.com', base)
    admin = session(service, 'admin@gmail.com', base)
    results = []
    def ok(name, passed):
        assert passed, name
        results.append({'scenario': name, 'passed': True})
        print('PASS:', name, flush=True)
    meta = employee.get('/api/meta').json()
    version = meta['dataVersion']
    ok('Firestore official snapshot', meta['analysisSource']['state'] == 'ready' and meta['documents'] == 275)
    overview = employee.get('/api/overview').json()
    ok('Overview routes have real targets', all(r['route'].startswith('/') for r in overview['workItems']))
    demand = employee.get('/api/demand', params={'part': 'Part 92'}).json()
    maintenance = employee.get('/api/maintenance').json()
    quality = employee.get('/api/quality', params={'test': 'Test08_NG_chg'}).json()
    ok('Three analytical endpoints', demand['part'] == 'Part 92' and bool(maintenance['events']) and bool(quality))
    targets = [dict(track='demand', target='Part 92', date=demand['date'], model=demand['model'], decision='reviewed'),
        dict(track='maintenance', target=maintenance['events'][0]['id'], decision='reviewed'),
        dict(track='quality', target='Test08_NG_chg', decision='retest')]
    saved = []
    for target in targets:
        body = {**target, 'note': f'[E2E {args.scope}] 실제 Firebase 저장 검증', 'dataVersion': version,
                'requestId': uuid4().hex, 'expectedRevision': 0}
        result = employee.post('/api/records', json=body)
        ok(target['track'] + ' real save', result.status_code == 200)
        record = result.json()
        saved.append(record)
        again = employee.post('/api/records', json=body)
        ok(target['track'] + ' idempotent retry', again.status_code == 200 and again.json()['id'] == record['id'])
        direct = service.record_collection('manufacturingRecords').document(record['id']).get().to_dict()
        ok(target['track'] + ' durable document', direct['note'] == body['note'] and direct['revision'] == 1)
        history = admin.get('/api/records', params={'key': record['key']}).json()['records']
        ok(target['track'] + ' cross-user shared history', any(r['id'] == record['id'] for r in history))
    # Two independently authenticated employees write against the same observed revision.
    common = {**targets[-1], 'dataVersion': version, 'expectedRevision': 1, 'note': f'[E2E {args.scope}] 동시 수정 검증'}
    bodies = [{**common, 'requestId': uuid4().hex, 'decision': 'hold'}, {**common, 'requestId': uuid4().hex, 'decision': 'clear'}]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(client.post, '/api/records', json=body) for client, body in zip([employee, admin], bodies)]
        responses = [f.result() for f in futures]
    ok('Concurrent update rejects one writer', sorted(r.status_code for r in responses) == [200, 409])
    winner = next(r.json() for r in responses if r.status_code == 200)
    ok('Exactly one revision advance', winner['revision'] == 2)
    bad = employee.post('/api/records', json={**bodies[0], 'requestId': uuid4().hex, 'expectedRevision': 2, 'dataVersion': 'old'})
    ok('Outdated analysis cannot be saved', bad.status_code == 409)
    ok('Logout clears session', employee.post('/auth/logout').status_code == 200 and employee.get('/api/records').status_code == 401)
    employee.close()
    employee = session(service, 'test@gmail.com', base)
    states = employee.get('/api/records/state').json()['records']
    ok('Fresh login restores all three saved states', all(r['key'] in states for r in saved))
    history = employee.get('/api/records').json()['records']
    ok('History preserves original and corrected records', len(history) == 4 and any(r['id'] == winner['id'] for r in history))
    ok('Anonymous access denied', httpx.get(base+'/api/records').status_code == 401)
    report = {'at': datetime.now(timezone.utc).isoformat(), 'scope': args.scope, 'project': service.config['project_id'],
        'dataVersion': version, 'results': results, 'records': [{'track': r['track'], 'id': r['id']} for r in saved] + [{'track': 'quality', 'id': winner['id']}],
        'source': employee.get('/api/meta').json()['analysisSource'], 'authentication': 'Existing employee/admin; Firebase custom-token exchange and verified session; browser separately verifies password login',
        'isolation': 'Only manufacturingVerification/{scope}; normal work state and official analysis are unchanged'}
    out = ROOT/'.local/verification'/args.scope
    out.mkdir(parents=True, exist_ok=True)
    (out/'api-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    employee.close(); admin.close()
    print(f'Complete: {len(results)} checks. Report: {out / "api-report.json"}')


def storage_check(service, args):
    """Real storage layer, synthetic actors; explicitly NOT a login/E2E result."""
    from fastapi import HTTPException
    from backend.analysis_store import FirestoreAnalysis
    from backend.data import Repository
    from backend.main import create_api
    from backend.tests.fakes import FakeService, FakeCopilot
    from fastapi.testclient import TestClient
    store = FirestoreAnalysis(service, Repository())
    repo, source = store.acquire()
    results = []
    def ok(name, passed):
        assert passed, name
        results.append({'scenario': name, 'passed': True})
        print('PASS:', name, flush=True)
    # Authentication is an explicit in-process double here; remote analysis remains real.
    client = TestClient(create_api(False, FakeService(), FakeCopilot(), store))
    client.cookies.update({'manufacturing_session': 'a', 'manufacturing_csrf': 'x'})
    client.headers.update({'Origin': 'http://testserver', 'X-CSRF-Token': 'x'})
    for path in ('/api/meta', '/api/overview', '/api/demand?part=Part%2092', '/api/maintenance', '/api/quality?test=Test08_NG_chg'):
        response = client.get(path)
        ok('Real Firestore analysis: ' + path, response.status_code == 200 and response.headers.get('X-Analysis-Source') == 'firestore')
    demand = repo.demand(part='Part 92')
    targets = [dict(track='demand', target='Part 92', date=demand['date'], model=demand['model'], decision='reviewed'),
        dict(track='maintenance', target=repo.maintenance()['events'][0]['id'], decision='reviewed'),
        dict(track='quality', target='Test08_NG_chg', decision='retest')]
    users = [dict(uid='verification-employee', name='검증용 작성자', role='employee'),
             dict(uid='verification-admin', name='검증용 검토자', role='admin')]
    saved = []
    for target in targets:
        key = f"{target['track']}:{target['target']}" + (f":{target['date']}:{target['model']}" if target['track'] == 'demand' else '')
        body = {**target, 'note': f'[STORAGE TEST {args.scope}] 인증 검증과 별도인 저장 계층 검증',
                'dataVersion': source['dataVersion'], 'requestId': uuid4().hex, 'expectedRevision': 0}
        record = service.save_record(body, key, users[0], source['dataVersion'])
        saved.append(record)
        ok(target['track'] + ' real transaction', record['revision'] == 1)
        repeated = service.save_record(body, key, users[0], source['dataVersion'])
        ok(target['track'] + ' idempotent duplicate', repeated['id'] == record['id'])
    common = {**targets[-1], 'note': f'[STORAGE TEST {args.scope}] 동시 수정', 'dataVersion': source['dataVersion'], 'expectedRevision': 1}
    def compete(index):
        try:
            result = service.save_record({**common, 'requestId': uuid4().hex, 'decision': ['hold', 'clear'][index]}, saved[-1]['key'], users[index], source['dataVersion'])
            return 200, result
        except HTTPException as exc:
            return exc.status_code, None
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(compete, [0, 1]))
    ok('Real concurrent transaction: one commit, one conflict', sorted(r[0] for r in responses) == [200, 409])
    restored = type(service)(verification_scope=args.scope)
    states = restored.states()
    history = restored.history()
    ok('Fresh service restores three states', len(states) == 3 and states[saved[-1]['key']]['revision'] == 2)
    ok('Append-only history: four records', len(history) == 4)
    ok('Filtered history restores original and correction', len(restored.history(saved[-1]['key'])) == 2)
    report = {'at': datetime.now(timezone.utc).isoformat(), 'scope': args.scope, 'results': results,
        'source': store.status(), 'recordCount': len(history), 'stateCount': len(states),
        'authentication': 'NOT VERIFIED: synthetic actors for storage; FakeService only inside in-process API analysis test',
        'isolation': 'manufacturingVerification/{scope}; no operational records, auth accounts or official data changed'}
    out = ROOT/'.local/verification'/args.scope
    out.mkdir(parents=True, exist_ok=True)
    (out/'storage-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Complete: {len(results)} checks. Report: {out / "storage-report.json"}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['serve', 'check', 'storage'])
    parser.add_argument('--scope', required=True)
    parser.add_argument('--port', type=int, default=8079)
    args = parser.parse_args()
    os.environ['MANUFACTURING_API_URL'] = f'http://127.0.0.1:{args.port}'
    os.environ['MANUFACTURING_MODE'] = 'connected'
    from backend.firebase_service import FirebaseService
    service = FirebaseService(verification_scope=args.scope)
    if args.action == 'check':
        return check(service, args)
    if args.action == 'storage':
        return storage_check(service, args)
    from backend.preflight import check as preflight
    preflight(True)
    from backend.analysis_store import FirestoreAnalysis
    from backend.data import Repository
    from backend.main import create_api
    import uvicorn
    app = create_api(service=service, analysis_store=FirestoreAnalysis(service, Repository()))
    uvicorn.run(app, host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
