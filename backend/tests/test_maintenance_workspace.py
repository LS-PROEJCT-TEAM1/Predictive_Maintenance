import hashlib
import io
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from backend.data import Repository, ROOT
from backend.main import create_api
from backend.maintenance_engine import inspect, analyze, assets, MAX_BYTES
from backend.maintenance_routes import bundle, report
from backend.runtime_assets import RUNTIME, verify_runtime
from backend.tests.fakes import FakeService


class MaintenanceWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Repository()
        cls.frame = pd.read_csv(RUNTIME/'maintenance/sample.csv').head(78)
        cls.frame.columns = cls.frame.columns.str.strip()
        cls.payload = cls.frame.to_csv(index=False).encode('utf-8-sig')

    def client(self):
        client = TestClient(create_api(False,service=FakeService()))
        client.cookies.update({'manufacturing_session':'a','manufacturing_csrf':'token'})
        client.headers.update({'Origin':'http://testserver','X-CSRF-Token':'token'})
        return client

    def test_frozen_pair_matches_all_official_measurements(self):
        for run in self.repo.meta()['runs']:
            prefix = f'{self.repo.base}/maintenanceRuns/{run}/measurementChunks/'
            points = sorted([p for k,v in self.repo.docs.items() if k.startswith(prefix) for p in v['points']],key=lambda p:p['sourceRow'])
            frame = pd.DataFrame([{'PageNo':p['pageNo'],'WorkingTime':p['workingTime'],'RealPower':p['signals']['realPower'],
                'Speed':p['signals']['speed'],'Length':p['signals']['length'],'SetPower':p['signals']['setPower'],
                'GateOnTime':p['signals']['gateOnTime'],'SetFrequency':1000,'SetDuty':100} for p in points])
            replay = analyze(inspect(frame.to_csv(index=False).encode()))
            old = self.repo.maintenance(run)
            self.assertEqual([p['prediction'] for p in replay['points']],[p['prediction'] for p in old['points']])
            np.testing.assert_allclose([p['supRatio'] for p in replay['points']],[p['supRatio'] for p in old['points']],atol=1e-8,rtol=1e-6)
            self.assertEqual([(e['start'],e['end']) for e in replay['events']],[(e['start'],e['end']) for e in old['events']])
            self.assertEqual(replay['alertCycles'],old['alertCycles'])
            np.testing.assert_allclose([p['expected'] for p in replay['points']],[p['expected'] for p in old['points']])

    def test_invalid_inputs_never_produce_normal_predictions(self):
        variants = []
        for col,value in [('RealPower',np.nan),('Speed',float('inf')),('SetPower',101),('Length',-1),('PageNo',1.5),('WorkingTime','bad')]:
            bad = self.frame.copy()
            bad[col] = bad[col].astype(object)
            bad.loc[0,col] = value
            variants.append(bad)
        variants += [self.frame.iloc[:-1], self.frame.drop(columns='Speed')]
        bad = self.frame.copy(); bad.loc[1,'WorkingTime'] = bad.loc[0,'WorkingTime']; variants.append(bad)
        for bad in variants:
            with self.assertRaises(ValueError):
                inspect(bad.to_csv(index=False).encode())
        with self.assertRaises(ValueError):
            inspect(b'x'*(MAX_BYTES+1))
        with self.assertRaises(ValueError):
            inspect(b'PageNo, PageNo\n1,1\n')

    def test_zero_and_unseen_recipe_preserved_without_quality_grade(self):
        raw = self.frame.copy()
        raw.loc[0,'RealPower'] = 0
        raw.loc[0,'SetPower'] = 35
        item = inspect(raw.to_csv(index=False).encode())
        data = analyze(item)
        self.assertEqual(data['points'][0]['power'],0)
        self.assertTrue(data['points'][0]['unseen'])
        self.assertEqual(len(data['points']),78)
        self.assertTrue(all(p['label'] is None for p in data['points']))
        self.assertEqual({p['상태'] for p in data['cycleDetails']} - {'점검 대상','경보 없음'},set())

    def test_labels_bound_to_hash_and_row_identity(self):
        labels = pd.DataFrame({'sourceRow':range(78),'sourceSha256':hashlib.sha256(self.payload).hexdigest(),'label':0})
        self.assertIsNotNone(inspect(self.payload,labels.to_csv(index=False).encode())['labels'])
        for col,value in [('sourceSha256','wrong'),('sourceRow',77),('label',2)]:
            bad = labels.copy(); bad.loc[0,col] = value
            with self.assertRaises(ValueError):
                inspect(self.payload,bad.to_csv(index=False).encode())

    def test_upload_owner_expiry_and_export(self):
        client = self.client()
        uploaded = client.post('/api/maintenance/inspect',files={'file':('x.csv',self.payload)})
        self.assertEqual(uploaded.status_code,200,uploaded.text)
        token = uploaded.json()['token']
        url = '/api/maintenance/uploads/'+token
        self.assertEqual(client.get(url+'/export').status_code,422)
        self.assertEqual(client.post(url+'/analyze').status_code,200)
        self.assertEqual(client.get(url+'/labels').status_code,200)
        result = client.get(url+'/export')
        self.assertEqual(result.status_code,200)
        with zipfile.ZipFile(io.BytesIO(result.content)) as z:
            self.assertIn('conditions.csv',z.namelist())
            self.assertIn('정답 라벨 없음',z.read('report.md').decode('utf-8-sig'))
        client.cookies.set('manufacturing_session','b')
        self.assertEqual(client.post(url+'/analyze').status_code,404)
        self.assertEqual(client.get(url+'/labels').status_code,404)
        self.assertEqual(client.get(url+'/export').status_code,404)
        client.cookies.set('manufacturing_session','a')
        import time
        future = time.monotonic()+901
        with patch('backend.maintenance_routes.time.monotonic',return_value=future):
            self.assertEqual(client.get(url+'/export').status_code,404)

    def test_report_truth_scope_and_integrity(self):
        self.assertEqual(sum(p['rows'] for p in assets()['parity']),5226)
        verify_runtime()
        for run in self.repo.meta()['runs']:
            data = self.repo.maintenance(run)
            text,metrics = report(data)
            self.assertIn('maintenance-alarm-v3',text)
            self.assertEqual(sum(metrics[k] for k in ['TP','FP','FN','TN']),len(data['points']))
            with zipfile.ZipFile(io.BytesIO(bundle(data))) as z:
                self.assertEqual(len(pd.read_csv(z.open('measurements.csv'))),len(data['points']))
        response = self.client().get('/api/maintenance/report?format=print')
        self.assertEqual(response.status_code,200)
        self.assertIn('window.print()',response.text)

    def test_auth_and_csrf(self):
        client = TestClient(create_api(False,service=FakeService()))
        self.assertEqual(client.get('/api/maintenance/report').status_code,401)
        self.assertEqual(client.post('/api/maintenance/inspect',files={'file':('x.csv',self.payload)}).status_code,403)

    def test_linked_event_keeps_existing_record_identity(self):
        client = self.client()
        data = client.get('/api/maintenance').json()
        event = data['events'][0]
        self.assertTrue(any(event['event'] in c['관련 이벤트'].split(', ') for c in data['cycleDetails']))
        body = {'track':'maintenance','target':event['id'],'decision':'reviewed','note':'출력과 설정 조건 확인',
                'expectedRevision':0,'requestId':'maintenance-test-000001'}
        saved = client.post('/api/records',json=body)
        self.assertEqual(saved.status_code,200,saved.text)
        self.assertEqual(saved.json()['key'],'maintenance:'+event['id'])
        self.assertEqual(saved.json()['dataVersion'],data['version'])


if __name__ == '__main__':
    unittest.main()
