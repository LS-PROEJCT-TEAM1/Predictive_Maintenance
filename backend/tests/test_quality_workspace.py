import hashlib
import io
import json
import unittest

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.data import Repository
from backend.main import create_api
from backend.quality_engine import CV, TP, FEATURES, clean_frame, features, scores, inspect
from backend.quality_routes import diagnostics, exploration_assets
from backend.runtime_assets import quality_path, verify_runtime
from backend.tests.fakes import FakeService


class QualityWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Repository()
        cls.raw = pd.read_csv(quality_path('Test03_OK_chg')).head(30)
        cls.payload = cls.raw.to_csv(index=False).encode('utf-8')

    def client(self):
        client = TestClient(create_api(False, service=FakeService()))
        client.cookies.update({'manufacturing_session': 'a', 'manufacturing_csrf': 'token'})
        client.headers.update({'Origin': 'http://testserver', 'X-CSRF-Token': 'token'})
        return client

    def test_frozen_model_all_five_tests_match_saved_predictions(self):
        stored = pd.DataFrame(self.repo.artifact('quality_scores'))
        for test in self.repo.meta()['tests']:
            clean = clean_frame(pd.read_csv(quality_path(test)))
            t2, spe, pred = scores(features(clean))
            old = stored[stored['파일'] == test].sort_values('시점')
            np.testing.assert_array_equal(pred, old['예측'])
            np.testing.assert_allclose(t2, old['Hotelling_T2'], atol=.0001, rtol=1e-7)
            np.testing.assert_allclose(spe, old['SPE'], atol=.0001, rtol=1e-7)

    def test_exact_confusion_uses_full_rows_not_chart_sample(self):
        result = diagnostics(self.repo, 'Test07_NG_dchg')
        self.assertEqual(result['selected']['FN'], 82)
        self.assertEqual(result['selected']['TP'], 145)
        self.assertEqual(result['all']['FP'], 10)
        self.assertEqual(sum(result['all'][k] for k in ('TP', 'FP', 'FN', 'TN')), 10770)
        self.assertLessEqual(len(result['points']), 502)
        self.assertEqual(len(result['variance']), 6)
        self.assertEqual(result['normalRows'], 56805)

    def test_normal_curve_provenance_and_runtime_integrity(self):
        data = exploration_assets()
        normal = [x for x in data['curves'] if x['grade'] == 'normal-reference']
        self.assertEqual(len(normal), 10)
        self.assertEqual({x['mode'] for x in normal}, {'chg', 'dchg'})
        self.assertTrue(all(len(x['points']) <= 201 for x in normal))
        self.assertIn('quality/pca_frozen.json', {p['file'] for p in verify_runtime()['files']})
        sample = self.client().get('/api/quality/sample')
        self.assertEqual(sample.status_code, 200)
        self.assertEqual(len(pd.read_csv(io.BytesIO(sample.content))), 1421)

    def test_upload_analysis_does_not_change_seed_and_requires_owner(self):
        client = self.client()
        before = self.repo.manifest.copy()
        result = client.post('/api/quality/inspect', files={'file': ('trial.csv', self.payload)}, data={'mode': 'chg'})
        self.assertEqual(result.status_code, 200, result.text)
        token = result.json()['token']
        template = client.get(f'/api/quality/uploads/{token}/labels')
        self.assertIn('sourceSha256', template.text)
        analysis = client.post(f'/api/quality/uploads/{token}/analyze')
        self.assertEqual(analysis.status_code, 200, analysis.text)
        self.assertEqual(len(analysis.json()['results']), 30)
        self.assertIsNone(analysis.json()['confusion'])
        self.assertEqual(Repository().manifest, before)
        client.cookies.set('manufacturing_session', 'b')
        self.assertEqual(client.post(f'/api/quality/uploads/{token}/analyze').status_code, 404)
        self.assertEqual(client.get(f'/api/quality/uploads/{token}/labels').status_code, 404)

    def test_label_binding_hash_and_row_ids_not_just_length(self):
        digest = hashlib.sha256(self.payload).hexdigest()
        labels = pd.DataFrame({'sourceRow': range(30), 'sourceSha256': digest, 'label': [0]*29+[1]})
        shuffled = labels.sample(frac=1, random_state=42)
        result = inspect(self.payload, 'chg', shuffled.to_csv(index=False).encode())
        self.assertEqual(result['labels'][-1], 1)
        self.assertEqual(result['labels'].sum(), 1)
        labels.loc[0, 'sourceSha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            inspect(self.payload, 'chg', labels.to_csv(index=False).encode())
        labels['sourceSha256'] = digest
        labels.loc[0, 'sourceRow'] = 1
        with self.assertRaisesRegex(ValueError, '일대일'):
            inspect(self.payload, 'chg', labels.to_csv(index=False).encode())

    def test_invalid_contracts_rejected(self):
        variants = [self.raw.drop(columns=[CV[0]]), self.raw.iloc[::-1], pd.concat([self.raw, self.raw.iloc[[0]]]),
                    self.raw.assign(SerialNumber=range(len(self.raw)))]
        for frame in variants:
            with self.assertRaises(ValueError):
                inspect(frame.to_csv(index=False).encode(), 'chg')
        with self.assertRaises(ValueError):
            inspect(self.payload, 'automatic')
        with self.assertRaises(ValueError):
            inspect(self.payload, 'chg', b'')
        client = self.client()
        self.assertEqual(client.post('/api/quality/inspect', files={'file': ('bad.csv', b'bad')}, data={'mode': 'chg'}).status_code, 422)

    def test_faults_reported_without_modifying_raw(self):
        frame = self.raw.copy()
        frame[CV[-1]] = 16.3
        result = inspect(frame.to_csv(index=False).encode(), 'chg')
        self.assertEqual(frame[CV[-1]].iloc[0], 16.3)
        self.assertTrue(result['frame'][CV[-1]].between(2, 5).all())
        self.assertEqual(result['public']['beforeAfter'][2]['처리 전'], len(frame))
        self.assertEqual(result['public']['beforeAfter'][3]['처리 전'], 1)

    def test_history_has_no_synthetic_records_and_filters_track(self):
        client = self.client()
        self.assertEqual(client.get('/api/quality/history').json()['rows'], [])
        body = {'track': 'quality', 'target': 'Test07_NG_dchg', 'decision': 'hold', 'note': '점검 메모',
                'expectedRevision': 0, 'requestId': 'quality-test-000001'}
        self.assertEqual(client.post('/api/records', json=body).status_code, 200)
        records = client.get('/api/quality/history?test=Test07_NG_dchg').json()['rows']
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['작업자 판정'], '출하 보류')
        self.assertEqual(client.get('/api/quality/history?test=Test03_OK_chg').json()['rows'], [])

    def test_no_auth_or_csrf_upload_rejected(self):
        client = TestClient(create_api(False, service=FakeService()))
        self.assertEqual(client.get('/api/quality/diagnostics').status_code, 401)
        self.assertEqual(client.post('/api/quality/inspect', files={'file': ('x.csv', self.payload)}, data={'mode': 'chg'}).status_code, 403)


if __name__ == '__main__':
    unittest.main()
