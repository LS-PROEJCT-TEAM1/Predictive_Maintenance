import hashlib
import unittest

from fastapi.testclient import TestClient

from backend.data import SEED
from backend.main import create_api
from backend.tests.fakes import FakeService


class LocalApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(create_api(mount_ui=False, service=FakeService()))
        cls.client.cookies.update({'manufacturing_session': 'a', 'manufacturing_csrf': 'test'})
        cls.client.headers.update({'Origin': 'http://testserver', 'X-CSRF-Token': 'test'})

    def test_official_seed_and_connection_status(self):
        data = self.client.get('/api/health').json()
        self.assertEqual(data['documents'], 266)
        self.assertEqual(data['dataVersion'], '2026-09-26.v2')
        self.assertEqual(data['firebase'], 'configured')

    def test_demand_aggregate_matches_parts_and_filters(self):
        data = self.client.get('/api/demand').json()
        self.assertEqual(data['count'], 109)
        self.assertEqual(data['reviewCount'], 56)
        self.assertAlmostEqual(data['forecast'], sum(row['forecast'] for row in data['rows']), places=2)
        row = next(row for row in data['rows'] if row['part'] == 'Part 92')
        part = self.client.get('/api/demand', params={'part': 'Part 92'}).json()
        self.assertEqual(part['forecast'], row['forecast'])
        other = self.client.get('/api/demand', params={'part': 'Part 92', 'model': 'XGBoost'}).json()
        self.assertNotEqual(part['forecast'], other['forecast'])

    def test_event_intervals_match_predicted_rows(self):
        data = self.client.get('/api/maintenance').json()
        self.assertEqual(len(data['events']), 3)
        self.assertEqual(sum(e['rows'] for e in data['events']), data['anomalyRows'])
        self.assertEqual(sum(p['prediction'] for p in data['points']), data['anomalyRows'])
        for event in data['events']:
            self.assertTrue(all(p['prediction'] for p in data['points'][event['start']:event['end']+1]))

    def test_quality_selection_and_progress(self):
        full = self.client.get('/api/quality').json()
        partial = self.client.get('/api/quality', params={'cell': 'M01CV01', 'progress': 50}).json()
        self.assertEqual(full['abnormalPointCount'], 145)
        self.assertEqual(full['abnormalSegmentCount'], 1)
        self.assertLessEqual(max(p['progressPct'] for p in partial['cellSeries']), 50)
        self.assertLess(len(partial['series']), len(full['series']))
        self.assertEqual(partial['selectedCell'], 'M01CV01')

    def test_validation_quality_and_result_types(self):
        for track in ['demand', 'maintenance', 'quality']:
            for path in [f'/api/validation/{track}', f'/api/data-quality/{track}']:
                with self.subTest(path=path):
                    response = self.client.get(path)
                    self.assertEqual(response.status_code, 200)
                    self.assertTrue(response.json()['rows'])
            for kind in ['raw', 'processed', 'prediction', 'evaluation']:
                self.assertEqual(self.client.get(f'/api/results/{track}', params={'kind': kind}).status_code, 200)

    def test_invalid_contexts_are_rejected(self):
        for path in ['/api/demand?part=missing', '/api/demand?date=2099-01-01', '/api/demand?model=invalid',
                     '/api/maintenance?run=missing', '/api/quality?test=missing', '/api/quality?cell=invalid',
                     '/api/quality?progress=0', '/api/quality?progress=101', '/api/results/invalid']:
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 422)

    def test_preview_is_scoped_and_never_changes_seed(self):
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SEED.iterdir() if p.is_file()}
        body = {'track': 'demand', 'target': 'Part 92', 'date': '2021-11-01', 'model': '3-day Moving Average', 'decision': 'reviewed', 'note': 'test'}
        result = self.client.post('/api/preview/decision', json=body)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()['persistence'], 'preview-only')
        self.assertIn('2021-11-01', result.json()['key'])
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SEED.iterdir() if p.is_file()}
        self.assertEqual(before, after)

    def test_preview_rejects_invalid_decision_and_target(self):
        for body in [{'track': 'quality', 'target': 'Test07_NG_dchg', 'decision': 'reviewed'},
                     {'track': 'quality', 'target': 'missing', 'decision': 'hold'},
                     {'track': 'maintenance', 'target': 'fake', 'decision': 'reviewed'},
                     {'track': 'demand', 'target': 'Part 1', 'decision': 'reviewed'}]:
            self.assertEqual(self.client.post('/api/preview/decision', json=body).status_code, 422)

    def test_csv_inference_uses_existing_model(self):
        template = self.client.get('/api/demand/template').content
        result = self.client.post('/api/demand/infer', files={'file': ('input.csv', template, 'text/csv')})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()['recommended_forecast'], 110)
        self.assertEqual(result.json()['target_date'], '2026-09-29')
        self.assertTrue(result.json()['known_part'])
        self.assertIsInstance(result.json()['xgboost_prediction'], float)

    def test_csv_rejects_bad_shape_negative_values_and_oversize(self):
        template = self.client.get('/api/demand/template').content
        for payload in [b'wrong\n1\n', template.replace(b',100,', b',-100,'), template.replace(b'2026-09-25', b'2026-09-24')]:
            result = self.client.post('/api/demand/infer', files={'file': ('bad.csv', payload, 'text/csv')})
            self.assertEqual(result.status_code, 422)
        result = self.client.post('/api/demand/infer', files={'file': ('big.csv', b'x' * 1_000_001, 'text/csv')})
        self.assertEqual(result.status_code, 413)

    def test_export_delivers_filtered_rows_and_neutralizes_formulas(self):
        response = self.client.post('/api/exports', json={'name': 'demand_review', 'rows': [{'part': 'Part 92', 'note': '=1+1', 'gap': -12}]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['count'], 1)
        download = self.client.get(response.json()['url'])
        self.assertEqual(download.status_code, 200)
        self.assertIn('attachment;', download.headers['content-disposition'])
        self.assertIn("'=1+1", download.content.decode('utf-8-sig'))
        self.assertIn('-12', download.content.decode('utf-8-sig'))
        self.assertEqual(self.client.get('/api/exports/missing').status_code, 404)


if __name__ == '__main__':
    unittest.main()
