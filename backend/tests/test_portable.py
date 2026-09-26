import gzip
import hashlib
import os
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.runtime_assets import ROOT, RUNTIME, verify_runtime
from backend.main import create_api
from backend.data import Repository


class PortableTests(unittest.TestCase):
    def test_bundle_integrity_and_all_quality_tests(self):
        manifest = verify_runtime()
        repo = Repository()
        self.assertEqual(manifest['dataVersion'], repo.manifest['dataVersion'])
        for entry in manifest['files']:
            if entry['file'].endswith('.gz'):
                raw = gzip.decompress((RUNTIME / entry['file']).read_bytes())
                self.assertEqual(hashlib.sha256(raw).hexdigest(), entry['sourceSha256'])
        for test in repo.meta()['tests']:
            for progress in (1, 50, 100):
                result = repo.quality(test, 'M02CV01', progress)
                self.assertTrue(result['cellSeries'])
                self.assertEqual(result['progress'], progress)
        for track in ('demand', 'maintenance', 'quality'):
            self.assertTrue(repo.validation(track)['rows'])

    def test_demo_never_constructs_cloud_clients_and_cannot_save(self):
        with patch.dict(os.environ, {'MANUFACTURING_MODE': 'demo'}), patch('backend.main.FirebaseService') as firebase, patch('backend.main.Copilot') as copilot:
            client = TestClient(create_api(False))
        firebase.assert_not_called()
        copilot.assert_not_called()
        self.assertEqual(client.get('/api/demand').status_code, 401)
        self.assertEqual(client.get('/api/health').json()['firebase'], 'disabled')
        self.assertIn('체험 시작', client.get('/login').text)
        client.headers.update({'Origin': 'http://testserver', 'X-CSRF-Token': client.cookies['manufacturing_csrf']})
        self.assertEqual(client.post('/auth/login', json={'email':'demo@local.test','password':'local-demo'}).status_code, 200)
        client.headers['X-CSRF-Token'] = client.cookies['manufacturing_csrf']
        self.assertEqual(client.get('/api/demand').status_code, 200)
        self.assertEqual(client.get('/api/records/state').status_code, 503)
        self.assertEqual(client.post('/api/copilot/ask', json={'question':'근거는?'}).status_code, 503)
        body = {'track':'quality','target':'Test07_NG_dchg','decision':'retest','requestId':'demo-request-0001','expectedRevision':0}
        self.assertEqual(client.post('/api/records', json=body).status_code, 503)
        client.post('/auth/logout')
        self.assertEqual(client.get('/api/demand').status_code, 401)


if __name__ == '__main__':
    unittest.main()
