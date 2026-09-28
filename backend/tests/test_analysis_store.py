import copy
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi import HTTPException
from fastapi.testclient import TestClient
from backend.analysis_store import FirestoreAnalysis
from backend.data import Repository
from backend.main import create_api
from backend.tests.fakes import FakeService, FakeCopilot


class AnalysisStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = Repository()

    def setUp(self):
        self.time = 0
        self.store = FirestoreAnalysis(None, self.contract, clock=lambda: self.time)
        self.root = SimpleNamespace(update_time='release-1', to_dict=lambda: self.contract.docs[self.contract.base])
        self.store._read_root = Mock(return_value=self.root)
        self.store._read_documents = Mock(side_effect=lambda: copy.deepcopy(self.contract.docs))

    def test_remote_snapshot_and_root_only_cache_check(self):
        repo, status = self.store.acquire()
        self.assertEqual(repo.docs, self.contract.docs)
        self.assertIsNot(repo.docs, self.contract.docs)
        self.assertEqual(status['state'], 'ready')
        for _ in range(10):
            self.store.acquire()
        self.assertEqual(self.store._read_root.call_count, 2)
        self.time = 61
        self.store.acquire()
        self.assertEqual(self.store._read_root.call_count, 3)
        self.store._read_documents.assert_called_once()

    def test_parallel_first_reads_load_once(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.store.acquire(), range(4)))
        self.assertTrue(all(r[0] is results[0][0] for r in results))
        self.store._read_documents.assert_called_once()

    def test_initial_failure_has_no_seed_fallback_or_secret_exception(self):
        self.store._read_root.side_effect = ValueError('SECRET SDK PATH')
        with self.assertRaises(HTTPException) as caught:
            self.store.acquire()
        self.assertEqual(caught.exception.status_code, 503)
        self.assertNotIn('SECRET', caught.exception.detail)
        for _ in range(3):
            with self.assertRaises(HTTPException):
                self.store.acquire(force=True)
        self.store._read_root.assert_called_once()

    def test_stale_snapshot_backoff_and_recovery(self):
        first, _ = self.store.acquire()
        self.time = 61
        self.store._read_root.side_effect = RuntimeError('offline')
        repo, status = self.store.acquire()
        self.assertIs(repo, first)
        self.assertEqual(status['state'], 'stale')
        self.store.acquire(force=True)
        self.assertEqual(self.store._read_root.call_count, 3)
        self.time = 122
        self.store._read_root.side_effect = None
        self.assertEqual(self.store.acquire()[1]['state'], 'ready')
        self.store._read_documents.assert_called_once()

    def test_incompatible_release_never_mixes_local_runtime(self):
        self.store._read_root.return_value = SimpleNamespace(update_time='release-2', to_dict=lambda: {'dataVersion': 'new', 'schemaVersion': 3})
        with self.assertRaises(HTTPException):
            self.store.acquire()
        self.store._read_documents.assert_not_called()

    def test_partial_or_modified_documents_rejected(self):
        self.store._read_documents.side_effect = lambda: {}
        with self.assertRaises(HTTPException):
            self.store.acquire()
        self.assertEqual(self.store.status()['documents'], 0)

    def test_deployment_changed_during_read_rejected(self):
        other = SimpleNamespace(update_time='release-2', to_dict=self.root.to_dict)
        self.store._read_root.side_effect = [self.root, other]
        with self.assertRaises(HTTPException):
            self.store.acquire()

    def client(self):
        service = FakeService()
        client = TestClient(create_api(False, service, FakeCopilot(), self.store))
        client.cookies.update({'manufacturing_session': 'a', 'manufacturing_csrf': 'x'})
        client.headers.update({'Origin': 'http://testserver', 'X-CSRF-Token': 'x'})
        return client, service

    def test_authentication_before_database_read(self):
        client = TestClient(create_api(False, FakeService(), FakeCopilot(), self.store))
        self.assertEqual(client.get('/api/meta').status_code, 401)
        self.assertEqual(client.get('/api/health').status_code, 200)
        self.assertEqual(client.get('/api/health').json()['analysisSource']['state'], 'pending')
        self.store._read_root.assert_not_called()

    def test_api_provenance_pinning_and_version_guard(self):
        client, service = self.client()
        response = client.get('/api/demand')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['X-Analysis-Source'], 'firestore')
        self.assertEqual(client.get('/api/meta').json()['analysisSource']['state'], 'ready')
        body = {'track': 'quality', 'target': 'Test07_NG_dchg', 'decision': 'retest', 'requestId': 'test-1234567890123', 'expectedRevision': 0}
        self.assertEqual(client.post('/api/records', json=body).status_code, 409)
        self.assertEqual(client.post('/api/records', json={**body, 'dataVersion': 'old'}).status_code, 409)
        self.assertEqual(service.saved, {})
        self.assertEqual(client.post('/api/records', json={**body, 'dataVersion': self.contract.manifest['dataVersion']}).status_code, 200)

    def test_stale_allows_read_but_blocks_write_and_keeps_history_accessible(self):
        client, service = self.client()
        client.get('/api/meta')
        self.store._read_root.side_effect = RuntimeError('offline')
        self.time = 61
        response = client.get('/api/demand')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['X-Analysis-State'], 'stale')
        self.assertEqual(client.post('/api/analysis/refresh').json()['analysisSource']['state'], 'stale')
        body = {'track': 'quality', 'target': 'Test07_NG_dchg', 'decision': 'retest', 'requestId': 'test-1234567890123', 'expectedRevision': 0, 'dataVersion': self.contract.manifest['dataVersion']}
        self.assertEqual(client.post('/api/records', json=body).status_code, 503)
        self.assertEqual(client.get('/api/records').status_code, 200)
        self.assertEqual(service.saved, {})

    def test_old_copilot_draft_rejected(self):
        client, service = self.client()
        thread = 'e' * 32
        service.chats[('a', thread)] = {'messages': [{'role': 'assistant', 'status': 'answered', 'context': {'dataVersion': 'old'}}]}
        self.assertEqual(client.post('/api/copilot/draft', json={'threadId': thread}).status_code, 409)


if __name__ == '__main__':
    unittest.main()
