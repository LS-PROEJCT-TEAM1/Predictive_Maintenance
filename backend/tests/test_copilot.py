import unittest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
from backend.copilot import Copilot
from backend.data import Repository
from backend.main import create_api
from backend.tests.fakes import FakeService, FakeCopilot
from fastapi.testclient import TestClient
from google.api_core.exceptions import ResourceExhausted


class CopilotTests(unittest.TestCase):
    def setUp(self):
        self.copilot = Copilot(Repository())
        self.copilot.config['gemini_key'] = 'fake-test-key'

    def response(self, value, status=200):
        response = MagicMock(status_code=status)
        response.json.return_value = value
        return response

    def test_arbitrary_sources_and_bad_context_rejected(self):
        with self.assertRaises(HTTPException): self.copilot.source('../settings')
        with self.assertRaises(ValueError): self.copilot.context({'track': 'quality', 'test': 'fake'})

    def test_citations_are_allowlisted_and_missing_grounding_rejected(self):
        response = self.response({'candidates': [{'content': {'parts': [{'text': '{"answer":"invented","citations":["NOT_REAL"]}'}]}}]})
        with patch.object(self.copilot, 'retrieve', return_value=[]), patch('backend.copilot.httpx.post', return_value=response):
            answer = self.copilot.answer('question', {'track': 'overview'}, [])
        self.assertNotEqual(answer['text'], 'invented')
        self.assertEqual(answer['citations'], [])

    def test_gemini_quota_is_not_retried_or_paid_fallback(self):
        with patch.object(self.copilot, 'retrieve', return_value=[]), patch('backend.copilot.httpx.post', return_value=self.response({}, 429)) as generate:
            with self.assertRaises(HTTPException) as error:
                self.copilot.answer('question', {'track': 'overview'}, [])
        self.assertEqual(error.exception.status_code, 429)
        self.assertEqual(generate.call_count, 1)

    def test_firestore_outage_prevents_gemini_spend(self):
        service, copilot = FakeService(), FakeCopilot()
        service.list_conversations = MagicMock(side_effect=ResourceExhausted('quota'))
        copilot.answer = MagicMock()
        client = TestClient(create_api(False, service, copilot))
        client.cookies.update({'manufacturing_session': 'a', 'manufacturing_csrf': 'x'})
        response = client.post('/api/copilot/ask', json={'question': 'test'}, headers={'Origin': 'http://testserver', 'X-CSRF-Token': 'x'})
        self.assertEqual(response.status_code, 503)
        copilot.answer.assert_not_called()

    def test_draft_requires_owner_and_does_not_save(self):
        service = FakeService()
        thread = 'a'*32
        service.chats[('a', thread)] = {'messages': [{'role': 'assistant', 'status': 'answered', 'text': '검토 근거', 'context': {'track': 'quality', 'testId': 'Test07_NG_dchg'}}]}
        client = TestClient(create_api(False, service, FakeCopilot()))
        client.cookies.update({'manufacturing_session': 'a', 'manufacturing_csrf': 'x'})
        client.headers.update({'Origin': 'http://testserver', 'X-CSRF-Token': 'x'})
        response = client.post('/api/copilot/draft', json={'threadId': thread})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['decision'], 'retest')
        self.assertEqual(service.saved, {})
        client.cookies.set('manufacturing_session', 'b')
        self.assertEqual(client.post('/api/copilot/draft', json={'threadId': thread}).status_code, 404)


if __name__ == '__main__': unittest.main()
