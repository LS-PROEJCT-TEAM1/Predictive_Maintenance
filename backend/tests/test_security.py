import unittest
from fastapi.testclient import TestClient
from backend.main import create_api
from backend.tests.fakes import FakeService, FakeCopilot


class SecurityTests(unittest.TestCase):
    def setUp(self):
        self.service = FakeService()
        self.client = TestClient(create_api(False, self.service, FakeCopilot()))

    def login(self, uid='a'):
        self.client.cookies.set('manufacturing_session', uid, domain='testserver.local', path='/')
        self.client.cookies.set('manufacturing_csrf', 'token', domain='testserver.local', path='/')
        self.client.headers.update({'Origin': 'http://testserver', 'X-CSRF-Token': 'token'})

    def test_all_private_surfaces_require_identity(self):
        for path in ['/', '/_dash-layout', '/_dash-dependencies', '/api/meta', '/api/records', '/api/copilot/conversations', '/docs', '/openapi.json']:
            self.assertEqual(self.client.get(path).status_code, 401, path)
        self.assertEqual(self.client.get('/login').status_code, 200)
        self.assertEqual(self.client.get('/api/health').status_code, 200)

    def test_navigation_redirects_to_login(self):
        response = self.client.get('/', headers={'Accept': 'text/html'}, follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers['location'], '/login')

    def test_login_accepts_json_and_sets_httponly_cookie(self):
        self.client.get('/login')
        token = self.client.cookies.get('manufacturing_csrf')
        response = self.client.post('/auth/login', json={'email': 'a@example.com', 'password': 'dummy'},
            headers={'Origin': 'http://testserver', 'X-CSRF-Token': token})
        self.assertEqual(response.status_code, 200)
        self.assertIn('HttpOnly', response.headers['set-cookie'])
        self.assertEqual(self.client.get('/api/me').status_code, 200)

    def test_csrf_and_cross_origin_rejected(self):
        self.login()
        payload = {'name': 'test', 'rows': []}
        self.assertEqual(self.client.post('/api/exports', json=payload, headers={'Origin': 'https://evil.example'}).status_code, 403)
        self.assertEqual(self.client.post('/api/exports', json=payload, headers={'X-CSRF-Token': 'wrong'}).status_code, 403)
        self.assertEqual(self.client.post('/_dash-update-component', json={}, headers={'Origin': 'https://evil.example'}).status_code, 403)

    def test_admin_cannot_read_other_authors_conversation(self):
        self.login()
        result = self.client.post('/api/copilot/ask', json={'question': '설명해 줘'}).json()
        self.assertEqual(len(self.client.get('/api/copilot/conversations').json()['threads']), 1)
        self.login('b')
        self.assertEqual(self.client.get('/api/copilot/conversations').json()['threads'], [])
        self.assertEqual(self.client.get('/api/copilot/conversations/'+result['id']).status_code, 404)

    def test_export_is_bound_to_author(self):
        self.login()
        url = self.client.post('/api/exports', json={'name': 'test', 'rows': [{'a': 1}]}).json()['url']
        self.login('b')
        self.assertEqual(self.client.get(url).status_code, 404)

    def test_save_actor_is_server_identity_and_revision_conflicts(self):
        self.login()
        body = {'track': 'quality', 'target': 'Test07_NG_dchg', 'decision': 'retest', 'requestId': 'x'*32, 'expectedRevision': 0, 'actorUid': 'b'}
        response = self.client.post('/api/records', json=body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['actorUid'], 'a')
        self.assertEqual(self.client.post('/api/records', json={**body, 'requestId': 'y'*32}).status_code, 409)

    def test_logout_clears_authentication_cookies(self):
        self.login()
        response = self.client.post('/auth/logout')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get('/api/me').status_code, 401)


if __name__ == '__main__':
    unittest.main()
