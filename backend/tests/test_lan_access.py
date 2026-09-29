import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import create_api
from backend.settings import _hosts
from backend.tests.fakes import FakeCopilot, FakeService


class AllowedHostsTests(unittest.TestCase):
    def test_parsing_accepts_ips_and_names_only(self):
        self.assertEqual(_hosts(''), [])
        self.assertEqual(_hosts(' 10.101.134.171, pc-01.local ;192.168.0.5'), ['10.101.134.171', 'pc-01.local', '192.168.0.5'])
        for bad in ('*', 'http://10.0.0.1', '10.0.0.1:8070', '*.example.com'):
            with self.assertRaises(ValueError):
                _hosts(bad)

    def client(self, hosts):
        with patch.dict(os.environ, {'ALLOWED_HOSTS': hosts}):
            return TestClient(create_api(False, FakeService(), FakeCopilot()))

    def test_lan_ip_is_rejected_until_listed(self):
        closed = self.client('')
        self.assertEqual(closed.get('/api/health', headers={'Host': '10.101.134.171:8070'}).status_code, 400)
        opened = self.client('10.101.134.171')
        self.assertEqual(opened.get('/api/health', headers={'Host': '10.101.134.171:8070'}).status_code, 200)
        self.assertEqual(opened.get('/api/health', headers={'Host': 'evil.example:8070'}).status_code, 400)
        self.assertEqual(opened.get('/api/health').status_code, 200)

    def test_login_works_from_lan_origin(self):
        client = self.client('10.101.134.171')
        base = {'Host': '10.101.134.171:8070'}
        client.get('/login', headers=base)
        token = client.cookies.get('manufacturing_csrf')
        response = client.post('/auth/login', json={'email': 'a@example.com', 'password': 'x'},
                               headers={**base, 'Origin': 'http://10.101.134.171:8070', 'X-CSRF-Token': token})
        self.assertEqual(response.status_code, 200)


if __name__ == '__main__':
    unittest.main()
