import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.auth_log import AuthLogger, LogKeyError, NullAuthLogger, PiiCipher, build_auth_logger, _init_keys
from backend.main import create_api
from backend.tests.fakes import FakeCopilot, FakeService

PASSWORD = 'S3cret-Pa55!'


def cipher(key_id='k1'):
    return PiiCipher(b'\x01' * 32, b'\x02' * 32, key_id)


class PasswordService(FakeService):
    def login(self, email, password):
        if email.lower() != 'a@example.com':
            raise HTTPException(403, '직원 접근 권한이 없습니다.')
        if password != PASSWORD:
            raise HTTPException(401, '이메일 또는 비밀번호를 확인하세요.')
        return 'a', self.users['a']


class CipherTests(unittest.TestCase):
    def test_roundtrip_and_random_nonce(self):
        c = cipher()
        first, second = c.encrypt('log1', 'email', 'a@example.com'), c.encrypt('log1', 'email', 'a@example.com')
        self.assertNotEqual(first['ct'], second['ct'])
        self.assertEqual(c.decrypt('log1', 'email', first), 'a@example.com')
        self.assertNotIn('example', json.dumps(first))

    def test_ciphertext_is_bound_to_document_and_field(self):
        c = cipher()
        box = c.encrypt('log1', 'email', 'a@example.com')
        for log_id, field in (('log2', 'email'), ('log1', 'ip')):
            with self.assertRaises(LogKeyError):
                c.decrypt(log_id, field, box)
        tampered = {**box, 'ct': base64.b64encode(b'x' + base64.b64decode(box['ct'])[1:]).decode()}
        with self.assertRaises(LogKeyError):
            c.decrypt('log1', 'email', tampered)
        with self.assertRaises(LogKeyError):
            PiiCipher(b'\x09' * 32, b'\x02' * 32).decrypt('log1', 'email', box)

    def test_index_is_normalized_and_keyed(self):
        c = cipher()
        self.assertEqual(c.index(' A@Example.com '), c.index('a@example.com'))
        self.assertNotEqual(c.index('a@example.com'), PiiCipher(b'\x01' * 32, b'\x03' * 32).index('a@example.com'))

    def test_config_validation(self):
        key = base64.b64encode(os.urandom(32)).decode()
        self.assertIsNone(PiiCipher.from_config({}))
        self.assertIsInstance(PiiCipher.from_config({'log_enc_key': key, 'log_hmac_key': base64.b64encode(os.urandom(32)).decode()}), PiiCipher)
        for bad in ({'log_enc_key': key}, {'log_enc_key': key, 'log_hmac_key': key},
                    {'log_enc_key': 'short', 'log_hmac_key': key},
                    {'log_enc_key': base64.b64encode(b'1' * 16).decode(), 'log_hmac_key': key}):
            with self.assertRaises(LogKeyError):
                PiiCipher.from_config(bad)

    def test_non_firebase_services_do_not_log(self):
        self.assertIsInstance(build_auth_logger(FakeService(), {}), NullAuthLogger)

    def test_init_keys_appends_once_without_printing_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / '.env'
            env.write_text('BATTERYFLOW_MODE=auto\nLOG_ENC_KEY=\n', encoding='utf-8')
            with mock.patch('builtins.print') as printed:
                _init_keys(env)
            from dotenv import dotenv_values
            values = dotenv_values(env)
            self.assertEqual(values['BATTERYFLOW_MODE'], 'auto')
            config = {'log_enc_key': values['LOG_ENC_KEY'], 'log_hmac_key': values['LOG_HMAC_KEY']}
            self.assertIsInstance(PiiCipher.from_config(config), PiiCipher)
            self.assertNotIn(values['LOG_ENC_KEY'], str(printed.call_args_list))
            with self.assertRaises(SystemExit):
                _init_keys(env)


class LoginAuditTests(unittest.TestCase):
    def setUp(self):
        self.saved = []
        self.cipher = cipher()
        self.logger = AuthLogger(lambda log_id, doc: self.saved.append(doc), self.cipher)
        self.client = TestClient(create_api(False, PasswordService(), FakeCopilot(), auth_log=self.logger))
        self.client.get('/login')
        self.client.headers.update({'Origin': 'http://testserver', 'X-CSRF-Token': self.client.cookies.get('manufacturing_csrf')})

    def events(self):
        self.assertTrue(self.logger.flush())
        return self.saved

    def plain(self, doc):
        return self.cipher.decrypt_document(doc)

    def test_success_logout_and_no_password_anywhere(self):
        self.assertEqual(self.client.post('/auth/login', json={'email': 'A@example.com', 'password': PASSWORD}).status_code, 200)
        self.client.headers['X-CSRF-Token'] = self.client.cookies.get('manufacturing_csrf')  # rotated at login
        self.assertEqual(self.client.post('/auth/logout').status_code, 200)
        success, logout = self.events()
        self.assertEqual((success['event'], logout['event']), ('login_success', 'logout'))
        self.assertEqual((success['uid'], success['role'], logout['uid']), ('a', 'employee', 'a'))
        self.assertEqual(success['sessionId'], logout['sessionId'])
        self.assertEqual(self.plain(success)['email'], 'a@example.com')
        self.assertEqual(self.plain(success)['name'], 'Alice')
        self.assertEqual(success['emailHmac'], self.cipher.index('a@example.com'))
        self.assertGreater(success['expireAt'], success['at'])
        raw = json.dumps(self.saved, default=str)
        for secret in (PASSWORD, 'a@example.com', 'Alice', 'testclient'):
            self.assertNotIn(secret, raw)
        self.assertNotIn('password', raw.lower())

    def test_failures_are_classified(self):
        self.assertEqual(self.client.post('/auth/login', json={'email': 'a@example.com', 'password': 'wrong'}).status_code, 401)
        self.assertEqual(self.client.post('/auth/login', json={'email': 'x@example.com', 'password': PASSWORD}).status_code, 403)
        wrong, denied = self.events()
        self.assertEqual((wrong['event'], wrong['reason'], wrong['uid']), ('login_failed', 'invalid_credentials', None))
        self.assertEqual(denied['reason'], 'no_permission')
        self.assertEqual(self.plain(denied)['email'], 'x@example.com')
        self.assertNotIn('wrong', json.dumps(self.saved, default=str))

    def test_rate_limit_is_logged(self):
        for _ in range(11):
            response = self.client.post('/auth/login', json={'email': 'a@example.com', 'password': 'wrong'})
        self.assertEqual(response.status_code, 429)
        self.assertEqual(self.client.post('/auth/login', json={'email': 'a@example.com', 'password': 'wrong'}).status_code, 429)
        reasons = [e['reason'] for e in self.events()]
        self.assertEqual(reasons.count('rate_limited'), 1)
        self.assertEqual(reasons.count('invalid_credentials'), 10)

    def test_expired_session_logged_once(self):
        self.client.cookies.set('manufacturing_session', 'expired-cookie', domain='testserver.local', path='/')
        self.assertEqual(self.client.get('/api/me').status_code, 401)
        self.assertEqual(self.client.get('/api/me').status_code, 401)
        self.client.cookies.clear()
        self.assertEqual(self.client.get('/api/me').status_code, 401)
        events = self.events()
        self.assertEqual([e['event'] for e in events], ['session_expired'])
        self.assertIsNone(events[0]['uid'])

    def test_sink_failure_never_blocks_login(self):
        def broken(log_id, doc):
            raise RuntimeError('firestore down')
        client = TestClient(create_api(False, PasswordService(), FakeCopilot(), auth_log=AuthLogger(broken, self.cipher)))
        client.get('/login')
        response = client.post('/auth/login', json={'email': 'a@example.com', 'password': PASSWORD},
                               headers={'Origin': 'http://testserver', 'X-CSRF-Token': client.cookies.get('manufacturing_csrf')})
        self.assertEqual(response.status_code, 200)


class AdminLogViewTests(unittest.TestCase):
    def setUp(self):
        self.cipher = cipher()
        self.store = []
        source = lambda limit, event=None: [d for d in reversed(self.store) if not event or d['event'] == event][:limit]
        self.logger = AuthLogger(lambda log_id, doc: self.store.append(doc), self.cipher, source=source)
        self.client = TestClient(create_api(False, PasswordService(), FakeCopilot(), auth_log=self.logger))
        self.logger.record('login_success', email='a@example.com', user={'uid': 'a', 'name': 'Alice', 'role': 'employee'}, ip='10.0.0.1', cookie='c1')
        self.logger.record('login_failed', email='x@example.com', reason='invalid_credentials', ip='10.0.0.2')
        self.logger.flush()

    def as_user(self, uid):
        self.client.cookies.set('manufacturing_session', uid, domain='testserver.local', path='/')

    def test_employee_and_anonymous_are_denied(self):
        self.assertEqual(self.client.get('/api/admin/auth-logs').status_code, 401)
        self.as_user('a')
        self.assertEqual(self.client.get('/api/admin/auth-logs').status_code, 403)

    def test_admin_sees_decrypted_rows_newest_first(self):
        self.as_user('b')
        rows = self.client.get('/api/admin/auth-logs').json()['logs']
        self.assertEqual([r['event'] for r in rows], ['login_failed', 'login_success'])
        self.assertEqual((rows[0]['email'], rows[0]['reason'], rows[0]['eventLabel']), ('x@example.com', '이메일·비밀번호 오류', '로그인 실패'))
        self.assertEqual((rows[1]['name'], rows[1]['role'], rows[1]['ip']), ('Alice', '직원', '10.0.0.1'))
        only = self.client.get('/api/admin/auth-logs', params={'event': 'login_success'}).json()['logs']
        self.assertEqual([r['event'] for r in only], ['login_success'])
        self.assertEqual(self.client.get('/api/admin/auth-logs', params={'event': 'bogus'}).status_code, 422)

    def test_rows_with_another_key_are_marked_not_dropped(self):
        other = AuthLogger(lambda log_id, doc: self.store.append(doc), cipher('k2'))
        other.record('logout', user={'uid': 'a', 'name': 'Alice', 'role': 'employee'}, cookie='c1')
        other.flush()
        self.as_user('b')
        rows = self.client.get('/api/admin/auth-logs').json()['logs']
        self.assertEqual(rows[0]['email'], '복호화 불가(키 불일치)')
        self.assertEqual(len(rows), 3)

    def test_disabled_logging_reports_clearly(self):
        client = TestClient(create_api(False, PasswordService(), FakeCopilot()))
        client.cookies.set('manufacturing_session', 'b', domain='testserver.local', path='/')
        self.assertEqual(client.get('/api/admin/auth-logs').status_code, 503)


if __name__ == '__main__':
    unittest.main()
