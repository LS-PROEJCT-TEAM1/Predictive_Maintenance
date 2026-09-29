"""Login audit log for Firestore.

- Passwords are never stored in any form (not even hashed). Firebase Auth owns them.
- Personal data (email, name, IP, User-Agent) is encrypted with AES-256-GCM.
- Deterministic HMAC-SHA256 indexes allow lookups such as "failures for this email"
  without storing the plaintext.
- Writes run on a background thread so a logging failure never blocks sign-in.

CLI (run from the project root):
    .venv\\Scripts\\python.exe -m backend.auth_log init-keys
    .venv\\Scripts\\python.exe -m backend.auth_log show --limit 20
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import logging
import os
import queue
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

SCHEMA_VERSION = 1
COLLECTION = 'manufacturingAuthLogs'
EVENTS = ('login_success', 'login_failed', 'logout', 'session_expired')
FAILURE_REASONS = {401: 'invalid_credentials', 403: 'no_permission', 429: 'rate_limited', 503: 'auth_unavailable'}
PII_FIELDS = ('email', 'name', 'ip', 'userAgent')
DEFAULT_RETENTION_DAYS = 180
SESSION_DEDUPE_SECONDS = 600

log = logging.getLogger('batteryflow.auth_log')


class LogKeyError(ValueError):
    pass


def _decode_key(name, value):
    try:
        raw = base64.b64decode(value.strip(), validate=True)
    except (binascii.Error, ValueError):
        raise LogKeyError(f'{name}는 base64 형식이어야 합니다. init-keys로 다시 만드세요.') from None
    if len(raw) != 32:
        raise LogKeyError(f'{name}는 32바이트(AES-256) 키여야 합니다. init-keys로 다시 만드세요.')
    return raw


def _b64(data):
    return base64.b64encode(data).decode('ascii')


def session_id(cookie):
    """Short, non-reversible session reference. The cookie itself is never stored."""
    return hashlib.sha256(cookie.encode()).hexdigest()[:16] if cookie else None


class PiiCipher:
    """AES-256-GCM field encryption bound to (schema, keyId, logId, field) via AAD."""

    def __init__(self, enc_key: bytes, hmac_key: bytes, key_id: str = 'k1'):
        if len(enc_key) != 32 or len(hmac_key) != 32:
            raise LogKeyError('로그 암호화 키는 32바이트여야 합니다.')
        if enc_key == hmac_key:
            raise LogKeyError('LOG_ENC_KEY와 LOG_HMAC_KEY는 서로 달라야 합니다.')
        self._aead = AESGCM(enc_key)
        self._hmac_key = hmac_key
        self.key_id = key_id

    @classmethod
    def from_config(cls, config):
        enc, mac = config.get('log_enc_key', ''), config.get('log_hmac_key', '')
        if not enc and not mac:
            return None
        if not enc or not mac:
            raise LogKeyError('LOG_ENC_KEY와 LOG_HMAC_KEY를 함께 설정하세요.')
        key_id = config.get('log_key_id') or 'k1'
        return cls(_decode_key('LOG_ENC_KEY', enc), _decode_key('LOG_HMAC_KEY', mac), key_id)

    def _aad(self, log_id, field):
        return f'{SCHEMA_VERSION}|{self.key_id}|{log_id}|{field}'.encode()

    def encrypt(self, log_id, field, value):
        nonce = os.urandom(12)
        ct = self._aead.encrypt(nonce, str(value).encode('utf-8'), self._aad(log_id, field))
        return {'nonce': _b64(nonce), 'ct': _b64(ct)}

    def decrypt(self, log_id, field, box):
        try:
            data = self._aead.decrypt(base64.b64decode(box['nonce']), base64.b64decode(box['ct']),
                                      self._aad(log_id, field))
        except (InvalidTag, KeyError, binascii.Error, ValueError):
            raise LogKeyError('복호화 실패: 키가 다르거나 기록이 변조되었습니다.') from None
        return data.decode('utf-8')

    def index(self, value):
        normalized = str(value).strip().lower().encode('utf-8')
        return hmac.new(self._hmac_key, normalized, hashlib.sha256).hexdigest()

    def decrypt_document(self, doc):
        """Return a copy with the pii block decrypted (for admin tooling only)."""
        if doc.get('keyId') != self.key_id:
            raise LogKeyError(f"이 기록은 키 {doc.get('keyId')}로 암호화되었습니다. 현재 키: {self.key_id}")
        plain = {field: self.decrypt(doc['id'], field, box) for field, box in (doc.get('pii') or {}).items()}
        return {**{k: v for k, v in doc.items() if k != 'pii'}, **plain}


KST = timezone(timedelta(hours=9))
EVENT_LABELS = {'login_success': '로그인', 'login_failed': '로그인 실패', 'logout': '로그아웃', 'session_expired': '세션 만료'}
REASON_LABELS = {'invalid_credentials': '이메일·비밀번호 오류', 'no_permission': '앱 권한 없음', 'rate_limited': '시도 횟수 초과',
                 'auth_unavailable': '인증 서비스 연결 실패', 'expired_or_revoked': '만료·권한 변경', 'error': '기타 오류'}


def display_row(cipher, doc):
    """Decrypted, display-ready row for the admin screen. Unreadable rows are marked, never dropped."""
    try:
        plain = cipher.decrypt_document(doc)
        readable = True
    except LogKeyError:
        plain, readable = {k: v for k, v in doc.items() if k != 'pii'}, False
    at = plain.get('at')
    return {
        'id': plain.get('id'),
        'at': at.astimezone(KST).strftime('%Y-%m-%d %H:%M:%S') if hasattr(at, 'astimezone') else str(at or ''),
        'event': plain.get('event'),
        'eventLabel': EVENT_LABELS.get(plain.get('event'), plain.get('event')),
        'reason': REASON_LABELS.get(plain.get('reason'), plain.get('reason') or ''),
        'email': plain.get('email') or ('' if readable else '복호화 불가(키 불일치)'),
        'name': plain.get('name') or '',
        'uid': plain.get('uid') or '',
        'role': {'admin': '관리자', 'employee': '직원'}.get(plain.get('role'), ''),
        'ip': plain.get('ip') or '',
        'userAgent': plain.get('userAgent') or '',
        'sessionId': plain.get('sessionId') or '',
    }


class NullAuthLogger:
    enabled = False

    def record(self, *args, **kwargs):
        return None

    def recent(self, limit=100, event=None):
        from fastapi import HTTPException
        raise HTTPException(503, '로그인 기록이 비활성화되어 있습니다. 연결 모드와 .env의 LOG_ENC_KEY/LOG_HMAC_KEY를 확인하세요.')

    def flush(self, timeout=5.0):
        return True


class AuthLogger:
    enabled = True

    def __init__(self, sink, cipher: PiiCipher, retention_days=DEFAULT_RETENTION_DAYS, max_queue=1000, source=None):
        self._sink = sink
        self._source = source
        self.cipher = cipher
        self.retention = timedelta(days=int(retention_days))
        self._queue = queue.Queue(maxsize=max_queue)
        self._recent_sessions = {}
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, name='auth-log-writer', daemon=True)
        self._worker.start()

    def build(self, event, *, ip=None, user_agent=None, user=None, email=None, reason=None, cookie=None):
        if event not in EVENTS:
            raise ValueError('unknown auth log event')
        log_id = secrets.token_hex(16)
        at = datetime.now(timezone.utc)
        # Logout carries no typed email, so fall back to the verified account email.
        pii_values = {'email': (email or (user or {}).get('email') or '').strip().lower() or None, 'name': (user or {}).get('name'),
                      'ip': ip, 'userAgent': (user_agent or '')[:300] or None}
        doc = {
            'schemaVersion': SCHEMA_VERSION,
            'id': log_id,
            'event': event,
            'at': at,
            'expireAt': at + self.retention,
            'uid': (user or {}).get('uid'),
            'role': (user or {}).get('role'),
            'reason': reason,
            'sessionId': session_id(cookie),
            'emailHmac': self.cipher.index(pii_values['email']) if pii_values['email'] else None,
            'ipHmac': self.cipher.index(ip) if ip else None,
            'keyId': self.cipher.key_id,
            'pii': {field: self.cipher.encrypt(log_id, field, value)
                    for field, value in pii_values.items() if value},
        }
        return doc

    def record(self, event, request=None, **fields):
        try:
            if request is not None:
                fields.setdefault('ip', request.client.host if request.client else None)
                fields.setdefault('user_agent', request.headers.get('user-agent'))
            if event == 'session_expired' and not self._first_expiry(fields.get('cookie')):
                return None
            doc = self.build(event, **fields)
            self._queue.put_nowait(doc)
            return doc
        except queue.Full:
            log.warning('로그인 기록 대기열이 가득 차 1건을 건너뛰었습니다.')
        except Exception:
            log.warning('로그인 기록을 만들지 못했습니다.', exc_info=False)
        return None

    def _first_expiry(self, cookie):
        key = session_id(cookie)
        if not key:
            return False
        now = time.monotonic()
        with self._lock:
            self._recent_sessions = {k: v for k, v in self._recent_sessions.items() if v > now}
            if key in self._recent_sessions:
                return False
            self._recent_sessions[key] = now + SESSION_DEDUPE_SECONDS
            return True

    def _run(self):
        while True:
            doc = self._queue.get()
            try:
                self._sink(doc['id'], doc)
            except Exception as error:
                # No personal data in this message: only event name and error type.
                log.warning('로그인 기록 저장 실패 (%s): %s', doc.get('event'), type(error).__name__)
            finally:
                self._queue.task_done()

    def recent(self, limit=100, event=None):
        """Newest first, decrypted for display. Callers must enforce admin access."""
        if self._source is None:
            return []
        rows = [display_row(self.cipher, doc) for doc in self._source(max(1, min(int(limit), 500)), event)]
        # Logout/expiry events carry no email; show the one from the same session's login when it is on screen.
        emails = {r['sessionId']: r['email'] for r in rows if r['event'] == 'login_success' and r['sessionId'] and r['email']}
        for r in rows:
            if not r['email'] and r['sessionId'] in emails:
                r['email'] = emails[r['sessionId']]
        return rows

    def flush(self, timeout=5.0):
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.01)
        return True


def firestore_sink(service):
    def write(log_id, doc):
        service.record_collection(COLLECTION).document(log_id).create(doc)
    return write


def firestore_source(service):
    def read(limit, event=None):
        from firebase_admin import firestore
        from google.cloud.firestore_v1.base_query import FieldFilter
        query = service.record_collection(COLLECTION)
        if event:
            # Equality filter + in-memory sort avoids needing a composite index.
            rows = [s.to_dict() for s in query.where(filter=FieldFilter('event', '==', event)).limit(1000).stream()]
            return sorted(rows, key=lambda r: r['at'], reverse=True)[:limit]
        return [s.to_dict() for s in query.order_by('at', direction=firestore.Query.DESCENDING).limit(limit).stream()]
    return read


def build_auth_logger(service, config=None):
    """Connected Firebase mode with keys → AuthLogger. Demo, test doubles or no keys → no-op."""
    from backend.firebase_service import FirebaseService
    if getattr(service, 'mode', None) == 'demo' or not isinstance(service, FirebaseService):
        return NullAuthLogger()
    if config is None:
        from backend.settings import settings
        config = settings()
    cipher = PiiCipher.from_config(config)
    if cipher is None:
        print('로그인 기록 비활성화: .env에 LOG_ENC_KEY/LOG_HMAC_KEY가 없습니다. '
              '"python -m backend.auth_log init-keys"로 만드세요.', flush=True)
        return NullAuthLogger()
    return AuthLogger(firestore_sink(service), cipher, config.get('log_retention_days') or DEFAULT_RETENTION_DAYS,
                      source=firestore_source(service))


# ---------------------------------------------------------------- CLI

def _init_keys(env_path):
    from dotenv import dotenv_values
    existing = dotenv_values(env_path) if env_path.is_file() else {}
    if existing.get('LOG_ENC_KEY') or existing.get('LOG_HMAC_KEY'):
        raise SystemExit('.env에 이미 로그 키가 있습니다. 키를 바꾸면 기존 기록을 복호화할 수 없으니 그대로 두세요.')
    lines = ['', '# Login audit log keys (AES-256-GCM / HMAC-SHA256). Share only with the .env itself.',
             'LOG_KEY_ID=k1', 'LOG_ENC_KEY=' + _b64(os.urandom(32)), 'LOG_HMAC_KEY=' + _b64(os.urandom(32)),
             f'LOG_RETENTION_DAYS={DEFAULT_RETENTION_DAYS}', '']
    prefix = '' if not env_path.is_file() or env_path.read_bytes().endswith(b'\n') else '\n'
    with env_path.open('a', encoding='utf-8', newline='\r\n' if os.name == 'nt' else '\n') as handle:
        handle.write(prefix + '\n'.join(lines))
    print(f'{env_path.name}에 로그 암호화 키를 추가했습니다. 키 값은 화면에 출력하지 않습니다.')
    print('팀원이 같은 로그를 읽으려면 이 .env를 팀원과 같은 방식으로 안전하게 공유하세요.')


def _show(limit, event=None, email=None):
    from backend.firebase_service import FirebaseService
    from backend.settings import settings
    from google.cloud.firestore_v1.base_query import FieldFilter
    config = settings()
    cipher = PiiCipher.from_config(config)
    if cipher is None:
        raise SystemExit('.env에 LOG_ENC_KEY/LOG_HMAC_KEY가 없습니다.')
    service = FirebaseService()
    if email:
        query = service.record_collection(COLLECTION).where(filter=FieldFilter('emailHmac', '==', cipher.index(email)))
        rows = sorted((s.to_dict() for s in query.limit(1000).stream()), key=lambda r: r['at'], reverse=True)[:limit]
    else:
        rows = firestore_source(service)(limit, event)
    for doc in rows:
        try:
            row = cipher.decrypt_document(doc)
        except LogKeyError as error:
            row = {**doc, 'email': f'<{error}>'}
        at = row['at'].astimezone(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{at}  {row['event']:<15} {row.get('reason') or '-':<20} {row.get('email') or '-':<30} "
              f"{row.get('name') or '-':<10} {row.get('uid') or '-':<28} {row.get('ip') or '-'}")
    print(f'{len(rows)}건')


def main(argv=None):
    import argparse
    from backend.settings import ROOT
    parser = argparse.ArgumentParser(description='로그인 기록 키 생성 및 조회 (관리자용)')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('init-keys', help='.env에 LOG_ENC_KEY/LOG_HMAC_KEY를 새로 추가')
    show = sub.add_parser('show', help='최근 로그인 기록을 복호화해 출력')
    show.add_argument('--limit', type=int, default=20)
    show.add_argument('--event', choices=EVENTS)
    show.add_argument('--email', help='이 이메일의 기록만 (HMAC 인덱스로 조회)')
    args = parser.parse_args(argv)
    if args.command == 'init-keys':
        _init_keys(ROOT / '.env')
    else:
        _show(max(1, min(args.limit, 500)), args.event, args.email)


if __name__ == '__main__':
    main()
