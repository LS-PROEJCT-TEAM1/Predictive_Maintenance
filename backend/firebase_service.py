"""Server-only identity and durable records. Never writes official seed collections."""
from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone

import firebase_admin
import httpx
from firebase_admin import auth, credentials, firestore
from fastapi import HTTPException
from google.cloud.firestore_v1.base_query import FieldFilter

from backend.settings import settings


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class FirebaseService:
    def __init__(self):
        self.config = settings()
        self._app = None
        self._lock = threading.RLock()
        self._cache = {}

    @property
    def app(self):
        with self._lock:
            if self._app is None:
                path = self.config.get('service_account_file')
                if not path:
                    raise HTTPException(503, 'Firebase 연결 설정이 필요합니다.')
                self._app = firebase_admin.initialize_app(credentials.Certificate(path),
                    {'projectId': self.config['project_id']}, name='manufacturing-'+secrets.token_hex(6))
            return self._app

    @property
    def db(self):
        return firestore.client(self.app)

    def employee(self, uid):
        # Current Auth record, not stale claims embedded in an eight-hour cookie.
        account = auth.get_user(uid, app=self.app)
        claims = account.custom_claims or {}
        if account.disabled or claims.get('manufacturingRole') not in ('admin', 'employee'):
            raise HTTPException(403, '직원 접근 권한이 없습니다. 관리자에게 문의하세요.')
        return {'uid': uid, 'name': account.display_name or '직원', 'role': claims['manufacturingRole']}

    def login(self, email, password):
        try:
            response = httpx.post('https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword',
                params={'key': self.config.get('firebase_web_key', '')},
                json={'email': email, 'password': password, 'returnSecureToken': True}, timeout=20)
            if response.status_code != 200:
                raise HTTPException(401, '이메일 또는 비밀번호를 확인하세요.')
            token = response.json()['idToken']
            decoded = auth.verify_id_token(token, app=self.app, check_revoked=True, clock_skew_seconds=60)
            user = self.employee(decoded['uid'])
            cookie = auth.create_session_cookie(token, expires_in=timedelta(hours=8), app=self.app)
            return cookie, user
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(503, '인증 서비스에 연결하지 못했습니다. 잠시 후 다시 시도하세요.') from None

    def verify(self, cookie, fresh=False):
        if not cookie:
            raise HTTPException(401, '로그인이 필요합니다.')
        key = digest(cookie)
        with self._lock:
            cached = self._cache.get(key)
        if not fresh and cached and cached[0] > time.monotonic():
            return cached[1]
        try:
            decoded = auth.verify_session_cookie(cookie, check_revoked=True, app=self.app, clock_skew_seconds=60)
            user = self.employee(decoded['uid'])
        except HTTPException:
            raise
        except (auth.InvalidSessionCookieError, auth.ExpiredSessionCookieError,
                auth.RevokedSessionCookieError, auth.UserDisabledError, auth.UserNotFoundError, ValueError):
            raise HTTPException(401, '로그인이 만료되었거나 계정이 변경되었습니다. 다시 로그인하세요.') from None
        except Exception:
            raise HTTPException(503, '직원 인증을 확인할 수 없습니다.') from None
        with self._lock:
            if len(self._cache) > 1000:
                self._cache.clear()
            self._cache[key] = (time.monotonic()+10, user)
        return user

    def logout(self, cookie):
        with self._lock:
            self._cache.pop(digest(cookie or ''), None)

    def states(self):
        return {d['key']: d for snap in self.db.collection('manufacturingReviewState').stream()
                if (d := snap.to_dict())}

    def history(self, key=None, limit=100):
        query = self.db.collection('manufacturingRecords')
        if key:
            query = query.where(filter=FieldFilter('key', '==', key))
            rows = [s.to_dict() for s in query.limit(500).stream()]
            return sorted(rows, key=lambda r: r['at'], reverse=True)[:limit]
        return [s.to_dict() for s in query.order_by('at', direction=firestore.Query.DESCENDING).limit(limit).stream()]

    def save_record(self, body, key, user, version):
        record_id = digest(user['uid']+':'+body['requestId'])
        record_ref = self.db.collection('manufacturingRecords').document(record_id)
        state_ref = self.db.collection('manufacturingReviewState').document(digest(key))
        fingerprint = digest(json.dumps(body, sort_keys=True, ensure_ascii=False))
        @firestore.transactional
        def save(transaction):
            existing = record_ref.get(transaction=transaction).to_dict()
            old = state_ref.get(transaction=transaction).to_dict() or {}
            if existing:
                if existing['fingerprint'] != fingerprint:
                    raise HTTPException(409, '같은 요청 번호의 내용이 달라졌습니다. 다시 확인하세요.')
                return existing
            if old.get('revision', 0) != body['expectedRevision']:
                raise HTTPException(409, '다른 직원이 먼저 기록했습니다. 새로고침 후 다시 확인하세요.')
            record = {**body, 'id': record_id, 'key': key, 'at': now(), 'actor': user['name'],
                'actorUid': user['uid'], 'role': user['role'], 'dataVersion': version,
                'revision': old.get('revision', 0)+1, 'previousDecision': old.get('decision'),
                'persistence': 'firestore', 'fingerprint': fingerprint}
            transaction.create(record_ref, record)
            transaction.set(state_ref, record)
            return record
        return save(self.db.transaction())

    def conversations(self, uid):
        # UID is taken from verified identity, never request parameters.
        return self.db.collection('manufacturingConversations').document(uid).collection('threads')

    def list_conversations(self, uid):
        return [{k: d.get(k) for k in ('id', 'title', 'updatedAt', 'revision')}
                for s in self.conversations(uid).order_by('updatedAt', direction=firestore.Query.DESCENDING).limit(30).stream()
                if (d := s.to_dict())]

    def conversation(self, uid, thread_id):
        doc = self.conversations(uid).document(thread_id).get().to_dict()
        if not doc:
            raise HTTPException(404, '대화를 찾을 수 없습니다.')
        return doc

    def save_conversation(self, uid, thread_id, revision, question, result):
        ref = self.conversations(uid).document(thread_id)
        @firestore.transactional
        def save(transaction):
            old = ref.get(transaction=transaction).to_dict() or {'messages': [], 'revision': 0}
            if old['revision'] != revision:
                raise HTTPException(409, '대화가 변경되었습니다. 이력을 다시 열어 주세요.')
            messages = old['messages'] + [{'role': 'user', 'text': question, 'at': now()},
                {'role': 'assistant', **result, 'at': now()}]
            if len(messages) > 40:
                raise HTTPException(422, '대화당 20회까지 질문할 수 있습니다. 새 대화를 시작하세요.')
            doc = {'id': thread_id, 'ownerUid': uid, 'title': old.get('title', question[:60]),
                'updatedAt': now(), 'revision': revision+1, 'messages': messages}
            transaction.set(ref, doc)
            return doc
        return save(self.db.transaction())
