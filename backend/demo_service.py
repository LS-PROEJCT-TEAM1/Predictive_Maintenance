"""Explicit local read-only session. Never contacts Firebase or Gemini."""
import secrets
import time
from threading import Lock
from fastapi import HTTPException


class DemoService:
    mode = 'demo'

    def __init__(self):
        self.sessions = {}
        self.lock = Lock()

    def login(self, email, password):
        if email != 'demo@local.test' or password != 'local-demo':
            raise HTTPException(401, '체험 시작 버튼을 이용하세요. 실제 직원 계정은 연결 모드에서 사용하세요.')
        token = secrets.token_urlsafe(32)
        user = {'uid': 'local-demo', 'name': '로컬 체험', 'role': 'employee'}
        with self.lock:
            now = time.monotonic()
            self.sessions = {k: v for k, v in self.sessions.items() if v[0] > now}
            if len(self.sessions) >= 100:
                raise HTTPException(429, '체험 세션이 많습니다. 로컬 서버를 재시작하세요.')
            self.sessions[token] = (now + 28800, user)
        return token, user.copy()

    def verify(self, cookie, fresh=False):
        with self.lock:
            entry = self.sessions.get(cookie)
            if not entry or entry[0] <= time.monotonic():
                raise HTTPException(401, '로컬 체험을 시작해 주세요.')
            return entry[1].copy()

    def logout(self, cookie):
        with self.lock:
            self.sessions.pop(cookie, None)

    def unavailable(self, *args, **kwargs):
        raise HTTPException(503, '로컬 체험 모드에서는 업무 기록 저장·조회와 Copilot을 사용하지 않습니다. 실제 연결 모드가 필요합니다.')

    states = history = save_record = list_conversations = conversation = save_conversation = unavailable


class DemoCopilot:
    answer = source = DemoService.unavailable
