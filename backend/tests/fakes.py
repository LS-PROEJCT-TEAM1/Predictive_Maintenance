from copy import deepcopy
from fastapi import HTTPException


class FakeService:
    """Explicit injected double only; production factory never selects this class."""
    def __init__(self):
        self.users = {'a': {'uid': 'a', 'name': 'Alice', 'role': 'employee'}, 'b': {'uid': 'b', 'name': 'Bob', 'role': 'admin'}}
        self.saved = {}
        self.chats = {}

    def verify(self, cookie, fresh=False):
        if cookie not in self.users:
            raise HTTPException(401, '로그인 필요')
        return self.users[cookie]

    def states(self):
        return deepcopy(self.saved)

    def history(self, key=None):
        return list(self.saved.values())

    def save_record(self, body, key, user, version):
        old = self.saved.get(key, {})
        if old.get('revision', 0) != body['expectedRevision']:
            raise HTTPException(409, 'conflict')
        result = {**body, 'key': key, 'revision': body['expectedRevision']+1, 'actorUid': user['uid'], 'dataVersion': version}
        self.saved[key] = result
        return result

    def list_conversations(self, uid):
        return [v for (owner, _), v in self.chats.items() if owner == uid]

    def conversation(self, uid, thread_id):
        result = self.chats.get((uid, thread_id))
        if result is None:
            raise HTTPException(404, 'not found')
        return result

    def save_conversation(self, uid, thread_id, revision, question, result):
        row = {'id': thread_id, 'revision': revision+1, 'messages': [{'role': 'assistant', **result}], 'title': question}
        self.chats[(uid, thread_id)] = row
        return row

    def logout(self, cookie):
        pass

    def login(self, email, password):
        return 'a', self.users['a']


class FakeCopilot:
    def answer(self, question, context, history):
        return {'text': '근거 확인', 'citations': [], 'status': 'answered'}
