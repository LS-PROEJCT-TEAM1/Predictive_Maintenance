"""Exercise production storage logic with an in-memory Firestore transaction double."""
import unittest
from copy import deepcopy
from unittest.mock import patch
from fastapi import HTTPException
from backend.firebase_service import FirebaseService


class Snapshot:
    def __init__(self, value): self.value = value
    def to_dict(self): return deepcopy(self.value)


class Reference:
    def __init__(self, db, path=''): self.db, self.path = db, path
    def collection(self, name): return Reference(self.db, self.path+'/'+name)
    def document(self, name): return Reference(self.db, self.path+'/'+name)
    def get(self, **kwargs): return Snapshot(self.db.rows.get(self.path))


class Transaction:
    def __init__(self, db): self.db = db
    def create(self, ref, data):
        if ref.path in self.db.rows: raise AssertionError('must append')
        self.db.rows[ref.path] = deepcopy(data)
    def set(self, ref, data): self.db.rows[ref.path] = deepcopy(data)


class Database(Reference):
    def __init__(self):
        super().__init__(self)
        self.rows = {}
    def transaction(self): return Transaction(self)


class Service(FirebaseService):
    def __init__(self): self.database = Database()
    @property
    def db(self): return self.database


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.service = Service()
        self.decorator = patch('backend.firebase_service.firestore.transactional', side_effect=lambda function: function)
        self.decorator.start()
        self.addCleanup(self.decorator.stop)
        self.user = {'uid': 'a', 'name': 'Alice', 'role': 'employee'}
        self.body = {'track': 'quality', 'target': 'Test07_NG_dchg', 'decision': 'retest', 'note': '검토', 'requestId': 'abc'*8, 'expectedRevision': 0}

    def test_idempotency_and_append_only_corrections(self):
        first = self.service.save_record(self.body, 'quality:test', self.user, 'v2')
        again = self.service.save_record(self.body, 'quality:test', self.user, 'v2')
        self.assertEqual(first, again)
        self.assertEqual(len(self.service.db.rows), 2)
        corrected = self.service.save_record({**self.body, 'decision': 'hold', 'requestId': 'next', 'expectedRevision': 1}, 'quality:test', self.user, 'v2')
        self.assertEqual(corrected['previousDecision'], 'retest')
        self.assertEqual(corrected['revision'], 2)
        self.assertEqual(len(self.service.db.rows), 3)
        self.assertEqual(self.service.db.rows['/manufacturingRecords/'+first['id']]['decision'], 'retest')

    def test_mutated_retry_and_stale_writes_rejected(self):
        self.service.save_record(self.body, 'quality:test', self.user, 'v2')
        for body in [{**self.body, 'note': 'changed'}, {**self.body, 'requestId': 'another'}]:
            with self.assertRaises(HTTPException) as error:
                self.service.save_record(body, 'quality:test', self.user, 'v2')
            self.assertEqual(error.exception.status_code, 409)

    def test_conversation_uid_boundary_and_revision(self):
        result = {'text': '근거', 'citations': [], 'status': 'answered'}
        self.service.save_conversation('a', 'thread', 0, '질문', result)
        self.assertEqual(self.service.conversation('a', 'thread')['revision'], 1)
        with self.assertRaises(HTTPException) as error:
            self.service.conversation('b', 'thread')
        self.assertEqual(error.exception.status_code, 404)
        with self.assertRaises(HTTPException) as error:
            self.service.save_conversation('a', 'thread', 0, 'stale', result)
        self.assertEqual(error.exception.status_code, 409)


if __name__ == '__main__': unittest.main()
