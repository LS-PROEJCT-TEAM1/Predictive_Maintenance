"""Read-only, release-validated Firestore snapshots. No network work at import.

The bundled manifest is a compatibility contract, never a connected-mode fallback.
Raw sensor files and inference artifacts remain local and must match this release.
"""
from contextvars import ContextVar
from datetime import datetime, timezone
from threading import RLock
import time

from fastapi import HTTPException
from backend.data import Repository


def utcnow():
    return datetime.now(timezone.utc).isoformat()


class ReleaseMismatch(Exception):
    pass


class FirestoreAnalysis:
    def __init__(self, service, contract, ttl=60, clock=time.monotonic):
        self.service, self.contract = service, contract
        self.ttl, self.clock = ttl, clock
        self._lock = RLock()
        self._snapshot = None
        self._marker = None
        self._until = 0
        self._checked_at = self._loaded_at = None
        self._error = None
        self._attempts = self._loads = 0
        self._pinned = ContextVar('analysis_snapshot_' + str(id(self)), default=None)

    def _read_root(self):
        return self.service.db.document(self.contract.base).get(timeout=8, retry=None)

    def _read_documents(self):
        db = self.service.db
        refs = [db.document(path) for path in self.contract.docs]
        return {snap.reference.path: snap.to_dict()
                for snap in db.get_all(refs, timeout=20, retry=None) if snap.exists}

    def acquire(self, force=False):
        with self._lock:
            # A failed check has a backoff, even if a client repeatedly clicks refresh.
            if self.clock() >= self._until or (force and not self._error):
                self._attempts += 1
                try:
                    root = self._read_root()
                    data = root.to_dict() or {}
                    if data.get('dataVersion') != self.contract.manifest['dataVersion'] or data.get('schemaVersion') != self.contract.manifest['schemaVersion']:
                        raise ReleaseMismatch('공식 분석 버전과 로컬 실행 자료가 다릅니다. 실행 자료를 업데이트해야 합니다.')
                    if self._snapshot is None or root.update_time != self._marker:
                        docs = self._read_documents()
                        after = self._read_root()
                        if root.update_time != after.update_time:
                            raise ReleaseMismatch('공식 자료 갱신 중입니다. 잠시 후 다시 확인하세요.')
                        # A release version binds results to raw measurements/model artifacts.
                        # Reject partial or silently edited releases rather than mixing evidence.
                        if docs != self.contract.docs:
                            raise ReleaseMismatch('공식 자료의 문서·내용이 배포 버전과 일치하지 않습니다. 자료 동기화를 확인하세요.')
                        self._snapshot = Repository.from_documents(self.contract.manifest, docs)
                        self._marker = after.update_time
                        self._loaded_at = utcnow()
                        self._loads += 1
                    self._checked_at = utcnow()
                    self._error = None
                except ReleaseMismatch as exc:
                    self._error = str(exc)
                except Exception:
                    self._error = 'Firestore 연결 또는 할당량을 확인하지 못했습니다.'
                self._until = self.clock() + self.ttl
            if self._snapshot is None:
                raise HTTPException(503, (self._error or '공식 분석 자료 연결 대기 중입니다.') + ' 로컬 시드로 대체하지 않았습니다.')
            return self._snapshot, self.status()

    def status(self):
        with self._lock:
            return {'source': 'firestore', 'state': 'stale' if self._snapshot and self._error else 'ready' if self._snapshot else 'unavailable' if self._error else 'pending',
                    'dataVersion': self._snapshot.manifest['dataVersion'] if self._snapshot else None,
                    'checkedAt': self._checked_at, 'loadedAt': self._loaded_at,
                    'message': self._error, 'documents': len(self._snapshot.docs) if self._snapshot else 0,
                    'rootChecks': self._attempts, 'snapshotLoads': self._loads,
                    'localAssets': '품질 원본 측정·추론 모델은 동일 버전의 로컬 실행 자료'}

    def bind(self, snapshot):
        return self._pinned.set(snapshot)

    def unbind(self, token):
        self._pinned.reset(token)

    def current(self):
        return self._pinned.get() or self.acquire()

    def meta(self):
        repo, status = self.current()
        return {**repo.meta(), 'mode': 'firestore-analysis', 'analysisSource': status}

    @property
    def source_description(self):
        _, status = self.current()
        return ('Firestore 마지막 확인 자료 (최신 확인 실패)' if status['state'] == 'stale' else 'Firestore 공식 분석 자료') + ' · 실시간 현장 데이터 아님'

    def __getattr__(self, name):
        return getattr(self.current()[0], name)
