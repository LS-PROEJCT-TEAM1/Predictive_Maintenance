"""Read-only export for the explicitly offline demo. Never writes to Firestore."""
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.firebase_service import FirebaseService

if __name__ == '__main__':
    db = FirebaseService().db
    docs = list(db.collection('battery_packs').stream(timeout=30, retry=None))
    payload = {'packs': {doc.id: doc.to_dict() for doc in docs}, 'details': {},
               'dashboard': db.document('battery_meta/dashboard').get(timeout=30, retry=None).to_dict(),
               'config': db.document('battery_meta/config').get(timeout=30, retry=None).to_dict()}
    refs = [doc.reference.collection(group).document(name) for doc in docs
            for group, name in [('heatmap', 'cells'), ('series', 'score')]]
    for start in range(0, len(refs), 50):
        for doc in db.get_all(refs[start:start+50], timeout=30, retry=None):
            pack_id, group = doc.reference.path.split('/')[1:3]
            payload['details'].setdefault(pack_id, {})[group] = doc.to_dict()
    destination = ROOT / 'runtime' / 'quality' / 'battery_pack_demo.json.gz'
    destination.write_bytes(gzip.compress(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')))
    print(json.dumps({'packs': len(docs), 'details': len(payload['details']), 'bytes': destination.stat().st_size}))
