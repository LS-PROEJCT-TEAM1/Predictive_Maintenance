"""Maintainer only: build a small reproducible runtime bundle from existing artifacts."""
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.data import Repository
from backend.runtime_assets import ARTIFACTS, RUNTIME


def main():
    repo = Repository()
    entries = []

    def put(relative, source, compressed=False):
        raw = source.read_bytes()
        data = gzip.compress(raw, mtime=0) if compressed else raw
        target = RUNTIME / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        entries.append({'file': relative, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                        'source': source.relative_to(ROOT).as_posix(), 'sourceSha256': hashlib.sha256(raw).hexdigest()})

    for test in repo.meta()['tests']:
        put('quality/' + test + '.csv.gz', ROOT / '배터리 품질보증 모델/data/raw_data/test' / (test + '.csv'), True)
    for name, source in ARTIFACTS.items():
        put('validation/' + name + '.csv', ROOT / source)
    for name in ['metadata.json', 'tree_preprocessor.joblib', 'xgboost.json']:
        put('demand/' + name, ROOT / '발주량 예측 모델/models/purged_v1/source_total' / name)
    (RUNTIME / 'manifest.json').write_text(json.dumps({'dataVersion': repo.manifest['dataVersion'], 'files': entries}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Runtime: {len(entries)} files, {sum(x['bytes'] for x in entries)/1048576:.2f} MiB")


if __name__ == '__main__':
    main()
