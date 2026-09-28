"""Restore only this revision's files. Dry run by default; never touch credentials."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile
from datetime import datetime

here=Path(__file__).resolve().parent
root=here.parent.parent.resolve()
snapshot=json.loads((here/'snapshot.json').read_text(encoding='utf-8'))
revision=json.loads((here/'revision.json').read_text(encoding='utf-8'))
before={r['path']:r['sha256'] for r in snapshot['files']}
def target(name):
    path=(root/name).resolve()
    if not path.is_relative_to(root) or path==root:raise SystemExit('Unsafe path')
    return path
with zipfile.ZipFile(here/'rollback.zip') as archive:
    for item in revision['files']:
        name=item['path']; path=target(name)
        if name in before and hashlib.sha256(archive.read(name)).hexdigest()!=before[name]:
            raise SystemExit('Snapshot hash mismatch: '+name)
        actual=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if actual!=item['afterSha256']:
            raise SystemExit('Changed since revision; inspect and preserve before restoring: '+name)
    print('Verified rollback targets:',len(revision['files']))
    for item in revision['files']:print(item['path'])
    if '--restore' not in sys.argv:
        print('Dry run only. Stop the server and add --restore to apply.')
        raise SystemExit(0)
    backup=here/('before-restore-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'.zip')
    with zipfile.ZipFile(backup,'x',zipfile.ZIP_DEFLATED) as saved:
        for item in revision['files']:
            path=target(item['path'])
            if path.exists():saved.write(path,item['path'])
    for item in revision['files']:
        name=item['path'];path=target(name)
        if name in before:
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(archive.read(name))
        elif path.exists():
            path.unlink()  # Only the exact new file, already hash-checked and backed up.
    print('Restored. Prior revision preserved at:',backup)
