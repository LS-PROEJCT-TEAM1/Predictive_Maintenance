"""Backup, diff and atomically synchronize the official workspace only.

Unrelated roots, employee identities and application records are never accessed.
Pruning is restricted to paths from archived official seed manifests.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.data import Repository
from backend.settings import settings
from firestore.deploy_firestore import iter_seed, validate_seed, firestore_client


def diff(local, remote, retired):
    create = sorted(local.keys()-remote.keys())
    update = sorted(p for p in local.keys() & remote.keys() if local[p]!=remote[p])
    delete = sorted((remote.keys()-local.keys()) & retired)
    preserved = sorted(remote.keys()-local.keys()-retired)
    return {'create':create,'update':update,'delete':delete,'preserved':preserved,
            'unchanged':len(local)-len(create)-len(update)}


def read_tree(reference):
    snapshots = {}
    def visit(ref, snapshot=None):
        snap = snapshot if snapshot is not None else ref.get(timeout=20,retry=None)
        if snap.exists:
            snapshots[ref.path] = snap
        for collection in ref.collections(timeout=20,retry=None):
            for child in collection.stream(timeout=25,retry=None):
                visit(child.reference,child)
    visit(reference)
    return snapshots


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-id',required=True)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    config=settings()
    if config['project_id']!=args.project_id:
        raise ValueError('Target project does not match local settings')
    import os
    os.environ['GOOGLE_APPLICATION_CREDENTIALS']=config['service_account_file']
    repo=Repository()
    seed=ROOT/'firestore/seed'
    count,_=validate_seed(seed)
    local={x['path']:x['data'] for x in iter_seed(seed)}
    root=repo.manifest['rootDocument']
    if root!='manufacturingAi/manufacturing-ai' or any(p!=root and not p.startswith(root+'/') for p in local):
        raise ValueError('Unexpected seed scope')
    retired=set()
    for manifest in (ROOT/'firestore/versions').glob('*/manifest.json'):
        retired.update(x['path'] for x in iter_seed(manifest.parent))
    db=firestore_client(args.project_id,'(default)')
    snapshots=read_tree(db.document(root))
    remote={p:s.to_dict() for p,s in snapshots.items()}
    plan=diff(local,remote,retired)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder=ROOT/'.local/firestore-deployments'/stamp
    folder.mkdir(parents=True)
    # This backup contains only the official analysis workspace, never credentials.
    backup={'project':args.project_id,'root':root,'documents':remote}
    (folder/'before.json').write_text(json.dumps(backup,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    summary={'project':args.project_id,'database':'(default)','root':root,'dataVersion':repo.manifest['dataVersion'],
             'localDocuments':count,'remoteBefore':len(remote),'counts':{k:len(v) if isinstance(v,list) else v for k,v in plan.items()},
             'plan':plan,'backup':str(folder/'before.json'),'mode':'apply' if args.apply else 'dry-run'}
    (folder/'plan.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='plan'},ensure_ascii=False),flush=True)
    if not args.apply:
        return
    from google.cloud import firestore as fs
    batch=db.batch()
    for path in plan['create']:
        batch.create(db.document(path),local[path])
    for path in plan['update']:
        # Replace top-level fields, including removal of obsolete fields, with a
        # precondition so concurrent changes cannot silently be overwritten.
        value={**local[path],**{k:fs.DELETE_FIELD for k in remote[path].keys()-local[path].keys()}}
        batch.update(db.document(path),value,option=db.write_option(last_update_time=snapshots[path].update_time))
    for path in plan['delete']:
        batch.delete(db.document(path),option=db.write_option(last_update_time=snapshots[path].update_time))
    writes=len(plan['create'])+len(plan['update'])+len(plan['delete'])
    size=sum(type(w).pb(w).ByteSize() for w in batch._write_pbs)
    if writes>500 or size>9_000_000:
        raise ValueError(f'Atomic commit budget exceeded: {writes} operations, {size} bytes; no writes executed')
    summary['commitOperations']=writes
    summary['commitBytes']=size
    if writes:
        batch.commit(timeout=45,retry=None)
    after=read_tree(db.document(root))
    actual={p:s.to_dict() for p,s in after.items()}
    missing=sorted(local.keys()-actual.keys())
    changed=sorted(p for p in local.keys() & actual.keys() if local[p]!=actual[p])
    unexpected=sorted(actual.keys()-local.keys()-set(plan['preserved']))
    summary.update({'verifiedAt':datetime.now(timezone.utc).isoformat(),'remoteAfter':len(actual),
                    'matchingDocuments':len(local)-len(missing)-len(changed),'missing':missing,'different':changed,'unexpected':unexpected,
                    'seedManifestSha256':hashlib.sha256((seed/'manifest.json').read_bytes()).hexdigest()})
    summary['status']='verified' if not missing and not changed and not unexpected else 'verification_failed'
    (folder/'result.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='plan'},ensure_ascii=False),flush=True)
    if summary['status']!='verified':
        raise RuntimeError('Post-deployment verification failed; inspect the saved result')


if __name__=='__main__':
    main()
