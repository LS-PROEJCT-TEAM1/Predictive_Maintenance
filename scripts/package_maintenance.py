"""Package existing operating parameters and normal-fit summaries; never refit."""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.data import Repository
from backend.runtime_assets import RUNTIME
from backend.maintenance_engine import replay, assets


def main():
    research = ROOT/'배터리 예지보전 모델'
    directory = research/'outputs/track_b_final_v2/models'
    sup = joblib.load(directory/'LogisticCurrent.joblib')
    unsup = joblib.load(directory/'RobustPhaseZ.joblib')
    data_root = research/'Dataset_전자부품(배터리팩) 예지보전 AI 데이터셋/data'
    normal_path = data_root/'raw_data/train/Training_Data.csv'
    original = pd.read_csv(normal_path)
    original.columns = original.columns.str.strip()
    split_path = research/'outputs/track_b_final_v2/historical_split_manifest.csv'
    split = pd.read_csv(split_path)
    indexes = [i for start in split.loc[split.split.eq('fit'),'original_start_row'] for i in range(int(start),int(start)+39)]
    train = original.iloc[indexes]
    pipe = sup['model']
    summary = []
    for power, frame in train.groupby('SetPower'):
        summary.append({'setPower':float(power),'rows':len(frame),'q':np.quantile(frame.RealPower,[0,.25,.5,.75,1]).tolist()})
    params = {'version':sup['pipeline_version'],'median':unsup['reference']['median'].sort_index().tolist(),
        'scale':unsup['reference']['scale'].sort_index().tolist(),'threshold':float(unsup['threshold']),
        'logistic':{'features':sup['feature_names'],'center':pipe['scale'].center_.tolist(),'scale':pipe['scale'].scale_.tolist(),
                    'coef':pipe['model'].coef_[0].tolist(),'intercept':float(pipe['model'].intercept_[0]),'threshold':float(sup['threshold'])},
        'recipes':train[['Speed','Length','SetPower']].drop_duplicates().to_numpy(float).tolist(),
        'frequency':float(train.SetFrequency.iloc[0]),'duty':float(train.SetDuty.iloc[0]),
        'normalSummary':summary,'normalRows':len(train),'fitEnd':split.loc[split.split.eq('fit'),'end'].max(),
        'sources':[{'file':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                   for p in [normal_path,split_path,directory/'LogisticCurrent.joblib',directory/'RobustPhaseZ.joblib']], 'parity':[]}
    target = RUNTIME/'maintenance/frozen.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(params,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    assets.cache_clear()
    repo = Repository()
    for name in ['WeldingTest_01_OK','WeldingTest_02_OK','WeldingTest_03_NG','WeldingTest_04_NG']:
        frame = pd.read_csv(data_root/f'raw_data/test/{name}.csv')
        frame.columns = frame.columns.str.strip()
        frame.WorkingTime = pd.to_datetime(frame.WorkingTime)
        s,u,_ = replay(frame)
        points = sorted([p for k,v in repo.docs.items() if k.startswith(f'{repo.base}/maintenanceRuns/{name}/measurementChunks/') for p in v['points']],key=lambda p:p['sourceRow'])
        expected_s = np.array([p['models']['LogisticCurrent']['score'] for p in points])
        expected_u = np.array([p['models']['RobustPhaseZ']['score'] for p in points])
        np.testing.assert_allclose(s,expected_s,rtol=1e-6,atol=1e-8)
        np.testing.assert_allclose(u,expected_u,rtol=1e-6,atol=1e-8)
        np.testing.assert_array_equal(s>=sup['threshold'],expected_s>=sup['threshold'])
        np.testing.assert_array_equal(u>=unsup['threshold'],expected_u>=unsup['threshold'])
        params['parity'].append({'file':name,'rows':len(frame),'predictionMismatch':0})
    target.write_text(json.dumps(params,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf-8')
    # Sample is an existing normal test, not a manufactured operating example.
    sample = RUNTIME/'maintenance/sample.csv'
    sample.write_bytes((data_root/'raw_data/test/WeldingTest_02_OK.csv').read_bytes())
    manifest_path = RUNTIME/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['files'] = [p for p in manifest['files'] if not p['file'].startswith('maintenance/')]
    for path in [target,sample]:
        raw = path.read_bytes()
        manifest['files'].append({'file':path.relative_to(RUNTIME).as_posix(),'bytes':len(raw),
            'sha256':hashlib.sha256(raw).hexdigest(),'source':'scripts/package_maintenance.py'})
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'parity':params['parity'],'bytes':target.stat().st_size+sample.stat().st_size}))


if __name__ == '__main__':
    main()
