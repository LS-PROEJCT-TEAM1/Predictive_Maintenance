"""Reproduce the existing PCA parameters and verify parity before packaging.

Maintainer-only: normal reference data is required. No model/threshold search,
no overwriting of research outputs or official Firestore seeds.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.quality_engine import CV, TP, FEATURES, clean_frame, features, scores
from backend.runtime_assets import RUNTIME


def main():
    research = ROOT/'배터리 품질보증 모델'
    frames, curves, boxes, sources = [], [], [], []
    def load(path):
        sources.append({'file': path.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        return pd.read_csv(path)
    def summaries(name, mode, grade, clean):
        f = features(clean)
        ids = np.unique(np.linspace(0, len(f)-1, min(201, len(f)), dtype=int))
        curves.append({'file': name, 'mode': mode, 'grade': grade, 'rows': len(f),
                       'points': [{'progress': round(i/(len(f)-1)*100, 3), 'dv': round(float(f.iloc[i][FEATURES[0]])*1000, 4)} for i in ids]})
        q = np.quantile(f[FEATURES[4]], [0, .25, .5, .75, 1])
        boxes.append({'file': name, 'mode': mode, 'grade': grade, 'q': q.tolist(), 'rows': len(f)})
        return f
    for mode in ('chg', 'dchg'):
        for n in range(1000, 1005):
            name = f'{n}_{mode}'
            clean = clean_frame(load(research/f'data/preprocessed/train/{name}.csv'))
            frames.append(summaries(name, mode, 'normal-reference', clean))
    train = pd.concat(frames, ignore_index=True)
    scaler = StandardScaler().fit(train)
    pca = PCA(n_components=.95, random_state=42).fit(scaler.transform(train))
    parameters = {'version': 'quality-pca-existing-replay-v1', 'features': FEATURES, 'mean': scaler.mean_.tolist(),
                  'scale': scaler.scale_.tolist(), 'pcaMean': pca.mean_.tolist(), 'components': pca.components_.tolist(),
                  'variance': pca.explained_variance_.tolist(), 'varianceRatio': pca.explained_variance_ratio_.tolist(),
                  'normalRows': len(train), 'k': 10, 't2': 1., 'spe': 1.}
    t2, spe, _ = scores(train, parameters)
    parameters['t2'], parameters['spe'] = float(np.quantile(t2, .99)), float(np.quantile(spe, .99))
    score_path = research/'output/models/model_C_pca_t2_spe_이상점수.csv'
    stored = load(score_path)
    parity = []
    for name in stored['파일'].unique():
        clean = clean_frame(load(research/f'data/raw_data/test/{name}.csv'))
        old = stored[stored['파일'] == name].sort_values('시점')
        f = summaries(name, 'dchg' if name.endswith('dchg') else 'chg', 'OK' if '_OK_' in name else 'NG', clean)
        t2, spe, pred = scores(f, parameters)
        if len(old) != len(f) or not np.array_equal(pred, old['예측']):
            raise ValueError(f'{name}: stored prediction parity failed')
        np.testing.assert_allclose(t2, old['Hotelling_T2'], rtol=1e-7, atol=.0001)
        np.testing.assert_allclose(spe, old['SPE'], rtol=1e-7, atol=.0001)
        parity.append({'file': name, 'rows': len(f), 'predictionMismatch': 0})
    parameters.update(sources=sources, parity=parity, reproduction='Existing normal fit recipe reproduced; thresholds fixed at original q=.99, k=10; no selection on tests.')
    directory = RUNTIME/'quality'
    directory.mkdir(exist_ok=True)
    outputs = {'quality/pca_frozen.json': parameters, 'quality/exploration.json': {'curves': curves, 'temperatureBoxes': boxes, 'sources': sources}}
    manifest = json.loads((RUNTIME/'manifest.json').read_text(encoding='utf-8'))
    paths = set(outputs) | {'validation/quality_scores.csv', 'validation/quality_folds.csv'}
    manifest['files'] = [p for p in manifest['files'] if p['file'] not in paths]
    for name, value in outputs.items():
        raw = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
        (RUNTIME/name).write_bytes(raw)
        manifest['files'].append({'file': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'source': 'scripts/package_quality.py'})
    for name, source in [('quality_scores', score_path), ('quality_folds', research/'output/models/model_C_pca_t2_spe_교차검증_fold별.csv')]:
        raw = source.read_bytes()
        path = 'validation/'+name+'.csv'
        (RUNTIME/path).write_bytes(raw)
        manifest['files'].append({'file': path, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                                  'source': source.relative_to(ROOT).as_posix(), 'sourceSha256': hashlib.sha256(raw).hexdigest()})
    (RUNTIME/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'parity': parity, 'components': len(parameters['variance']), 'normalRows': len(train), 't2': parameters['t2'], 'spe': parameters['spe']}))


if __name__ == '__main__':
    main()
