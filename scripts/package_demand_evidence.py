"""Package audited evidence without retraining or changing the official seed."""
import hashlib
import json
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def build():
    source = ROOT / '발주량 예측 모델/outputs/audited_v2'
    model = ROOT / '발주량 예측 모델/models/audited_v2/source_total/deployment/xgboost.joblib'
    daily = pd.read_csv(source/'daily_history.csv')
    cv = pd.read_csv(source/'source_total/cv_predictions.csv')
    artifact = joblib.load(model)
    names = artifact['preprocessor'].get_feature_names_out()
    scores = artifact['model'].get_booster().get_score(importance_type='total_gain')
    grouped = {}
    for i, name in enumerate(names):
        label = name.split('__', 1)[-1]
        if label.startswith('part_number_'):
            label = 'part_number (전체 인코딩)'
        grouped[label] = grouped.get(label, 0) + scores.get(f'f{i}', scores.get(name, 0))
    total = sum(grouped.values())
    importance = [{'feature':k, 'percent':v/total*100} for k,v in sorted(grouped.items(),key=lambda x:-x[1])]
    files = [source/'daily_history.csv', source/'source_total/cv_predictions.csv', source/'data_quality.json', model]
    payload = {'version':'audited_v2', 'quality':json.loads((source/'data_quality.json').read_text(encoding='utf-8')),
        'daily':daily[['part_number','date','actual_d','plan_d1','plan_d2','plan_d3','plan_d4','plan_d5']].to_dict('records'),
        'cv':cv[['part_number','origin_date','target_date','actual','Fold']].to_dict('records'),
        'importance':importance, 'importanceMethod':'XGBoost total_gain · 배포용 비교 모델 · 전체 분할 개선량 비율',
        'sources':[{'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
    target = ROOT/'runtime/demand/evidence.json'
    target.write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False),encoding='utf-8')
    manifest_path = ROOT/'runtime/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['files'] = [x for x in manifest['files'] if x['file']!='demand/evidence.json']
    manifest['files'].append({'file':'demand/evidence.json','bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Packaged {len(daily)} daily rows, {len(cv)} CV rows, {len(importance)} feature groups')


if __name__ == '__main__':
    build()
