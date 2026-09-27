"""Export audited related-model comparisons. Training lives in retrain_audited.py.

Kept as the historical CLI entrypoint; never select parts using retrospective labels.
"""
import json
import pandas as pd
import numpy as np
from demand_contract import ROOT, metrics

def main():
    source=ROOT/'outputs/audited_v2/source_total'
    out=ROOT/'outputs/related_part_model';out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((ROOT/'outputs/audited_v2/metadata.json').read_text(encoding='utf-8'))
    cv=pd.read_csv(source/'cv_predictions.csv')
    retro=pd.read_csv(source/'final_holdout_predictions.csv')
    primary=meta['selection']['overall_method']
    base_error=metrics(cv.actual,cv[primary])['MAE']
    related_error=metrics(cv.actual,cv['Related LightGBM'])['MAE']
    global_enabled=related_error < base_error*.98
    parts=[]
    for part,g in cv.groupby('part_number'):
        a=metrics(g.actual,g[primary])['MAE'];b=metrics(g.actual,g['Related LightGBM'])['MAE']
        parts.append(dict(part_number=part,baseline_model=primary,baseline_mae=a,related_model_mae=b,
            improvement_pct=(a-b)/a*100 if a else 0,related_model_enabled=bool(global_enabled and len(g)>=8),
            validation_note='CV only; activation requires global CV improvement >=2% and >=8 part rows'))
    part_metrics=pd.DataFrame(parts)
    retro['standalone_prediction']=retro['LightGBM'];retro['related_prediction']=retro['Related LightGBM']
    # Keep all target dates so the drawer never substitutes another day's forecast.
    latest=retro.merge(part_metrics,on='part_number',how='left')
    latest['default_prediction']=latest[primary]
    latest['selected_prediction']=np.where(latest.related_model_enabled.fillna(False),latest.related_prediction,latest.default_prediction)
    latest['safety_buffer']=meta['uncertainty']['buffer']
    latest['recommended_quantity_default']=np.ceil(latest.default_prediction+latest.safety_buffer)
    latest['recommended_quantity_related']=np.ceil(latest.related_prediction+latest.safety_buffer)
    relations=json.loads((source/'retrospective_relations.json').read_text(encoding='utf-8'))
    daily=pd.read_csv(ROOT/'outputs/audited_v2/daily_history.csv')
    pivot=daily[daily.date<=relations['cutoff']].pivot(index='date',columns='part_number',values='actual_d')
    rows=[]
    for target,candidates in relations['mapping'].items():
        for rank,part in enumerate(candidates,1):
            pair=pivot[[target,part]].dropna()
            rows.append(dict(target_part=target,part_number=part,rank=rank,correlation=pair.corr().iloc[0,1],
                overlap_days=len(pair),relation_end=relations['cutoff'],business_relation_verified=False))
    pd.DataFrame(rows).to_csv(out/'related_part_candidates.csv',index=False,encoding='utf-8-sig')
    part_metrics.to_csv(out/'part_model_metrics.csv',index=False,encoding='utf-8-sig')
    latest.to_csv(out/'latest_predictions.csv',index=False,encoding='utf-8-sig')
    retro.to_csv(out/'holdout_predictions.csv',index=False,encoding='utf-8-sig')
    pd.read_csv(source/'cv_metrics_pooled.csv').to_csv(out/'model_metrics.csv',index=False,encoding='utf-8-sig')
    (out/'RELATED_PART_MODEL_REPORT.md').write_text(
        f'# 연관 부품 검증\n\n현재 학습: audited_v2. 모델 원본은 models/audited_v2에 있습니다.\n\n'
        f'전역 CV MAE: {primary} {base_error:.4f}, Related LightGBM {related_error:.4f}. '
        f'활성화: {global_enabled}. 부품별 회고 오차로 토글을 켜지 않습니다.\n\n'
        f'상관 계산 종료일: {relations["cutoff"]}. 검증된 BOM 관계가 아닌 통계적 후보입니다.\n',encoding='utf-8')
    print(f'Related export complete; global CV gate: {global_enabled}')

if __name__=='__main__':main()
