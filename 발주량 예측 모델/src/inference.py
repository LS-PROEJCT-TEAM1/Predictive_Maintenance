"""Use the frozen audited model policy, without request targets or test selection."""
from __future__ import annotations
import argparse,json,os
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from demand_contract import ROOT, SEQUENCE, BASELINES, feature_record

MODEL_DIR=Path(os.environ.get('MANUFACTURING_DEMAND_MODEL_DIR',ROOT/'models/audited_v2/source_total/deployment'))
REQUIRED_COLUMNS=['part_number','date',*SEQUENCE]

@lru_cache(maxsize=1)
def load_metadata():
    return json.loads((MODEL_DIR/'metadata.json').read_text(encoding='utf-8'))

def validate_records(records):
    missing=set(REQUIRED_COLUMNS)-set(records.columns)
    if missing:raise ValueError('Missing columns: '+', '.join(sorted(missing)))
    clean=records[REQUIRED_COLUMNS].copy()
    if not 3<=len(clean)<=60:raise ValueError('Provide 3 to 60 daily rows; 8 days are recommended.')
    if clean.part_number.isna().any() or clean.part_number.astype(str).nunique()!=1:
        raise ValueError('All rows must use the same valid part_number.')
    clean['part_number']=clean.part_number.astype(str).str.strip()
    if clean.part_number.eq('').any():raise ValueError('part_number is empty.')
    clean['date']=pd.to_datetime(clean.date,errors='coerce').dt.normalize()
    for col in SEQUENCE:clean[col]=pd.to_numeric(clean[col],errors='coerce')
    if clean.isna().any().any() or not np.isfinite(clean[SEQUENCE]).all().all():raise ValueError('Input contains missing or non-finite values.')
    if (clean[SEQUENCE]<0).any().any():raise ValueError('Quantities must be zero or positive.')
    if clean.date.duplicated().any():raise ValueError('Duplicate daily records require review.')
    clean=clean.sort_values('date').reset_index(drop=True)
    if not clean.date.tail(3).diff().dropna().dt.days.eq(1).all():raise ValueError('The latest three dates must be consecutive calendar days.')
    if clean.iloc[-1].part_number in load_metadata()['quarantined_parts']:
        raise ValueError('This part is quarantined because conflicting source snapshots have not been resolved.')
    return clean

def build_feature_row(records):
    clean=validate_records(records);origin=clean.iloc[-1].date
    row=feature_record(clean,origin)
    return pd.DataFrame([row]),row

@lru_cache(maxsize=8)
def load_artifact(model):
    if model=='LSTM':
        import torch
        return torch.load(MODEL_DIR/'lstm.pt',map_location='cpu',weights_only=False)
    return joblib.load(MODEL_DIR/(model.lower().replace(' ','_')+'.joblib'))

def predict_model(name,frame):
    if name in BASELINES:return float(frame.iloc[0][name])
    artifact=load_artifact(name)
    if name=='LSTM':
        import torch
        from torch import nn
        state=artifact['state'];hidden=artifact['hidden']
        class Net(nn.Module):
            def __init__(self):
                super().__init__();self.rnn=nn.LSTM(4,hidden,batch_first=True)
                self.embedding=nn.Embedding(len(state['parts'])+1,8)
                self.head=nn.Sequential(nn.Linear(hidden+8+len(state['cols']),32),nn.ReLU(),nn.Linear(32,1))
            def forward(self,x,p,c):
                _,(h,_)=self.rnn(x);return self.head(torch.cat([h[-1],self.embedding(p),c],dim=1)).squeeze(-1)
        seq=frame[[f'{c}_{s}' for s in ['lag2','lag1','origin'] for c in SEQUENCE]].to_numpy(float).reshape(-1,3,4)
        x=torch.tensor(state['seq_scaler'].transform(seq.reshape(-1,4)).reshape(-1,3,4),dtype=torch.float32)
        p=torch.tensor([state['parts'].get(frame.iloc[0].part_number,len(state['parts']))],dtype=torch.int64)
        context=torch.tensor(state['context'].transform(frame[state['cols']]),dtype=torch.float32)
        model=Net();model.load_state_dict(artifact['weights']);model.eval()
        with torch.no_grad():value=float(model(x,p,context)[0])*state['target_scale']
    else:
        if name=='Related LightGBM':raise ValueError('Related-part inference requires related snapshots.')
        value=float(artifact['model'].predict(artifact['preprocessor'].transform(frame))[0])
    return max(0.,value)

def predict_records(records):
    frame,context=build_feature_row(records);meta=load_metadata();part=context['part_number']
    known=part in meta['model']['part_to_id'];method=meta['selection']['overall_method'];aux=meta['selection']['ml_model']
    fallback=None
    if not known:fallback='학습 이력이 없는 부품: 최근값 기준 예측';method='Last Value'
    if method=='Related LightGBM':fallback='연관 부품의 현재 입력 미제공: 최근값 기준 예측';method='Last Value'
    forecast=predict_model(method,frame)
    auxiliary=predict_model(aux,frame) if known and aux!='Related LightGBM' else None
    return {'part_number':part,'origin_date':context['origin_date'].date().isoformat(),
        'target_date':context['target_date'].date().isoformat(),'known_part':known,
        'xgboost_prediction':predict_model('XGBoost',frame) if known else None,
        'auxiliary_model':aux,'auxiliary_prediction':auxiliary,
        'moving_average_3d':context['3-day Moving Average'],'plan_d3_reference':context['D+3 Plan Reference'],
        'recommended_model':method,'recommended_forecast':forecast,'model_version':meta['version'],
        'fallback_reason':fallback,'history_days_7':context['history_count7'],
        'input_warning':'최근 7일 중 일부 이력이 없어 관측된 날짜만 평균에 사용합니다.' if context['history_count7']<7 else None,
        'as_of':'일별 최종 로그 확정 이후','evaluation_note':'과거 자료의 회고 검증 결과이며 새 현장 성능 보장은 아닙니다.'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('input_csv',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    result=json.dumps(predict_records(pd.read_csv(args.input_csv)),ensure_ascii=False,indent=2)
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(result,encoding='utf-8')
    print(result)

if __name__=='__main__':main()
