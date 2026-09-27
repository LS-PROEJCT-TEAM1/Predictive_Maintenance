"""Offline-only nested temporal selection, refit, and retrospective evaluation.

No cloud credentials or SDK imports. Run once after reviewing training_plan_v2.json.
"""
from __future__ import annotations
import argparse, hashlib, json, platform, random, subprocess, time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from demand_contract import *

OUT=ROOT/'outputs/audited_v2'
MODELS=ROOT/'models/audited_v2'
SEEDS=[42,137,2026]
REL_FEATURES=[f'rel{i}_{c}' for i in range(3) for c in ['actual_d','plan_d3','available']]


def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),encoding='utf-8')


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)


def relations(daily,cutoff):
    d=daily[daily.date<=cutoff]
    pivot=d.pivot(index='date',columns='part_number',values='actual_d')
    corr=pivot.corr(min_periods=7)
    mapping={}
    for p in pivot.columns:
        ranked=corr[p].drop(index=p).dropna().sort_values(ascending=False)
        mapping[p]=ranked[ranked>0].head(3).index.tolist()
    return mapping


def add_relations(frame,daily,mapping):
    f=frame.copy();lookup=daily.set_index(['part_number','date'])
    for i in range(3):
        values=[]
        for row in f.itertuples():
            rel=mapping.get(row.part_number,[])
            key=(rel[i],row.origin_date) if len(rel)>i else None
            if key is not None and key in lookup.index:
                r=lookup.loc[key];values.append([float(r.actual_d),float(r.plan_d3),1.])
            else:values.append([np.nan,np.nan,0.])
        f[[f'rel{i}_actual_d',f'rel{i}_plan_d3',f'rel{i}_available']]=values
    return f


def preprocessor(cols):
    return ColumnTransformer([
        ('part',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['part_number']),
        ('numeric',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),cols)
    ],sparse_threshold=0,verbose_feature_names_out=True)


def tree_model(family,config,seed,iterations=500,early=False):
    depth=3 if config['size']=='small' else 5
    if family=='XGBoost':
        kwargs={'early_stopping_rounds':35} if early else {}
        return XGBRegressor(objective='reg:absoluteerror',eval_metric='mae',n_estimators=iterations,
            learning_rate=.04,max_depth=depth,min_child_weight=5,subsample=.9,colsample_bytree=.9,
            reg_lambda=5,random_state=seed,n_jobs=4,tree_method='hist',**kwargs)
    if family in ['LightGBM','Related LightGBM']:
        return lgb.LGBMRegressor(objective='regression_l1',n_estimators=iterations,learning_rate=.04,
            num_leaves=7 if depth==3 else 15,max_depth=depth,min_child_samples=12,
            subsample=.9,subsample_freq=1,colsample_bytree=.9,reg_lambda=3,random_state=seed,n_jobs=4,verbosity=-1)
    return CatBoostRegressor(loss_function='MAE',eval_metric='MAE',iterations=iterations,
        learning_rate=.04,depth=depth,l2_leaf_reg=5,random_seed=seed,thread_count=4,
        allow_writing_files=False,verbose=False)


class DemandNet(nn.Module):
    def __init__(self,part_count,calendar_count,hidden):
        super().__init__();self.rnn=nn.LSTM(4,hidden,batch_first=True)
        self.embedding=nn.Embedding(part_count+1,8)
        self.head=nn.Sequential(nn.Linear(hidden+8+calendar_count,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x,part,context):
        _,(h,_)=self.rnn(x)
        return self.head(torch.cat([h[-1],self.embedding(part),context],dim=1)).squeeze(-1)


def neural_arrays(frame,state=None,cols=CALENDAR):
    seq=frame[[f'{c}_{s}' for s in ['lag2','lag1','origin'] for c in SEQUENCE]].to_numpy(float).reshape(-1,3,4)
    if state is None:
        state={'seq_scaler':StandardScaler().fit(seq.reshape(-1,4)),
            'context':Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]).fit(frame[cols]),
            'parts':{p:i for i,p in enumerate(sorted(frame.part_number.unique()))},
            'target_scale':max(float(frame.target.std(ddof=0)),1.),'cols':cols}
    x=state['seq_scaler'].transform(seq.reshape(-1,4)).reshape(-1,3,4).astype('float32')
    context=state['context'].transform(frame[state['cols']]).astype('float32')
    ids=frame.part_number.map(state['parts']).fillna(len(state['parts'])).to_numpy('int64')
    return (torch.from_numpy(x),torch.from_numpy(ids),torch.from_numpy(context)),state


def fit_neural(train,val,config,seed,epochs=None):
    seed_all(seed)
    cols=CALENDAR+(EXTRA if config['features']=='enhanced' else [])
    arrays,state=neural_arrays(train,cols=cols)
    v,_=neural_arrays(val,state)
    model=DemandNet(len(state['parts']),len(cols),16 if config['size']=='small' else 32)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.001)
    target=torch.tensor(train.target.to_numpy('float32')/state['target_scale'])
    loader=DataLoader(TensorDataset(*arrays,target),batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(seed))
    best=float('inf');best_epoch=1;best_state=None;history=[];wait=0
    for epoch in range(1,(epochs or 100)+1):
        model.train();losses=[]
        for x,p,c,y in loader:
            optimizer.zero_grad();loss=nn.functional.l1_loss(model(x,p,c),y)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(loss.item())
        model.eval()
        with torch.no_grad():pv=np.maximum(0,model(*v).numpy()*state['target_scale'])
        unseen=~val.part_number.isin(state['parts']).to_numpy()
        pv[unseen]=val.loc[unseen,'Last Value'].to_numpy()
        mae=metrics(val.target,pv)['MAE'];history.append({'epoch':epoch,'train_mae':float(np.mean(losses)*state['target_scale']),'validation_mae':mae})
        if mae<best-1e-6:best=mae;best_epoch=epoch;best_state=deepcopy(model.state_dict());wait=0
        else:wait+=1
        if epochs is None and wait>=15:break
    if epochs is None:model.load_state_dict(best_state)
    else:best_epoch=epochs
    model.eval()
    return {'engine':'LSTM','state':state,'weights':model.state_dict(),'hidden':16 if config['size']=='small' else 32,
            'iterations':best_epoch,'config':config,'history':history,'seed':seed},best


def predict_artifact(artifact,frame,daily):
    family=artifact['engine']
    if family=='LSTM':
        arrays,state=neural_arrays(frame,artifact['state'])
        model=DemandNet(len(state['parts']),len(state['cols']),artifact['hidden'])
        model.load_state_dict(artifact['weights']);model.eval()
        with torch.no_grad():p=model(*arrays).numpy()*state['target_scale']
    else:
        f=add_relations(frame,daily,artifact['relations']) if family=='Related LightGBM' else frame
        p=artifact['model'].predict(artifact['preprocessor'].transform(f))
    p=np.maximum(0,p)
    # New parts are explicitly supported by a fixed recent-value fallback, never omitted.
    known=artifact['known_parts'] if 'known_parts' in artifact else list(artifact['state']['parts'])
    unseen=~frame.part_number.isin(known).to_numpy()
    p[unseen]=frame.loc[unseen,'Last Value'].to_numpy()
    return p


def fit_tree(train,val,daily,family,config,seed,iterations=None):
    cols=BASIC+(EXTRA if config['features']=='enhanced' else [])
    mapping={};cutoff=train.target_date.max()
    if family=='Related LightGBM':
        mapping=relations(daily,cutoff);train=add_relations(train,daily,mapping);val=add_relations(val,daily,mapping);cols+=REL_FEATURES
    prep=preprocessor(cols);x=prep.fit_transform(train);v=prep.transform(val)
    # Unseen parts use a fixed fallback, so their constant error cannot choose a tree iteration.
    known_validation=val.part_number.isin(train.part_number).to_numpy()
    eval_x,eval_y=v[known_validation],val.loc[known_validation,'target']
    if not known_validation.any():raise ValueError('Inner validation requires previously observed parts.')
    model=tree_model(family,config,seed,iterations or 500,early=iterations is None)
    if iterations is None:
        if family=='XGBoost':model.fit(x,train.target,eval_set=[(eval_x,eval_y)],verbose=False);n=model.best_iteration+1
        elif family in ['LightGBM','Related LightGBM']:
            model.fit(x,train.target,eval_set=[(eval_x,eval_y)],eval_metric='mae',callbacks=[lgb.early_stopping(35,verbose=False)]);n=model.best_iteration_
        else:model.fit(x,train.target,eval_set=(eval_x,eval_y),early_stopping_rounds=35);n=model.get_best_iteration()+1
    else:model.fit(x,train.target);n=iterations
    artifact={'engine':family,'preprocessor':prep,'model':model,'config':config,'iterations':n,
        'relations':mapping,'relation_cutoff':str(cutoff.date()),'known_parts':sorted(train.part_number.unique()),'seed':seed}
    # Rows here may already have relation features; these do not affect other columns.
    prediction=np.maximum(0,model.predict(v))
    prediction[~known_validation]=val.loc[~known_validation,'Last Value'].to_numpy()
    return artifact,metrics(val.target,prediction)['MAE']


def fit_family(frame,daily,masks,family,seed=42,frozen=None):
    train_mask,val_mask,test_mask,refit_mask=masks
    train,val,test,refit=[frame.loc[m].copy() for m in masks]
    fit=fit_neural if family=='LSTM' else None
    candidates=[]
    if frozen is None:
        configs=([{'features':'basic','size':'small'},{'features':'enhanced','size':'medium'}] if family=='LSTM' else
                 [{'features':f,'size':s} for f in (['enhanced'] if family=='Related LightGBM' else ['basic','enhanced']) for s in ['small','medium']])
        best=None
        for config in configs:
            a,score=fit_neural(train,val,config,seed) if fit else fit_tree(train,val,daily,family,config,seed)
            candidates.append({'model':family,**config,'validation_mae':score,'iterations':int(a['iterations'])})
            if best is None or score<best[0]:best=(score,a)
        frozen={'config':best[1]['config'],'iterations':int(best[1]['iterations'])}
    # Fixed iterations after validation, no early stopping or holdout access during refit.
    if fit:artifact,_=fit_neural(refit,refit,frozen['config'],seed,epochs=frozen['iterations'])
    else:artifact,_=fit_tree(refit,refit,daily,family,frozen['config'],seed,iterations=frozen['iterations'])
    artifact['training_target_end']=str(refit.target_date.max().date())
    return predict_artifact(artifact,test,daily),artifact,candidates,frozen


def save_artifact(folder,family,artifact):
    folder.mkdir(parents=True,exist_ok=True)
    if family=='LSTM':torch.save(artifact,folder/'lstm.pt')
    else:joblib.dump(artifact,folder/(family.lower().replace(' ','_')+'.joblib'))


def metric_table(pred):
    return pd.DataFrame([{'Model':m,'N':len(pred),'Parts':pred.part_number.nunique(),
        'Eligible_For_Overall_Ranking':True,'Eligible_For_ML_Ranking':m in ML_NAMES,
        **metrics(pred.actual,pred[m])} for m in MODEL_NAMES]).sort_values(['MAE','RMSE']).reset_index(drop=True)


def evaluate(daily,scenario,locked=None):
    frame=build_samples(daily);plan=evaluation_plan(frame);folder=OUT/scenario;folder.mkdir(parents=True,exist_ok=True)
    predictions=[];audit=[];tuning=[];decisions={};artifacts={}
    for label,masks in plan:
        if label=='retrospective':
            # Global family selection frozen before touching retrospective labels.
            cv=pd.concat(predictions,ignore_index=True);rank=metric_table(cv)
            selection={'overall_method':str(rank.iloc[0].Model),
                'ml_model':str(rank[rank.Eligible_For_ML_Ranking].iloc[0].Model),
                'overall_cv_mae':float(rank.iloc[0].MAE),
                'ml_cv_mae':float(rank[rank.Eligible_For_ML_Ranking].iloc[0].MAE),
                'selection_basis':'nested_purged_walk_forward_global_policy',
                'final_holdout_is_selection_independent':False,
                'evaluation_label':'retrospective_previously_observed_period',
                'new_unseen_validation_available':False}
            dump(folder/'selection_frozen_before_retrospective.json',selection)
        tr,va,te,rf=masks;test=frame.loc[te]
        p=test[['part_number','origin_date','target_date','target']+BASELINES].rename(columns={'target':'actual'}).copy()
        p['moving_average_3d']=p['3-day Moving Average'];p['plan_d3_reference']=p['D+3 Plan Reference']
        p.insert(0,'Fold',label);p.insert(0,'Scenario',scenario)
        audit.append({'Scenario':scenario,'Split':label,'Train_Rows':int(tr.sum()),'Validation_Rows':int(va.sum()),'Refit_Rows':int(rf.sum()),'Test_Rows':int(te.sum()),
            'Train_Target_End':str(frame.loc[tr,'target_date'].max().date()),'Validation_Origin_Start':str(frame.loc[va,'origin_date'].min().date()),
            'Validation_Target_End':str(frame.loc[va,'target_date'].max().date()),'Refit_Target_End':str(frame.loc[rf,'target_date'].max().date()),
            'Test_Origin_Start':str(test.origin_date.min().date()),'Test_Target_Start':str(test.target_date.min().date()),'Test_Target_End':str(test.target_date.max().date()),
            'Train_Target_Days':frame.loc[tr,'target_date'].nunique(),'Validation_Target_Days':frame.loc[va,'target_date'].nunique(),
            'Cold_Start_Test_Rows':int((~test.part_number.isin(frame.loc[rf,'part_number'])).sum())})
        decisions[label]={}
        for family in ML_NAMES:
            print(f'{scenario} {label}: {family}',flush=True)
            values,artifact,candidates,choice=fit_family(frame,daily,masks,family,frozen=locked[label][family] if locked else None)
            p[family]=values;decisions[label][family]=choice
            tuning.extend([{'split':label,**r} for r in candidates])
            save_artifact(MODELS/scenario/label,family,artifact)
            if family=='LSTM':pd.DataFrame(artifact['history']).to_csv(folder/f'{label}_lstm_history.csv',index=False)
            if family=='Related LightGBM':dump(folder/f'{label}_relations.json',{'cutoff':artifact['relation_cutoff'],'mapping':artifact['relations']})
            if label=='retrospective':artifacts[family]=artifact
        if label=='retrospective':retro=p.reset_index(drop=True)
        else:predictions.append(p)
    cv=pd.concat(predictions,ignore_index=True)
    for name,data in [('cv',cv),('final_holdout',retro)]:
        data.to_csv(folder/f'{name}_predictions.csv',index=False)
        metric_table(data).to_csv(folder/('cv_metrics_pooled.csv' if name=='cv' else 'final_holdout_metrics.csv'),index=False)
        part=[]
        for p,g in data.groupby('part_number'):
            for m in MODEL_NAMES:part.append({'part_number':p,'Model':m,'N':len(g),**metrics(g.actual,g[m])})
        pd.DataFrame(part).to_csv(folder/f'{name}_part_metrics.csv',index=False)
    byfold=[]
    for i,(label,g) in enumerate(cv.groupby('Fold',sort=True),1):
        table=metric_table(g);table.insert(0,'Fold',i);table.insert(0,'Scenario',scenario);byfold.append(table)
    pd.concat(byfold).to_csv(folder/'cv_metrics_by_fold.csv',index=False)
    pd.DataFrame(audit).to_csv(folder/'split_audit.csv',index=False)
    pd.DataFrame(tuning).to_csv(folder/'tuning_trials.csv',index=False)
    dump(folder/'frozen_configurations.json',decisions)
    return frame,plan,cv,retro,selection,decisions,artifacts


def uncertainty(cv,retro,selection):
    model=selection['overall_method']
    # Pooled one-sided residual quantile, calibrated exclusively on CV forecasts.
    residual=(cv.actual-cv[model]).to_numpy();q=max(0,float(np.quantile(residual,.90,method='higher')))
    coverage=float(np.mean(retro.actual<=retro[model]+q))
    rng=np.random.default_rng(42);daily=cv.assign(delta=np.abs(cv[model]-cv.actual)-np.abs(cv['3-day Moving Average']-cv.actual)).groupby('target_date').delta.mean().to_numpy()
    # Resample whole target days, not falsely independent rows from correlated parts.
    samples=np.array([rng.choice(daily,size=len(daily),replace=True).mean() for _ in range(2000)])
    return {'method':'pooled_one_sided_CV_residual_quantile','nominal_coverage':.90,'buffer':q,
        'retrospective_coverage':coverage,'calibration_rows':len(cv),'calibration_target_days':cv.target_date.nunique(),
        'mae_delta_vs_ma3_day_bootstrap_interval95':np.quantile(samples,[.025,.975]).tolist(),
        'warning':'empirical demand bound, not safety stock or guaranteed service level; CV selected the policy, so calibration coverage may be optimistic'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--skip-sensitivity',action='store_true');args=parser.parse_args()
    if (OUT/'run_manifest.json').exists():raise SystemExit('Completed run already exists; preserve this version and create a new version for another experiment.')
    OUT.mkdir(parents=True,exist_ok=True)
    plan_doc={'version':VERSION,'created_at':datetime.now(timezone.utc).isoformat(),'seed':42,'stability_seeds':SEEDS,
        'quantity':'source_total','conflicts':'quarantine all parts with conflicting same-time snapshots',
        'target':'calendar D+3 final ERP daily order plan','input_availability':'end-of-day final snapshot only',
        'selection':'three purged outer folds of four dates; five-date inner validation; refit on labels strictly before outer origin',
        'baselines':BASELINES,'families':ML_NAMES,'ranking':['pooled_MAE','RMSE'],
        'holdout':'last seven target dates, retrospective and previously observed, not new independent evidence',
        'feature_comparison':['basic','enhanced'],'tree_sizes':['small','medium'],'related_candidates':'train-only positive Pearson, >=7 shared days, top3; candidate not verified BOM',
        'cold_start':'Last Value fallback, counted in all comparisons','calibration':'CV residuals only; never optimize using retrospective labels'}
    dump(OUT/'training_plan_v2.json',plan_doc)
    start=time.time();daily,quality,conflicts=load_daily()
    conflicts.to_csv(OUT/'quarantined_source_rows.csv',index=False)
    daily.to_csv(OUT/'daily_history.csv',index=False)
    dump(OUT/'data_quality.json',quality)
    frame,plan,cv,retro,selection,decisions,artifacts=evaluate(daily,'source_total')
    interval=uncertainty(cv,retro,selection);dump(OUT/'interval_calibration.json',interval)
    # Seed stability on the family choices already frozen for each outer fold.
    stability=[]
    for label,masks in plan[:-1]:
        for family in sorted({selection['ml_model'],'LSTM'}):
            for seed in SEEDS[1:]:
                print(f'stability {label} {family} seed={seed}',flush=True)
                values,a,_,_=fit_family(frame,daily,masks,family,seed=seed,frozen=decisions[label][family])
                stability.append({'fold':label,'Model':family,'seed':seed,**metrics(frame.loc[masks[2],'target'],values)})
    pd.DataFrame(stability).to_csv(OUT/'seed_stability.csv',index=False)
    sensitivity=[]
    for policy in ['first','last']:
        alternate,q,_=load_daily(conflict_policy=policy);sf=build_samples(alternate)
        keys=cv[['part_number','origin_date','target_date']]
        # Shared non-conflicting cohort comparison plus explicitly separate disputed cohort.
        for label,g in [('common_cohort',keys.merge(sf,on=['part_number','origin_date','target_date'])),('disputed_parts',sf[sf.part_number.isin(quality['quarantined_parts'])])]:
            for m in BASELINES:sensitivity.append({'policy':policy,'cohort':label,'Model':m,'N':len(g),**metrics(g.target,g[m])})
    pd.DataFrame(sensitivity).to_csv(OUT/'conflict_policy_sensitivity.csv',index=False)
    if not args.skip_sensitivity:
        sd,sq,_=load_daily('slot_sum');_,_,scv,sretro,ssel,_,_=evaluate(sd,'slot_sum',locked=decisions)
        dump(OUT/'slot_sum_sensitivity.json',{'quantity':sq,'selection':ssel,'configurations':'source_total choices frozen; no second tuning',
            'purpose':'target-definition sensitivity only; never choose target definition by smaller error'})
    # Export the retrospective-fit model bundle. It has never trained on retrospective labels.
    dest=MODELS/'source_total/deployment';dest.mkdir(parents=True,exist_ok=True)
    for family,a in artifacts.items():save_artifact(dest,family,a)
    metadata={'version':VERSION,'selection':selection,'quality':quality,'uncertainty':interval,
        'quarantined_parts':quality['quarantined_parts'],'model_names':MODEL_NAMES,'baseline_names':BASELINES,
        'feature_contract':{'recent_consecutive_days':3,'preferred_history_days':8,'features_basic':BASIC,'features_extra':EXTRA,
        'as_of':'after final daily snapshot','units':'count','target':'final ERP daily order-plan quantity'},
        'configurations':decisions['retrospective'],'model':{'feature_columns':['part_number']+BASIC+EXTRA,
        'part_to_id':{p:i for i,p in enumerate(sorted(frame.loc[plan[-1][1][3],'part_number'].unique()))}}}
    dump(dest/'metadata.json',metadata);dump(OUT/'metadata.json',metadata)
    code_files=[Path(__file__),ROOT/'src/demand_contract.py']
    manifest={'version':VERSION,'generated_at':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.time()-start,
        'raw_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files},
        'python':platform.python_version(),'libraries':{'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__,'lightgbm':lgb.__version__},
        'selection':selection,'training_complete':True,'firestore_uploaded':False}
    dump(OUT/'run_manifest.json',manifest)
    print(json.dumps(selection,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
