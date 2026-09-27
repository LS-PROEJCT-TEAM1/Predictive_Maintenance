"""Versioned, training/inference shared demand contract. No training dependencies."""
from pathlib import Path
import numpy as np
import pandas as pd

VERSION = 'audited_v2'
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'Dataset_공급망 최적화 AI 데이터셋/data/data.xls'
QUANTITIES = ['actual_d', 'plan_d1', 'plan_d2', 'plan_d3', 'plan_d4', 'plan_d5']
SEQUENCE = ['actual_d', 'plan_d3', 'plan_d4', 'plan_d5']
CALENDAR = ['dow_sin', 'dow_cos', 'is_weekend', 'month']
BASIC = [f'{c}_{s}' for s in ['lag2', 'lag1', 'origin'] for c in SEQUENCE] + CALENDAR
EXTRA = ['actual_lag7', 'actual_mean7', 'actual_std7', 'history_count7',
         'plan_revision1', 'plan_revision2', 'plan_revision_rate', 'actual_change1', 'actual_change2']
BASELINES = ['Last Value', '3-day Moving Average', '7-day Moving Average', 'D+3 Plan Reference']
ML_NAMES = ['XGBoost', 'LightGBM', 'CatBoost', 'LSTM', 'Related LightGBM']
MODEL_NAMES = BASELINES + ML_NAMES


def load_daily(quantity_source='source_total', conflict_policy='quarantine'):
    raw = pd.read_excel(SOURCE)
    # Validate labels as well as positions before interpreting the legacy XLS.
    for i, label in {0:'Part Number', 11:'D일 투입예정 수량(D일계획)',
                     44:'D+3일 투입예정 수량(Total)', 55:'D+4일 투입예정 수량(Total)',
                     56:'D+5일 투입예정 수량', 83:'CRET_TIME'}.items():
        if raw.columns[i] != label:
            raise ValueError(f'Unexpected source schema at column {i}: {raw.columns[i]}')
    selected = pd.DataFrame({'part_number':raw.iloc[:,0], 'row_id':np.arange(len(raw)),
        'timestamp':pd.to_datetime(raw.iloc[:,83].astype(str),format='%Y%m%d%H%M',errors='raise')})
    mismatch = {}
    for c, start, stop, total in [('actual_d',1,11,11),('plan_d1',12,22,22),
        ('plan_d2',23,33,33),('plan_d3',34,44,44),('plan_d4',45,55,55)]:
        source = pd.to_numeric(raw.iloc[:,total], errors='raise')
        slots = raw.iloc[:,start:stop].sum(axis=1, min_count=stop-start)
        mismatch[c] = int(source.ne(slots).sum())
        selected[c] = source if quantity_source == 'source_total' else slots
    selected['plan_d5'] = pd.to_numeric(raw.iloc[:,56], errors='raise')
    if selected.isna().any().any() or (selected[QUANTITIES] < 0).any().any():
        raise ValueError('Missing, invalid or negative source values require explicit review.')
    selected['date'] = selected.timestamp.dt.normalize()
    # Conflict detection uses all original numeric columns, independent of quantity scenario.
    groups = raw.groupby([raw.iloc[:,0],raw.iloc[:,83]],sort=False)
    conflicts = groups.nunique().max(axis=1).gt(1)
    bad_keys = set(conflicts.index[conflicts])
    bad_parts = sorted({str(p) for p,t in bad_keys})
    conflict_rows = raw[[ (p,t) in bad_keys for p,t in zip(raw.iloc[:,0],raw.iloc[:,83]) ]].copy()
    conflict_rows.insert(0,'excel_row',conflict_rows.index+2)
    ordered = selected.sort_values(['part_number','timestamp','row_id'])
    if conflict_policy == 'quarantine':
        ordered = ordered[~ordered.part_number.isin(bad_parts)]
    elif conflict_policy == 'first':
        ordered = ordered.drop_duplicates(['part_number','timestamp'],keep='first')
    elif conflict_policy != 'last':
        raise ValueError('Unknown conflict policy')
    daily = ordered.drop_duplicates(['part_number','date'],keep='last').reset_index(drop=True)
    quality = {'source_rows':len(raw),'source_columns':len(raw.columns),'source_parts':raw.iloc[:,0].nunique(),
        'source_missing_cells':int(raw.isna().sum().sum()),'quantity_source':quantity_source,
        'conflict_policy':conflict_policy,'quarantined_parts':bad_parts,'conflicting_timestamp_groups':len(bad_keys),
        'conflict_rows':len(conflict_rows),'daily_rows':len(daily),'daily_parts':daily.part_number.nunique(),
        'observed_dates':daily.date.nunique(),'total_slot_mismatch_rows':mismatch,
        'forecast_as_of':'after the final daily source snapshot; not an intraday forecast',
        'target_definition':'final daily ERP order-plan quantity, not independently measured consumption'}
    return daily, quality, conflict_rows


def feature_record(group, origin):
    lookup = group.set_index('date')
    dates = [origin-pd.Timedelta(days=k) for k in [2,1,0]]
    if not all(d in lookup.index for d in dates):
        return None
    rows = lookup.loc[dates]
    result = {'part_number':str(rows.iloc[-1].part_number),'origin_date':origin,
              'target_date':origin+pd.Timedelta(days=3)}
    for suffix, (_, row) in zip(['lag2','lag1','origin'],rows.iterrows()):
        for c in SEQUENCE:result[f'{c}_{suffix}'] = float(row[c])
    result.update(dow_sin=float(np.sin(2*np.pi*origin.dayofweek/7)),
        dow_cos=float(np.cos(2*np.pi*origin.dayofweek/7)),is_weekend=int(origin.dayofweek>=5),month=origin.month)
    hist=lookup.reindex(pd.date_range(origin-pd.Timedelta(days=6),origin)).actual_d
    result.update(actual_lag7=float(lookup.actual_d.get(origin-pd.Timedelta(days=7),np.nan)),
        actual_mean7=float(hist.mean()),actual_std7=float(hist.std(ddof=0)),history_count7=int(hist.notna().sum()),
        plan_revision1=result['plan_d3_origin']-result['plan_d4_lag1'],
        plan_revision2=result['plan_d4_lag1']-result['plan_d5_lag2'],
        plan_revision_rate=(result['plan_d3_origin']-result['plan_d4_lag1'])/(1+abs(result['plan_d4_lag1'])),
        actual_change1=result['actual_d_origin']-result['actual_d_lag1'],
        actual_change2=result['actual_d_lag1']-result['actual_d_lag2'])
    result['Last Value']=result['actual_d_origin']
    result['3-day Moving Average']=float(rows.actual_d.mean())
    result['7-day Moving Average']=float(hist.mean())
    result['D+3 Plan Reference']=result['plan_d3_origin']
    return result


def build_samples(daily):
    rows=[]
    for part,g in daily.groupby('part_number',sort=True):
        g=g.sort_values('date'); target=g.set_index('date').actual_d
        for origin in g.date:
            r=feature_record(g,origin)
            if r is not None and r['target_date'] in target.index:
                r['target']=float(target.loc[r['target_date']]);rows.append(r)
    return pd.DataFrame(rows).sort_values(['target_date','part_number']).reset_index(drop=True)


def split_for_dates(frame,test_dates,validation_days=5):
    dates=pd.DatetimeIndex(frame.target_date.unique()).sort_values().to_numpy(dtype='datetime64[ns]')
    test_dates=pd.to_datetime(test_dates)
    test_origin=test_dates.min()-pd.Timedelta(days=3)
    val_pool=dates[dates<test_origin.to_datetime64()]
    val_dates=val_pool[-validation_days:]
    val_origin=pd.Timestamp(val_dates[0])-pd.Timedelta(days=3)
    train=frame.target_date.lt(val_origin).to_numpy()
    val=frame.target_date.isin(val_dates).to_numpy()
    test=frame.target_date.isin(test_dates).to_numpy()
    # Refit after hyperparameters/epochs freeze; all labels strictly before test origin.
    refit=frame.target_date.lt(test_origin).to_numpy()
    if len(val_dates)!=validation_days or frame.loc[train,'target_date'].nunique()<7:
        raise ValueError('Insufficient history for this split')
    assert frame.loc[train,'target_date'].max()<frame.loc[val,'origin_date'].min()
    assert frame.loc[val,'target_date'].max()<frame.loc[test,'origin_date'].min()
    assert frame.loc[refit,'target_date'].max()<frame.loc[test,'origin_date'].min()
    return train,val,test,refit


def evaluation_plan(frame):
    dates=pd.DatetimeIndex(frame.target_date.unique()).sort_values().to_numpy(dtype='datetime64[ns]')
    holdout=dates[-7:]
    available=dates[dates<(pd.Timestamp(holdout[0])-pd.Timedelta(days=3)).to_datetime64()]
    cv=np.array_split(available[-12:],3)
    return [(f'fold_{i+1}',split_for_dates(frame,d)) for i,d in enumerate(cv)]+[('retrospective',split_for_dates(frame,holdout))]


def metrics(actual,predicted):
    y=np.asarray(actual,float);p=np.asarray(predicted,float);e=p-y;den=np.abs(y).sum()
    variance=np.square(y-y.mean()).sum()
    return {'MAE':float(np.abs(e).mean()),'RMSE':float(np.sqrt(np.square(e).mean())),
        'WAPE_pct':float(np.abs(e).sum()/den*100) if den else np.nan,
        'Forecast_Bias':float(e.mean()),'R2':float(1-np.square(e).sum()/variance) if variance else np.nan}
