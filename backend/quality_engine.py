"""Fixed PCA inference for completed tests. Never fit on uploaded measurements."""
import csv
import hashlib
import io
import json
from functools import lru_cache

import numpy as np
import pandas as pd
from backend.runtime_assets import RUNTIME

CV = [f'M{m:02d}CV{c:02d}' for m in range(1, 17) for c in range(1, 12)]
TP = [f'M{m:02d}T{c:02d}' for m in range(1, 17) for c in range(1, 3)]
FEATURES = ['셀전압_편차', '셀전압_표준편차', '이상셀_개수', '셀전압_최대z', '모듈온도_편차',
            '모듈온도_표준편차', '셀전압평균_변화율', '셀전압편차_이동평균잔차', '모듈온도최대_변화율']
MAX_BYTES = 30_000_000


def clean_frame(frame):
    data = frame[CV + TP].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    for columns, lo, hi in [(CV, 2, 5), (TP, -20, 100)]:
        data[columns] = data[columns].where(data[columns].ge(lo) & data[columns].le(hi))
    data = data.interpolate(limit_direction='both').ffill().bfill()
    for columns in (CV, TP):
        good = [c for c in columns if data[c].notna().any()]
        if not good:
            raise ValueError('전압 또는 온도 전체에 보정 가능한 값이 없습니다.')
        for c in set(columns)-set(good):
            data[c] = data[good].median(axis=1)
    return data


def features(data):
    v, t = data[CV].to_numpy(), data[TP].to_numpy()
    mean, std = v.mean(axis=1), v.std(axis=1)
    z = np.abs(v-mean[:, None])/np.where(std[:, None] == 0, 1e-9, std[:, None])
    spread = pd.Series(np.ptp(v, axis=1))
    return pd.DataFrame({
        FEATURES[0]: spread, FEATURES[1]: std, FEATURES[2]: (z > 3).sum(axis=1), FEATURES[3]: z.max(axis=1),
        FEATURES[4]: np.ptp(t, axis=1), FEATURES[5]: t.std(axis=1),
        FEATURES[6]: pd.Series(mean).diff().fillna(0),
        FEATURES[7]: spread-spread.rolling(10, min_periods=1).mean(),
        FEATURES[8]: pd.Series(t.max(axis=1)).diff().fillna(0)})


@lru_cache(maxsize=1)
def model():
    return json.loads((RUNTIME/'quality/pca_frozen.json').read_text(encoding='utf-8'))


def scores(f, parameters=None):
    m = parameters or model()
    x = (f[FEATURES].to_numpy()-m['mean'])/m['scale']
    centered = x-m['pcaMean']
    projection = centered @ np.array(m['components']).T
    t2 = np.sum(projection**2 / m['variance'], axis=1)
    spe = np.sum((x-(projection @ np.array(m['components'])+m['pcaMean']))**2, axis=1)
    above = (t2 > m['t2']) | (spe > m['spe'])
    # Existing offline policy backfills the whole run once k consecutive rows are seen.
    ends = np.flatnonzero(pd.Series(above).rolling(m['k'], min_periods=m['k']).min().fillna(0).to_numpy())
    prediction = np.zeros(len(f), dtype=int)
    for end in ends:
        prediction[end-m['k']+1:end+1] = 1
    return t2, spe, prediction


def confusion(label, prediction):
    y, p = np.asarray(label), np.asarray(prediction)
    tn, fp, fn, tp = (int(((y == a) & (p == b)).sum()) for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)])
    return {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
            'Precision': tp/(tp+fp) if tp+fp else None, 'Recall': tp/(tp+fn) if tp+fn else None,
            'F1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'FPR': fp/(fp+tn) if fp+tn else None}


def read_csv(payload):
    if not payload or len(payload) > MAX_BYTES:
        raise ValueError('CSV는 30MB 이하여야 합니다.')
    try:
        text = payload.decode('utf-8-sig')
        header = next(csv.reader(io.StringIO(text)))
        if len(set(header)) != len(header):
            raise ValueError('중복 열 이름이 있습니다.')
        frame = pd.read_csv(io.StringIO(text), nrows=20001)
    except (UnicodeError, pd.errors.ParserError, StopIteration) as exc:
        raise ValueError('UTF-8 CSV 형식을 확인하세요.') from exc
    if not 10 <= len(frame) <= 20000:
        raise ValueError('완료된 시험 10~20,000행을 올려 주세요.')
    return frame


def inspect(payload, mode, label_payload=None):
    if mode not in ('chg', 'dchg'):
        raise ValueError('충전 또는 방전을 직접 선택하세요.')
    frame = read_csv(payload)
    missing = [c for c in CV+TP+['Date', 'Time', 'SerialNumber'] if c not in frame]
    if missing:
        raise ValueError('필수 열 누락: '+', '.join(missing[:8]))
    time = pd.to_datetime(frame['Date'].astype(str)+' '+frame['Time'].astype(str), errors='coerce')
    if time.isna().any() or time.duplicated().any() or not time.is_monotonic_increasing:
        raise ValueError('Date·Time은 유효하고 중복 없이 시간순이어야 합니다. 행을 자동 삭제하거나 재정렬하지 않습니다.')
    if frame['SerialNumber'].isna().any() or frame['SerialNumber'].nunique() != 1:
        raise ValueError('하나의 SerialNumber에 해당하는 시험만 올려 주세요.')
    numeric = frame[CV+TP].apply(pd.to_numeric, errors='coerce')
    finite = np.isfinite(numeric)
    invalid = ~finite | (numeric[CV+TP] < ([2]*176+[-20]*32)) | (numeric[CV+TP] > ([5]*176+[100]*32))
    if invalid.to_numpy().mean() > .2:
        raise ValueError('결측·범위 이탈이 전체 측정값의 20%를 넘습니다. 원본을 확인하세요.')
    clean = clean_frame(frame)
    digest = hashlib.sha256(payload).hexdigest()
    labels = None
    if label_payload is not None:
        lab = read_csv(label_payload)
        if not {'sourceRow', 'sourceSha256', 'label'} <= set(lab):
            raise ValueError('라벨 CSV에는 sourceRow, sourceSha256, label 열이 필요합니다.')
        ids = pd.to_numeric(lab['sourceRow'], errors='coerce')
        vals = pd.to_numeric(lab['label'], errors='coerce')
        if len(lab) != len(frame) or ids.duplicated().any() or set(ids) != set(range(len(frame))):
            raise ValueError('라벨 sourceRow가 원본의 모든 0-based 행 ID와 일대일로 대응해야 합니다.')
        if not lab['sourceSha256'].eq(digest).all() or not vals.isin([0, 1]).all():
            raise ValueError('라벨의 원본 SHA256 또는 0/1 값을 확인하세요.')
        labels = pd.Series(vals.to_numpy(), index=ids.astype(int)).sort_index().to_numpy(dtype=int)
    report = [
        {'항목': '입력 구조', '결과': '통과', '설명': f'{len(frame):,}행 · {len(frame.columns)}열 · 전압 176 / 온도 32'},
        {'항목': '시간·팩', '결과': '통과', '설명': '단일 팩 · 시간순 · 중복 시점 없음'},
        {'항목': '공정', '결과': '직접 선택', '설명': '충전' if mode == 'chg' else '방전'},
        {'항목': '원본 결측·비유한값', '결과': '확인' if (~finite).to_numpy().any() else '통과', '설명': f'{int((~finite).to_numpy().sum())}개 측정값'},
        {'항목': '물리 범위 이탈', '결과': '확인' if invalid.to_numpy().any() else '통과', '설명': f'{int((invalid & finite).to_numpy().sum())}개 측정값 · 원본 별도 보존'},
        {'항목': '시간 간격', '결과': '확인' if time.diff().dt.total_seconds().dropna().ne(1).any() else '통과', '설명': '1초 간격 외 입력은 학습 조건과 다를 수 있음'},
        {'항목': '정답 연결', '결과': '해시·행 ID 검증' if labels is not None else '미제공', '설명': '정답은 평가에만 사용 · 모델 입력에서 제외'},
    ]
    before_after = [{'항목': '행 수', '처리 전': len(frame), '처리 후': len(clean)},
                    {'항목': '입력 열 수', '처리 전': len(frame.columns), '처리 후': 208},
                    {'항목': '결측·범위 이탈 측정값', '처리 전': int(invalid.to_numpy().sum()), '처리 후': 0},
                    {'항목': '전체 구간 보정 채널', '처리 전': int(invalid.all().sum()), '처리 후': int(invalid.all().sum())}]
    sample_ids = np.unique(np.linspace(0, len(frame)-1, min(8, len(frame)), dtype=int)).tolist()
    sample_ids += np.flatnonzero(invalid.any(axis=1))[:8].tolist()
    changes = []
    for i in sorted(set(sample_ids)):
        columns = list(invalid.columns[invalid.iloc[i]])[:10] or [CV[0]]
        for c in columns:
            value = numeric.iloc[i][c]
            changes.append({'행 ID': int(i), '채널': c, '원본': float(value) if np.isfinite(value) else None, '보정': float(clean.iloc[i][c])})
    return {'frame': clean, 'labels': labels, 'mode': mode,
            'public': {'sha256': digest, 'rows': len(frame), 'checks': report, 'beforeAfter': before_after, 'changes': changes,
                       'features': FEATURES, 'notice': '완료 시험의 앞뒤 보간을 포함한 사후 분석입니다. 원본·공식 시드는 변경하지 않습니다.'}}


def analyze(item):
    frame = item['frame']
    t2, spe, prediction = scores(features(frame))
    ids = np.unique(np.linspace(0, len(frame)-1, min(500, len(frame)), dtype=int))
    m = model()
    return {'modelVersion': m['version'], 'rows': len(frame), 'abnormalRows': int(prediction.sum()),
            'thresholds': {'t2': m['t2'], 'spe': m['spe'], 'k': m['k']},
            'confusion': confusion(item['labels'], prediction) if item['labels'] is not None else None,
            'points': [{'index': int(i), 't2': float(t2[i]), 'spe': float(spe[i]), 'prediction': int(prediction[i])} for i in ids],
            'results': [{'sourceRow': i, 'T2': float(a), 'SPE': float(b), 'prediction': int(p),
                         'label': int(item['labels'][i]) if item['labels'] is not None else None} for i, (a, b, p) in enumerate(zip(t2, spe, prediction))],
            'notice': '고정 모델 · 임계값 변경 없음 · 원인 4종 분류 아님 · 공식 DB 미저장'}
