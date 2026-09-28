"""Shared sensor contract. Invalid input is never a normal process prediction."""
import numpy as np
import pandas as pd

RAW_COLUMNS = ['PageNo', 'Speed', 'Length', 'RealPower', 'SetFrequency',
               'SetDuty', 'SetPower', 'GateOnTime', 'WorkingTime']


def validate_signals(frame):
    work = frame.copy()
    work.columns = work.columns.str.strip()
    if work.columns.duplicated().any():
        raise ValueError('입력 오류: 중복된 열 이름이 있습니다.')
    missing = sorted(set(RAW_COLUMNS) - set(work.columns))
    if missing or work.empty:
        raise ValueError(f'입력 오류: 필수 열 누락 또는 빈 데이터입니다. {missing}')
    for col in RAW_COLUMNS[:-1]:
        values = pd.to_numeric(work[col], errors='coerce')
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f'입력 오류: {col}에 결측·무한대·숫자가 아닌 값이 있습니다.')
        if (values < 0).any():
            raise ValueError(f'입력 오류: {col}에 음수가 있습니다.')
        work[col] = values
    page = work.PageNo
    if not ((page == np.floor(page)) & page.between(1, 39)).all():
        raise ValueError('입력 오류: PageNo는 1~39의 정수여야 합니다.')
    if not work.SetPower.between(0, 100).all() or not work.SetDuty.between(0, 100).all():
        raise ValueError('입력 오류: SetPower·SetDuty는 0~100%여야 합니다.')
    try:
        dates = pd.to_datetime(work.WorkingTime, format='ISO8601', errors='raise')
    except (ValueError, TypeError) as exc:
        raise ValueError('입력 오류: WorkingTime 날짜 형식을 확인하세요.') from exc
    if dates.isna().any():
        raise ValueError('입력 오류: WorkingTime이 비어 있습니다.')
    work['WorkingTime'] = dates
    return work


def validate_sequence(frame):
    """New CSV inference accepts complete, strictly ordered welding cycles."""
    work = validate_signals(frame)
    if len(work) % 39 or work.PageNo.tolist() != list(range(1, 40)) * (len(work) // 39):
        raise ValueError('입력 오류: PageNo 1~39의 완전한 용접 사이클 순서가 필요합니다.')
    if (work.WorkingTime.diff().dt.total_seconds().dropna() <= 0).any():
        raise ValueError('입력 오류: WorkingTime 중복 또는 시간 역전이 있습니다.')
    return work
