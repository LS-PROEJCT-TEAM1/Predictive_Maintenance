"""이상 탐지 탭의 데이터 준비·필터·지표 계산 (화면 코드 없음)."""

import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)

from data_source import load

ALL_FILES = "전체 테스트"
ALL = "전체"

# 모듈 판정 (module_judgement.judgement 값)
GOOD = "정상"
RECHECK = "재검사 권고"
SUSPECT = "불량 의심"
JUDGE_ORDER = [SUSPECT, RECHECK, GOOD]          # 표 정렬 순서
JUDGE_LABEL = {GOOD: "양품", RECHECK: RECHECK, SUSPECT: SUSPECT}
DEFAULT_JUDGES = [RECHECK, SUSPECT]

PAGE_MIN, PAGE_MAX = 1, 39


def short_name(file_name):
    """WeldingTest_04_NG → 04_NG"""
    return str(file_name).replace("WeldingTest_", "")


# ---------------------------------------------------------------- 데이터 읽기

def get_predictions():
    df = load("predictions").copy()
    df["WorkingTime"] = pd.to_datetime(df["WorkingTime"])
    df = df.sort_values(["file", "row_index"]).reset_index(drop=True)

    # 이상 이벤트 = 파일 안에서 label==1 이 연속된 구간 하나
    is_label = df["label"] == 1
    prev_label = is_label.groupby(df["file"]).shift(fill_value=False)
    event_start = is_label & ~prev_label
    df["event_id"] = event_start.groupby(df["file"]).cumsum().where(is_label)
    return df


def get_modules():
    df = load("module_judgement").copy()
    df["start_time"] = pd.to_datetime(df["start_time"])
    return df


def get_metrics_by_file():
    return load("metrics_by_file")


def get_threshold_sweep():
    df = load("threshold_sweep").copy()
    return df.sort_values("calibration_quantile").reset_index(drop=True)


def get_default_quantile():
    sweep = get_threshold_sweep()
    return float(sweep.loc[sweep["is_default"] == 1, "calibration_quantile"].iloc[0])


def get_file_options():
    files = sorted(get_predictions()["file"].unique())
    return [{"label": ALL_FILES, "value": ALL_FILES}] + [
        {"label": short_name(f), "value": f} for f in files
    ]


def get_setpower_options():
    df = get_predictions()
    unseen = set(df.loc[df["unseen_setpower"] == 1, "SetPower"])
    options = [{"label": ALL, "value": ALL}]
    for value in sorted(df["SetPower"].unique()):
        label = f"{value}% (학습외)" if value in unseen else f"{value}%"
        options.append({"label": label, "value": int(value)})
    return options


def get_date_range():
    times = get_predictions()["WorkingTime"]
    return times.min().date(), times.max().date()


# ---------------------------------------------------------------- 필터

def filter_by_date(df, time_column, start_date, end_date):
    if start_date:
        df = df[df[time_column] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df[time_column] < pd.to_datetime(end_date) + pd.Timedelta(days=1)]
    return df


def filter_predictions(df, selected_file, start_date, end_date, selected_setpower, page_range):
    filtered_df = df
    if selected_file != ALL_FILES:
        filtered_df = filtered_df[filtered_df["file"] == selected_file]
    if selected_setpower != ALL:
        filtered_df = filtered_df[filtered_df["SetPower"] == selected_setpower]
    filtered_df = filtered_df[filtered_df["PageNo"].between(page_range[0], page_range[1])]
    return filter_by_date(filtered_df, "WorkingTime", start_date, end_date)


def is_row_filtered(start_date, end_date, selected_setpower, page_range):
    """파일 외의 필터(기간·SetPower·PageNo)가 기본값에서 바뀌었는지."""
    min_date, max_date = get_date_range()
    return (
        str(start_date)[:10] != str(min_date)
        or str(end_date)[:10] != str(max_date)
        or selected_setpower != ALL
        or list(page_range) != [PAGE_MIN, PAGE_MAX]
    )


def filter_modules(df, selected_file, start_date, end_date, search_text):
    filtered_df = df
    if selected_file != ALL_FILES:
        filtered_df = filtered_df[filtered_df["file"] == selected_file]
    filtered_df = filter_by_date(filtered_df, "start_time", start_date, end_date)

    search_digits = "".join(ch for ch in str(search_text or "") if ch.isdigit())
    if search_digits:
        filtered_df = filtered_df[filtered_df["cycle_id"] == int(search_digits)]
    return filtered_df


def filter_judges(df, selected_judges):
    if not selected_judges or ALL in selected_judges:
        return df
    return df[df["judgement"].isin(selected_judges)]


def sort_modules(df):
    order = {judge: i for i, judge in enumerate(JUDGE_ORDER)}
    return (
        df.assign(_order=df["judgement"].map(order))
        .sort_values(["_order", "file", "cycle_id"])
        .drop(columns="_order")
    )


# ---------------------------------------------------------------- 지표

def compute_metrics(df):
    """필터된 predictions 로 판정 지표를 다시 계산한다 (label 은 검증에만 사용)."""
    y_true = df["label"].astype(int)
    y_pred = df["anomaly_pred"].astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    detected_events = df.loc[y_pred == 1, ["file", "event_id"]].dropna().drop_duplicates()
    all_events = df[["file", "event_id"]].dropna().drop_duplicates()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "event_detected": len(detected_events),
        "event_total": len(all_events),
    }


def row_to_metrics(row):
    keys = ["accuracy", "precision", "recall", "f1", "fpr",
            "tn", "fp", "fn", "tp", "event_detected", "event_total"]
    return {key: row[key] for key in keys}


def get_metrics(filtered_pred, selected_file, applied_quantile, row_filtered):
    """지표 카드·혼동행렬에 쓸 값과 출처 문구를 돌려준다.

    - 임계값을 기본값에서 바꿨으면: threshold_sweep 해당 행 (전체 테스트 기준)
    - 파일만 골랐으면: metrics_by_file 해당 행
    - 기간·SetPower·PageNo 로 행을 좁혔으면: predictions 에서 다시 계산
    """
    if applied_quantile != get_default_quantile():
        sweep = get_threshold_sweep()
        row = sweep[sweep["calibration_quantile"] == applied_quantile].iloc[0]
        note = (f"임계값 시뮬레이션 {applied_quantile * 100:g}% = {row['threshold']:.2f} "
                f"· threshold_sweep 전체 테스트 기준")
        return row_to_metrics(row), note

    if not row_filtered:
        by_file = get_metrics_by_file()
        row = by_file[by_file["file"] == selected_file].iloc[0]
        return row_to_metrics(row), None

    if filtered_pred.empty:
        return None, None
    return compute_metrics(filtered_pred), "기간·SetPower·PageNo 필터 적용 · predictions 에서 다시 계산"


# ---------------------------------------------------------------- 모듈

def zero_power_pages(pred_df):
    """(file, cycle_id) → RealPower 0 W 인 PageNo 집합"""
    zero_df = pred_df[pred_df["RealPower"] == 0]
    return zero_df.groupby(["file", "cycle_id"])["PageNo"].apply(set).to_dict()


def parse_pages(value):
    """anomaly_page_nos("2,3,9" 또는 숫자·빈값) → PageNo 집합"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    return {int(float(x)) for x in str(value).split(",") if x.strip()}


def module_id(file_name, cycle_id):
    return f"{file_name}|{int(cycle_id)}"


def judge_counts(df):
    return df["judgement"].value_counts().to_dict()


# ---------------------------------------------------------------- 공정 조건별 이상 비율

def condition_ratio(filtered_pred):
    """파일 × SetPower 별 이상 행 ÷ 조건 행. 전 조건 이상 0 인 파일은 뺀다."""
    summary = (
        filtered_pred.groupby(["file", "SetPower"])["anomaly_pred"]
        .agg(anomaly="sum", total="count")
        .reset_index()
    )
    anomaly_files = summary.groupby("file")["anomaly"].sum()
    keep_files = anomaly_files[anomaly_files > 0].index
    summary = summary[summary["file"].isin(keep_files)]
    if summary.empty:
        return summary

    # 행이 없는 조합도 "–" 로 보여주기 위해 전체 조합을 만든다
    full_index = pd.MultiIndex.from_product(
        [sorted(keep_files), sorted(filtered_pred["SetPower"].unique())],
        names=["file", "SetPower"],
    )
    summary = summary.set_index(["file", "SetPower"]).reindex(full_index).reset_index()
    summary["ratio"] = summary["anomaly"] / summary["total"]
    summary["file_short"] = summary["file"].map(short_name)
    return summary
