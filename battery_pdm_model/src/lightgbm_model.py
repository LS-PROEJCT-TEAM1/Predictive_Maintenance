"""
배터리 레이저 용접 공정 이상 탐지 - LightGBM 정상 출력 회귀 모델 + 대시보드 연동 산출물 생성

final_three_model_comparison.py 의 3개 모델(RobustZ / LightGBM / N-HiTS) 공통 비교 중
LightGBM 부분을 추출한 파일이다. (compare_models.py 의 LightGBMDetector 와 관련 함수 포함)
대시보드는 LightGBM 모델 하나만 사용한다.
torch / N-HiTS 없이 lightgbm 만으로 실행된다.

흐름
  1) 데이터 : Training_Data.csv 를 39행(PageNo 1~39) 사이클 단위로 묶고,
             시간 순서 앞 70% = 학습, 뒤 30% = 정상 보정(calibration) 으로 분할
  2) 회귀   : 공정 조건 + 직전 출력 등 9개 Feature → RealPower 를 LGBMRegressor(huber, alpha=50) 로 예측
             (원본은 alpha 기본값 0.9 → 회귀가 학습되지 않아 alpha=50 으로 수정)
  3) 점수   : |잔차| 를 PageNo 별 중앙값/robust scale 로 표준화한 값 = 이상 점수
  4) 임계값 : 정상 보정 데이터 이상 점수의 99.9 백분위 (테스트 라벨은 임계값에 사용하지 않음)
  5) 평가   : 행 단위(Accuracy/Precision/Recall/F1/혼동행렬) + 이벤트 단위(탐지/오경보/지연)
  6) 저장   : outputs/lightgbm 에 대시보드 연동용 CSV 12개 + 점검 이력 템플릿 1개
"""
from __future__ import annotations

import importlib
import json
import xml.etree.ElementTree as ET
import site
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------
# 필요한 패키지가 '지금 실행 중인 파이썬'에 없으면 그 파이썬에 바로 설치
# ---------------------------------------------------------------
def ensure_package(import_name: str, pip_name: str | None = None) -> None:
    try:
        importlib.import_module(import_name)
        return
    except ImportError:
        pass
    pip_name = pip_name or import_name
    print(f"[설치] '{pip_name}' 패키지가 없어 설치합니다 → {sys.executable}")
    install_cmd = [sys.executable, "-m", "pip", "install", pip_name]
    try:
        subprocess.check_call(install_cmd)
    except subprocess.CalledProcessError:
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
        subprocess.check_call(install_cmd)
    user_site = site.getusersitepackages()
    if user_site not in sys.path:
        sys.path.append(user_site)
    importlib.invalidate_caches()
    importlib.import_module(import_name)


for _import_name, _pip_name in [("numpy", None), ("pandas", None), ("sklearn", "scikit-learn"), ("lightgbm", None)]:
    ensure_package(_import_name, _pip_name)

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


# ---------------------------------------------------------------
# 경로 / 설정 (final_three_model_comparison.py, compare_models.py 와 동일)
# ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 배터리 예지보전 모델
DATA_ROOT = PROJECT_ROOT / "Dataset_전자부품(배터리팩) 예지보전 AI 데이터셋" / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "lightgbm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
CALIBRATION_QUANTILE = 0.999
THRESHOLD_LABEL = "normal calibration score 99.9th percentile"
TEST_FILES = ["WeldingTest_01_OK", "WeldingTest_02_OK", "WeldingTest_03_NG", "WeldingTest_04_NG"]
FEATURE_COLUMNS = ["PageNo", "Speed", "Length", "SetPower", "GateOnTime", "PreviousPower", "TimeGap", "PageSin", "PageCos"]


# ===============================================================
# 1. 데이터 읽기 / 사이클 분할
# ===============================================================
def read_signal(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()  # 컬럼명 공백 제거
    frame["WorkingTime"] = pd.to_datetime(frame["WorkingTime"], errors="raise")
    return frame


def add_cycle_id(frame: pd.DataFrame) -> pd.DataFrame:
    # PageNo 가 1 로 돌아올 때마다 새 사이클(배터리모듈 1개 = 39개 용접 포인트)
    result = frame.copy()
    result["cycle_id"] = result["PageNo"].eq(1).cumsum().astype(int)
    print("사이클 아이디 부여 부분")
    print(result)
    return result


def split_all_cycles(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cycled = add_cycle_id(frame)
    cycles = []
    for cycle_id, group in cycled.groupby("cycle_id", sort=False):
        group = group.sort_values("PageNo").copy()
        if len(group) != 39 or group["PageNo"].tolist() != list(range(1, 40)):
            raise ValueError(f"Cycle {cycle_id} is incomplete; guidebook comparison keeps raw rows.")
        cycles.append((group["WorkingTime"].min(), group))
    cycles.sort(key=lambda item: item[0])
    cut = int(len(cycles) * 0.70)
    train = pd.concat([group for _, group in cycles[:cut]], ignore_index=True)
    calibration = pd.concat([group for _, group in cycles[cut:]], ignore_index=True)
    info = {
        "all_complete_cycles_kept": len(cycles),
        "train_cycles": cut,
        "calibration_cycles": len(cycles) - cut,
        "train_rows": len(train),
        "calibration_rows": len(calibration),
        "zero_power_cycles_kept": int(sum(group["RealPower"].eq(0).any() for _, group in cycles)),
        "train_start": str(train["WorkingTime"].min()),
        "train_end": str(train["WorkingTime"].max()),
        "calibration_start": str(calibration["WorkingTime"].min()),
        "calibration_end": str(calibration["WorkingTime"].max()),
    }
    return train, calibration, info


# ===============================================================
# 2. PageNo 별 robust 기준 / 임계값
# ===============================================================
def quantile_threshold(scores: np.ndarray, quantile: float = CALIBRATION_QUANTILE) -> float:
    finite = np.asarray(scores, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, quantile, method="higher"))


def phase_location_scale(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    # PageNo 별 중앙값과 scale = max(1.4826*MAD, 0.25*표준편차, 1.0)
    median = frame.groupby("PageNo")["RealPower"].median()
    mad = frame.groupby("PageNo")["RealPower"].apply(
        lambda values: float(np.median(np.abs(values - np.median(values))))
    )
    robust = 1.4826 * mad
    standard = frame.groupby("PageNo")["RealPower"].std().fillna(0.0)
    scale = pd.concat(
        [robust.rename("mad"), (0.25 * standard).rename("std_floor"), pd.Series(1.0, index=median.index, name="unit_floor")],
        axis=1,
    ).max(axis=1)
    return median.astype(float), scale.astype(float)


# ===============================================================
# 3. Feature 생성
# ===============================================================
def regression_features(frame: pd.DataFrame, phase_median: pd.Series) -> pd.DataFrame:
    work = frame.reset_index(drop=True)
    timestamps = work["WorkingTime"]
    # 직전 행 RealPower (사이클 첫 행(PageNo=1)은 학습 데이터 PageNo 39 중앙값으로 대체)
    previous_power = work["RealPower"].shift(1)
    default_previous = work["PageNo"].sub(1).replace(0, 39).map(phase_median)
    previous_power = previous_power.where(work["PageNo"].ne(1), default_previous)
    previous_power = previous_power.fillna(default_previous).astype(float)
    # 직전 행과의 시간 간격(초, 0~120 으로 제한)
    time_gap = timestamps.diff().dt.total_seconds().clip(lower=0.0, upper=120.0).fillna(0.0)
    # PageNo 를 원형(1~39)으로 표현
    page_angle = 2.0 * np.pi * (work["PageNo"].to_numpy(dtype=float) - 1.0) / 39.0
    return pd.DataFrame(
        {
            "PageNo": work["PageNo"].astype(int),
            "Speed": work["Speed"].astype(float),
            "Length": work["Length"].astype(float),
            "SetPower": work["SetPower"].astype(float),
            "GateOnTime": work["GateOnTime"].astype(float),
            "PreviousPower": previous_power,
            "TimeGap": time_gap,
            "PageSin": np.sin(page_angle),
            "PageCos": np.cos(page_angle),
        }
    )


# ===============================================================
# 4. LightGBM 이상 탐지기
# ===============================================================
class LightGBMDetector:
    name = "LightGBM"

    def __init__(self) -> None:
        self.threshold = float("nan")
        self.training_seconds = float("nan")

    def fit(self, train: pd.DataFrame, calibration: pd.DataFrame) -> "LightGBMDetector":
        started = time.perf_counter()
        self.phase_median, _ = phase_location_scale(train)
        features = regression_features(train, self.phase_median)
        target = train["RealPower"].to_numpy(dtype=float)
        validation_start = int(len(train) * 0.85)  # 학습 구간 뒤 15% 로 early stopping
        self.model = lgb.LGBMRegressor(
            objective="huber",
            n_estimators=500,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=40,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_lambda=1.0,
            alpha=50.0,  # huber 전환 기준(W). 기본값 0.9 는 출력(W) 단위에 비해 너무 작아 회귀가 학습되지 않음
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
        )
        self.model.fit(
            features.iloc[:validation_start],
            target[:validation_start],
            eval_set=[(features.iloc[validation_start:], target[validation_start:])],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        # 학습 데이터 |잔차| 의 PageNo 별 중앙값/scale → 점수 표준화 기준
        train_prediction = self.model.predict(features)
        residual_frame = pd.DataFrame(
            {"PageNo": train["PageNo"].to_numpy(), "RealPower": np.abs(target - train_prediction)}
        )
        self.residual_median, self.residual_scale = phase_location_scale(residual_frame)
        # 정상 보정 데이터 점수의 99.9 백분위 = 임계값
        self.threshold = quantile_threshold(self.score(calibration))
        self.training_seconds = time.perf_counter() - started
        return self

    def predict_power(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict(regression_features(frame, self.phase_median))

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        prediction = self.predict_power(frame)
        residual = np.abs(frame["RealPower"].to_numpy(dtype=float) - prediction)
        center = frame["PageNo"].map(self.residual_median).to_numpy(dtype=float)
        scale = frame["PageNo"].map(self.residual_scale).to_numpy(dtype=float)
        return np.abs(residual - center) / scale

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (self.score(frame) > self.threshold).astype(int)


# ===============================================================
# 5. 평가 함수
# ===============================================================
def calc_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def regions(values: np.ndarray) -> list[tuple[int, int]]:
    # 연속된 1 구간의 (시작, 끝) 인덱스 목록
    values = np.asarray(values, dtype=int)
    padded = np.pad(values, (1, 1))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def event_summary(labels: dict[str, np.ndarray], preds: dict[str, np.ndarray]) -> tuple[int, int, int]:
    total, detected, false_alarm_events = 0, 0, 0
    for name, truth in labels.items():
        prediction = preds[name]
        if truth.sum() == 0:
            false_alarm_events += len(regions(prediction))
        for start, end in regions(truth):
            total += 1
            if prediction[start : end + 1].any():
                detected += 1
        normal_positive = ((truth == 0) & (prediction == 1)).astype(int)
        false_alarm_events += len(regions(normal_positive)) if truth.sum() else 0
    return total, detected, false_alarm_events


# ===============================================================
# 6. 대시보드용 보조 함수
# ===============================================================
RISK_WATCH_RATIO = 0.8  # 위험도(score/threshold) 0.8 이상 = 관찰 필요
DRIFT_WARN_W = -20.0  # 첫 달 대비 20W(2000W 의 1%) 이상 하락 시 주의
REGRESSION_WARN_MAE = 60.0  # 가이드북 허용 한도 60~80W
SWEEP_QUANTILES = [0.990, 0.992, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999, 0.9995, 0.9999]


def save_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUTPUT_DIR / name, index=False, encoding="utf-8-sig")
    print(f"  저장: {name} ({len(frame)}행)")


def event_details(truth: np.ndarray, prediction: np.ndarray) -> dict:
    """파일 1개에 대한 이벤트 단위 지표"""
    events = regions(truth)
    detected, delays = 0, []
    for start, end in events:
        hit = np.flatnonzero(prediction[start : end + 1])
        if len(hit):
            detected += 1
            delays.append(int(hit[0]))
    if truth.sum() == 0:
        false_alarm = len(regions(prediction))
    else:
        false_alarm = len(regions(((truth == 0) & (prediction == 1)).astype(int)))
    return {
        "event_total": len(events),
        "event_detected": detected,
        "missed_events": len(events) - detected,
        "normal_false_alarm_events": false_alarm,
        "mean_detection_delay_rows": float(np.mean(delays)) if delays else np.nan,
        "max_detection_delay_rows": int(max(delays)) if delays else np.nan,
    }


def metric_row(truth: np.ndarray, prediction: np.ndarray) -> dict:
    m = calc_binary_metrics(truth, prediction)
    m["fpr"] = m["fp"] / (m["fp"] + m["tn"]) if (m["fp"] + m["tn"]) else 0.0
    return m


def power_group_medians(train: pd.DataFrame) -> pd.Series:
    return train.groupby("SetPower")["RealPower"].median()


def inspection_reason(row: pd.Series, group_median: pd.Series, known_setpower: set) -> str:
    if row["anomaly_pred"] == 0:
        return ""
    known = np.array(sorted(group_median.index))
    own = known[np.argmin(np.abs(known - row["SetPower"]))]
    closest_group = group_median.index[np.argmin(np.abs(group_median.to_numpy() - row["RealPower"]))]
    if row["RealPower"] <= 0:
        reason = "미출력(0W) - 레이저 발진/게이트 점검"
    elif abs(group_median[closest_group] - group_median[own]) > 200 and closest_group != own:
        reason = "설정-출력 불일치 - 데이터 수집/동기화 확인"
    elif row["RealPower"] < row["page_median_power"]:
        reason = "출력 저하 - 광학계/렌즈 오염, 레이저 소스 점검"
    else:
        reason = "출력 과다 - 출력 제어부/설정값 점검"
    if row["SetPower"] not in known_setpower:
        reason += " + 출력 설정값(레시피) 확인"
    return reason


def build_segments(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for file_name, g in pred.groupby("file", sort=False):
        g = g.reset_index(drop=True)
        for n, (start, end) in enumerate(regions(g["anomaly_pred"].to_numpy()), start=1):
            s = g.iloc[start : end + 1]
            rows.append({
                "segment_id": f"{file_name}_{n:03d}",
                "file": file_name,
                "start_time": s["WorkingTime"].min(),
                "end_time": s["WorkingTime"].max(),
                "start_row": int(s["row_index"].min()),
                "end_row": int(s["row_index"].max()),
                "anomaly_rows": len(s),
                "cycle_ids": ",".join(map(str, sorted(s["cycle_id"].unique()))),
                "page_nos": ",".join(map(str, sorted(s["PageNo"].unique()))),
                "set_powers": ",".join(map(str, sorted(s["SetPower"].unique()))),
                "mean_real_power": round(s["RealPower"].mean(), 1),
                "mean_predicted_power": round(s["predicted_power"].mean(), 1),
                "mean_page_median_power": round(s["page_median_power"].mean(), 1),
                "max_anomaly_score": round(s["anomaly_score"].max(), 3),
                "max_risk_ratio": round(s["risk_ratio"].max(), 3),
                "inspection_reason": s["inspection_reason"].mode().iloc[0],
                "contains_zero_power": bool((s["RealPower"] <= 0).any()),
                "true_anomaly_rows": int(s["label"].sum()),
            })
    seg = pd.DataFrame(rows)
    if seg.empty:
        return seg
    # 같은 파일·같은 PageNo 에서 단발성(3행 미만) 이상이 몇 번 반복됐는지
    single = seg[seg["anomaly_rows"] < 3].copy()
    single["_page"] = single["page_nos"].str.split(",")
    repeat = single.explode("_page").groupby(["file", "_page"]).size()
    seg["same_page_repeat"] = [
        max(repeat.get((f, p), 0) for p in pages.split(",")) if n < 3 else 0
        for f, pages, n in zip(seg["file"], seg["page_nos"], seg["anomaly_rows"])
    ]

    def priority(r):
        if r["anomaly_rows"] >= 39 or r["contains_zero_power"]:
            return "긴급"
        if r["anomaly_rows"] < 3 and r["same_page_repeat"] >= 3:
            return "높음(동일 포인트 반복)"
        if r["anomaly_rows"] >= 3:
            return "높음"
        return "보통"

    seg["priority"] = seg.apply(priority, axis=1)
    order = {"긴급": 0, "높음(동일 포인트 반복)": 1, "높음": 2, "보통": 3}
    seg["priority_order"] = seg["priority"].map(order)
    return seg.sort_values(["priority_order", "start_time"]).reset_index(drop=True)


def build_modules(pred: pd.DataFrame) -> pd.DataFrame:
    grouped = pred.groupby(["file", "cycle_id"], sort=False)
    mod = grouped.agg(
        start_time=("WorkingTime", "min"),
        end_time=("WorkingTime", "max"),
        points=("PageNo", "size"),
        anomaly_points=("anomaly_pred", "sum"),
        max_anomaly_score=("anomaly_score", "max"),
        zero_power_points=("RealPower", lambda v: int((v <= 0).sum())),
        true_anomaly_points=("label", "sum"),
    ).reset_index()
    pages = pred[pred["anomaly_pred"] == 1].groupby(["file", "cycle_id"])["PageNo"].apply(
        lambda v: ",".join(map(str, sorted(v.unique())))
    )
    mod["anomaly_page_nos"] = [pages.get((f, c), "") for f, c in zip(mod["file"], mod["cycle_id"])]

    def judge(r):
        if r["anomaly_points"] >= 3 or r["zero_power_points"] > 0:
            return "불량 의심"
        if r["anomaly_points"] >= 1:
            return "재검사 권고"
        return "정상"

    mod["judgement"] = mod.apply(judge, axis=1)
    mod["max_anomaly_score"] = mod["max_anomaly_score"].round(3)
    return mod


def equipment_info() -> pd.DataFrame:
    aasx_dir = PROJECT_ROOT / "Dataset_전자부품(배터리팩) 예지보전 AI 데이터셋" / "AASX 및 변환파일"
    rows = []
    xml_path = aasx_dir / "BatteryModuleWelding.xml"
    if xml_path.exists():
        strip = lambda tag: tag.split("}")[-1]

        def child(el, name):
            for c in el:
                if strip(c.tag) == name:
                    return c
            return None

        def walk(el, path):
            for c in el:
                tag = strip(c.tag)
                if tag in ("submodel", "submodelElementCollection"):
                    id_short = child(c, "idShort")
                    walk(c, path + [id_short.text if id_short is not None else tag])
                elif tag == "property":
                    id_short, value = child(c, "idShort"), child(c, "value")
                    raw = value.text if value is not None and value.text is not None else ""
                    rows.append({
                        "category": path[0] if path else "",
                        "item": ".".join(path[1:] + [id_short.text]),
                        "value": raw if raw not in ("-", "", "0000-00-00") else "미등록",
                        "source": "BatteryModuleWelding.xml",
                    })
                else:
                    walk(c, path)

        walk(ET.parse(xml_path).getroot(), [])
    cfg_path = aasx_dir / "syscfg.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        for gw in cfg.get("system", []):
            rows.append({"category": "수집 게이트웨이", "item": "GatewayName", "value": gw.get("GatewayName"), "source": "syscfg.json"})
            rows.append({"category": "수집 게이트웨이", "item": "NetworkConnection", "value": gw.get("NetworkConnection"), "source": "syscfg.json"})
            rows.append({"category": "수집 게이트웨이", "item": "SamplingInterval(ms)", "value": gw.get("SamplingInterval"), "source": "syscfg.json"})
            for dev in gw.get("FieldDevices", []):
                rows.append({"category": "수집 게이트웨이", "item": f"FieldDevice.{dev.get('DeviceName')}", "value": dev.get("NetworkConnection"), "source": "syscfg.json"})
    eng_path = aasx_dir / "engineering.csv"
    if eng_path.exists():
        for line in eng_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7 and parts[-1]:
                rows.append({"category": "태그-컬럼 매핑", "item": parts[0].split("s=")[-1], "value": parts[-1], "source": "engineering.csv"})
    frame = pd.DataFrame(rows, columns=["category", "item", "value", "source"])
    # OperationalData 의 값(0)은 실시간 태그 초기값이라 설비 정보에서 제외
    return frame[frame["category"] != "OperationalData"].reset_index(drop=True)


def data_quality(name: str, frame: pd.DataFrame, role: str, label: np.ndarray | None, train_recipes: set, train_setpower: set) -> list[dict]:
    required = ["PageNo", "Speed", "Length", "RealPower", "SetFrequency", "SetDuty", "SetPower", "GateOnTime", "WorkingTime"]
    cycled = add_cycle_id(frame)
    sizes = cycled.groupby("cycle_id")["PageNo"].apply(list)
    incomplete = int(sum(1 for pages in sizes if pages != list(range(1, 40))))
    numeric = frame[[c for c in required if c != "WorkingTime"]]
    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
    iqr = q3 - q1
    iqr_rows = int(((numeric < q1 - 4 * iqr) | (numeric > q3 + 4 * iqr)).any(axis=1).sum())
    constant_cols = [c for c in ["SetFrequency", "SetDuty"] if frame[c].nunique() <= 1]
    recipes = set(map(tuple, frame[["Speed", "Length", "SetPower"]].drop_duplicates().to_numpy().tolist()))
    unseen_recipe_rows = int((~frame[["Speed", "Length", "SetPower"]].apply(tuple, axis=1).isin(train_recipes)).sum()) if role == "test" else 0
    unseen_setpower_rows = int((~frame["SetPower"].isin(train_setpower)).sum()) if role == "test" else 0
    missing_cols = [c for c in required if c not in frame.columns]
    checks = [
        ("필수 컬럼", "정상" if not missing_cols else "오류", len(missing_cols), "누락: " + ",".join(missing_cols) if missing_cols else "9개 컬럼 모두 있음"),
        ("WorkingTime 날짜 형식", "정상", 0, f"{frame['WorkingTime'].min()} ~ {frame['WorkingTime'].max()}"),
        ("전체 행 수", "정상", len(frame), ""),
        ("사이클(모듈) 수", "정상", int(cycled["cycle_id"].nunique()), "PageNo 1~39 = 배터리모듈 1개"),
        ("불완전 사이클", "정상" if incomplete == 0 else "경고", incomplete, "39행·PageNo 순서가 맞지 않는 사이클"),
        ("결측치", "정상" if frame.isna().sum().sum() == 0 else "경고", int(frame.isna().sum().sum()), ""),
        ("중복 행", "정상" if frame.duplicated().sum() == 0 else "경고", int(frame.duplicated().sum()), ""),
        ("IQR 4배 이상치 행", "정상" if iqr_rows == 0 else "경고", iqr_rows, "제거하지 않고 기록만 함"),
        ("단일값 컬럼", "정상", len(constant_cols), ",".join(constant_cols) + " (분석 변수에서 제외)"),
        ("RealPower=0 행", "정상" if (frame["RealPower"] <= 0).sum() == 0 else "경고", int((frame["RealPower"] <= 0).sum()), "미출력"),
        ("레시피(Speed·Length·SetPower) 종류", "정상", len(recipes), ""),
    ]
    if role == "test":
        checks += [
            ("학습에 없는 SetPower 행", "정상" if unseen_setpower_rows == 0 else "경고", unseen_setpower_rows, "예: 35%"),
            ("학습에 없는 레시피 행", "정상" if unseen_recipe_rows == 0 else "경고", unseen_recipe_rows, ""),
        ]
    if label is not None:
        ok = len(label) == len(frame)
        checks.append(("라벨 행 수 일치", "정상" if ok else "오류", int(len(label)), f"라벨 이상 행 {int(label.sum())}개"))
    return [{"file": name, "role": role, "check_item": c, "result": r, "count": n, "description": d} for c, r, n, d in checks]


def condition_summary(frames: dict[str, pd.DataFrame], pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, frame in frames.items():
        work = frame.copy()
        work["Recipe"] = "S" + work["Speed"].astype(str) + "_L" + work["Length"].astype(str) + "_P" + work["SetPower"].astype(str)
        for group_type in ["SetPower", "Recipe", "PageNo"]:
            stats = work.groupby(group_type)["RealPower"].agg(["count", "mean", "std", "min", "median", "max"]).reset_index()
            stats = stats.rename(columns={group_type: "group_value"})
            stats.insert(0, "group_type", group_type)
            stats.insert(0, "dataset", dataset)
            if dataset != "Training_Data":
                p = pred[pred["file"] == dataset].copy()
                p["Recipe"] = "S" + p["Speed"].astype(str) + "_L" + p["Length"].astype(str) + "_P" + p["SetPower"].astype(str)
                cnt = p.groupby(group_type)["anomaly_pred"].sum()
                stats["anomaly_count"] = stats["group_value"].map(cnt).fillna(0).astype(int)
                lab = p.groupby(group_type)["label"].sum()
                stats["label_anomaly_count"] = stats["group_value"].map(lab).fillna(0).astype(int)
            rows.append(stats)
    out = pd.concat(rows, ignore_index=True)
    return out.round(3)


# ===============================================================
# 7. 실행
# ===============================================================
def main() -> None:
    np.random.seed(SEED)

    full_train = read_signal(DATA_ROOT / "raw_data" / "train" / "Training_Data.csv")
    train, calibration, split_info = split_all_cycles(full_train)
    tests = {name: read_signal(DATA_ROOT / "raw_data" / "test" / f"{name}.csv") for name in TEST_FILES}
    labels = {
        name: (
            pd.read_csv(DATA_ROOT / "preprocessed" / "test" / f"{name}_Label.csv")["label"].to_numpy(dtype=int)
            if name.endswith("NG")
            else np.zeros(len(tests[name]), dtype=int)
        )
        for name in TEST_FILES
    }

    # ---------------- 모델 학습 ----------------
    lightgbm = LightGBMDetector().fit(train, calibration)
    print(f"LightGBM 학습 완료: best_iteration={lightgbm.model.best_iteration_}, 임계값={lightgbm.threshold:.4f}")

    group_median = power_group_medians(train)
    known_setpower = set(train["SetPower"].unique())
    train_recipes = set(map(tuple, full_train[["Speed", "Length", "SetPower"]].drop_duplicates().to_numpy().tolist()))

    # ---------------- 2. predictions.csv ----------------
    pred_frames, preds = [], {}
    started, n_rows = time.perf_counter(), 0
    for name, frame in tests.items():
        predicted = lightgbm.predict_power(frame)
        score = lightgbm.score(frame)
        n_rows += len(frame)
        p = (score > lightgbm.threshold).astype(int)
        preds[name] = p
        out = add_cycle_id(frame)[["WorkingTime", "cycle_id", "PageNo", "Speed", "Length", "SetPower", "GateOnTime", "RealPower"]].copy()
        out.insert(0, "row_index", np.arange(len(frame)))
        out.insert(0, "file", name)
        out["predicted_power"] = np.round(predicted, 3)  # LightGBM 예측 출력
        out["page_median_power"] = out["PageNo"].map(lightgbm.phase_median).round(3)  # 학습 PageNo 중앙값(참고 기준선)
        out["abs_residual"] = np.round(np.abs(out["RealPower"] - predicted), 3)
        out["anomaly_score"] = np.round(score, 4)
        out["threshold"] = round(lightgbm.threshold, 4)
        out["risk_ratio"] = np.round(score / lightgbm.threshold, 4)
        out["risk_level"] = np.where(out["risk_ratio"] >= 1, "이상", np.where(out["risk_ratio"] >= RISK_WATCH_RATIO, "관찰", "정상"))
        out["anomaly_pred"] = p
        out["unseen_setpower"] = (~out["SetPower"].isin(known_setpower)).astype(int)
        out["label"] = labels[name]
        pred_frames.append(out)
    inference_ms = (time.perf_counter() - started) / n_rows * 1000 * 1000
    pred = pd.concat(pred_frames, ignore_index=True)
    pred["inspection_reason"] = pred.apply(inspection_reason, axis=1, group_median=group_median, known_setpower=known_setpower)

    # ---------------- 3~4. 점검 목록, 모듈 판정 ----------------
    segments = build_segments(pred)
    modules = build_modules(pred)

    # ---------------- 5. metrics_by_file.csv ----------------
    metric_rows = [{"file": n, **metric_row(labels[n], preds[n]), **event_details(labels[n], preds[n])} for n in TEST_FILES]
    y_all = np.concatenate([labels[n] for n in TEST_FILES])
    p_all = np.concatenate([preds[n] for n in TEST_FILES])
    per_file = [event_details(labels[n], preds[n]) for n in TEST_FILES]
    delays = [d["mean_detection_delay_rows"] for d in per_file if not pd.isna(d["mean_detection_delay_rows"])]
    metric_rows.append({"file": "전체 테스트", **metric_row(y_all, p_all),
                        "event_total": sum(d["event_total"] for d in per_file),
                        "event_detected": sum(d["event_detected"] for d in per_file),
                        "missed_events": sum(d["missed_events"] for d in per_file),
                        "normal_false_alarm_events": sum(d["normal_false_alarm_events"] for d in per_file),
                        "mean_detection_delay_rows": float(np.mean(delays)) if delays else np.nan,
                        "max_detection_delay_rows": max((d["max_detection_delay_rows"] for d in per_file
                                                         if not pd.isna(d["max_detection_delay_rows"])), default=np.nan)})
    metrics = pd.DataFrame(metric_rows)
    f1_03 = metrics.loc[metrics["file"] == "WeldingTest_03_NG", "f1"].iloc[0]
    f1_04 = metrics.loc[metrics["file"] == "WeldingTest_04_NG", "f1"].iloc[0]
    metrics["macro_f1_03_04"] = np.where(metrics["file"] == "전체 테스트", (f1_03 + f1_04) / 2, np.nan)
    metrics["threshold"] = lightgbm.threshold
    metrics["inference_ms_per_1000_rows"] = np.where(metrics["file"] == "전체 테스트", inference_ms, np.nan)
    metrics = metrics.round(4)

    # ---------------- 6. regression_metrics.csv ----------------
    reg_rows = []
    for name in TEST_FILES + ["전체 테스트"]:
        p = pred if name == "전체 테스트" else pred[pred["file"] == name]
        for subset, q in [("정상 행", p[p["label"] == 0]), ("전체 행", p)]:
            y, yhat = q["RealPower"].to_numpy(float), q["predicted_power"].to_numpy(float)
            mae = float(np.mean(np.abs(y - yhat)))
            reg_rows.append({
                "file": name, "subset": subset, "rows": len(q),
                "MAE": mae, "RMSE": float(np.sqrt(np.mean((y - yhat) ** 2))),
                "R2": float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)) if np.var(y) > 0 else np.nan,
                "accuracy_standard_w": 20.0, "tolerance_limit_w": REGRESSION_WARN_MAE,
                "retrain_warning": "재학습 필요" if subset == "정상 행" and mae > REGRESSION_WARN_MAE else "",
            })
    regression = pd.DataFrame(reg_rows).round(4)

    # ---------------- 7. threshold_sweep.csv ----------------
    calibration_scores = lightgbm.score(calibration)
    test_scores = {n: lightgbm.score(tests[n]) for n in TEST_FILES}
    sweep_rows = []
    for q in SWEEP_QUANTILES:
        thr = quantile_threshold(calibration_scores, q)
        p_by_file = {n: (test_scores[n] > thr).astype(int) for n in TEST_FILES}
        p_all_q = np.concatenate([p_by_file[n] for n in TEST_FILES])
        total, detected, false_alarm = event_summary(labels, p_by_file)
        sweep_rows.append({"calibration_quantile": q, "threshold": thr, **metric_row(y_all, p_all_q),
                           "event_detected": detected, "event_total": total,
                           "normal_false_alarm_events": false_alarm, "is_default": int(q == CALIBRATION_QUANTILE)})
    sweep = pd.DataFrame(sweep_rows).round(4)

    # ---------------- 8. model_baseline_by_pageno.csv ----------------
    baselines = pd.DataFrame({"PageNo": lightgbm.phase_median.index})
    baselines["page_median_power"] = baselines["PageNo"].map(lightgbm.phase_median)  # PreviousPower 기본값·참고 기준선
    baselines["residual_median"] = baselines["PageNo"].map(lightgbm.residual_median)  # 이상 점수 중심
    baselines["residual_scale"] = baselines["PageNo"].map(lightgbm.residual_scale)  # 이상 점수 scale
    baselines["threshold"] = lightgbm.threshold
    baselines["calibration_quantile"] = CALIBRATION_QUANTILE
    baselines["train_period"] = f"{split_info['train_start']} ~ {split_info['train_end']}"
    baselines = baselines.round(4)

    # ---------------- 1. data_quality_summary.csv ----------------
    dq_rows = data_quality("Training_Data", full_train, "train", None, train_recipes, known_setpower)
    for name in TEST_FILES:
        dq_rows += data_quality(name, tests[name], "test", labels[name] if name.endswith("NG") else None, train_recipes, known_setpower)
    dq = pd.DataFrame(dq_rows)

    # ---------------- 9~11. EDA ----------------
    condition = condition_summary({"Training_Data": full_train, **tests}, pred)
    monthly = (full_train.assign(month=full_train["WorkingTime"].dt.to_period("M").astype(str))
               .groupby(["month", "SetPower"])["RealPower"].agg(["count", "mean", "std"]).reset_index())
    first = monthly.sort_values("month").groupby("SetPower")["mean"].transform("first")
    monthly["change_from_first_w"] = monthly["mean"] - first
    monthly["change_from_first_pct"] = monthly["change_from_first_w"] / first * 100
    monthly["drift_status"] = np.where(monthly["change_from_first_w"] <= DRIFT_WARN_W, "주의", "정상")
    monthly = monthly.round(3)
    corr = full_train[["PageNo", "Speed", "Length", "SetPower", "GateOnTime", "RealPower"]].corr().round(4)
    corr.insert(0, "variable", corr.index)

    # ---------------- 12. equipment_info.csv ----------------
    equipment = equipment_info()

    # ---------------- 저장 ----------------
    print("\n[대시보드 연동 산출물 저장]")
    save_csv(dq, "data_quality_summary.csv")
    save_csv(pred, "predictions.csv")
    save_csv(segments.drop(columns=["priority_order"]) if not segments.empty else segments, "anomaly_segments.csv")
    save_csv(modules, "module_judgement.csv")
    save_csv(metrics, "metrics_by_file.csv")
    save_csv(regression, "regression_metrics.csv")
    save_csv(sweep, "threshold_sweep.csv")
    save_csv(baselines, "model_baseline_by_pageno.csv")
    save_csv(condition, "eda_condition_summary.csv")
    save_csv(monthly, "eda_monthly_trend.csv")
    save_csv(corr, "eda_correlation.csv")
    save_csv(equipment, "equipment_info.csv")

    # 13. 점검 이력: 대시보드에서 담당자가 입력하는 파일 → 이미 있으면 절대 덮어쓰지 않음
    log_path = OUTPUT_DIR / "inspection_log.csv"
    if not log_path.exists():
        pd.DataFrame(columns=["segment_id", "file", "status", "inspector", "memo", "action_time", "updated_at"]).to_csv(
            log_path, index=False, encoding="utf-8-sig")
        print("  생성: inspection_log.csv (빈 템플릿)")
    else:
        print("  유지: inspection_log.csv (기존 점검 기록 보존)")

    lightgbm.model.booster_.save_model(str(OUTPUT_DIR / "lightgbm_model.txt"))
    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance_gain": lightgbm.model.booster_.feature_importance(importance_type="gain"),
        "importance_split": lightgbm.model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)
    save_csv(importance, "lightgbm_feature_importance.csv")

    payload = {
        "protocol": {
            "source": "final_three_model_comparison.py 의 LightGBM 부분 추출 (대시보드는 LightGBM 단일 모델)",
            "training_split": "chronological cycle split: first 70% train, last 30% normal calibration",
            "threshold_rule": THRESHOLD_LABEL,
            "threshold_quantile": CALIBRATION_QUANTILE,
            "test_files": TEST_FILES,
        },
        "split": split_info,
        "threshold": float(lightgbm.threshold),
        "best_iteration": int(lightgbm.model.best_iteration_ or 0),
        "training_seconds": float(lightgbm.training_seconds),
        "inference_ms_per_1000_rows": float(inference_ms),
        "summary": metrics[metrics["file"] == "전체 테스트"].to_dict(orient="records"),
        "outputs": sorted(p.name for p in OUTPUT_DIR.glob("*.csv")),
    }
    (OUTPUT_DIR / "lightgbm_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n[전체 테스트 성능]")
    print(metrics[metrics["file"] == "전체 테스트"][["precision", "recall", "f1", "tn", "fp", "fn", "tp",
                                                    "event_detected", "event_total", "normal_false_alarm_events"]].to_string(index=False))
    print(f"\n결과 저장 완료: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
