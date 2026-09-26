from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from backend.runtime_assets import ARTIFACTS, quality_path, artifact_path

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "firestore/seed"
TRACKS = {"demand": "공급망 예측", "maintenance": "예지보전", "quality": "품질 보증"}


def serial(value):
    return json.loads(json.dumps(value, default=lambda x: x.item() if hasattr(x, "item") else str(x), ensure_ascii=False))


def records(frame):
    return json.loads(frame.to_json(orient="records", date_format="iso", force_ascii=False))


class Repository:
    def __init__(self, seed_dir=SEED):
        self.manifest = json.loads((seed_dir / "manifest.json").read_text(encoding="utf-8"))
        self.docs = {}
        for item in self.manifest["files"]:
            path = seed_dir / item["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"시드 해시 불일치: {item['file']}")
            for line in path.read_text(encoding="utf-8").splitlines():
                doc = json.loads(line)
                self.docs[doc["path"]] = doc["data"]
        self.base = self.manifest["rootDocument"]
        self.parts = {p["part_number"]: p for p in self.collection("demandParts")}
        self.dates = sorted({f["target_date"] for p in self.parts.values() for f in p["forecasts"]})

    def collection(self, name):
        prefix = f"{self.base}/{name}/"
        return [v for k, v in self.docs.items() if k.startswith(prefix) and "/" not in k[len(prefix):]]

    def get(self, name, item="current"):
        try:
            return self.docs[f"{self.base}/{name}/{item}"]
        except KeyError:
            raise ValueError("해당 데이터가 없습니다.") from None

    def meta(self):
        config = self.get("maintenanceConfig")
        return {"dataVersion": self.manifest["dataVersion"], "generatedAt": self.manifest["generatedAt"],
                "documents": len(self.docs), "mode": "local-preview", "dates": self.dates,
                "parts": sorted(self.parts, key=lambda p: int(p.split()[-1])),
                "runs": [r["sourceFile"] for r in self.collection("maintenanceRuns")],
                "tests": [t["testId"] for t in self.collection("qualityTests")],
                "supervised": sorted({m["model"] for m in self.collection("maintenanceModels") if m["family"] == "supervised"}),
                "unsupervised": sorted({m["model"] for m in self.collection("maintenanceModels") if m["family"] == "unsupervised"}),
                "defaultSupervised": config["defaultSupervisedModel"], "defaultUnsupervised": config["defaultUnsupervisedModel"]}

    def demand(self, date=None, part="ALL", model="3-day Moving Average"):
        date = date or self.dates[-1]
        if date not in self.dates or (part != "ALL" and part not in self.parts):
            raise ValueError("기준일 또는 부품을 확인하세요.")
        if model not in ["3-day Moving Average", "XGBoost", "LightGBM", "CatBoost", "LSTM"]:
            raise ValueError("지원하지 않는 모델입니다.")
        selected = list(self.parts.values()) if part == "ALL" else [self.parts[part]]
        trend, rows = {}, []
        for p in selected:
            for f in p["forecasts"]:
                if f["target_date"] > date:
                    continue
                bucket = trend.setdefault(f["target_date"], {"date": f["target_date"], "forecast": 0., "plan": 0., "actual": 0.})
                for key, source in [("forecast", model), ("plan", "D+3 Plan Reference"), ("actual", "actual")]:
                    bucket[key] += float(f[source])
                if f["target_date"] == date:
                    pred, plan = float(f[model]), float(f["D+3 Plan Reference"])
                    gap = pred-plan
                    rows.append({"part": p["part_number"], "forecast": round(pred, 2), "plan": plan,
                                 "actual": f["actual"], "gap": round(gap, 2), "gapPct": round(gap/plan*100, 1) if plan else None,
                                 "review": abs(gap) >= max(10, plan*.2),
                                 "direction": "상향 검토" if gap > 0 else "하향 검토", "date": date, "model": model})
        rows.sort(key=lambda r: abs(r["gap"]), reverse=True)
        total = sum(r["forecast"] for r in rows)
        plan = sum(r["plan"] for r in rows)
        return {"date": date, "part": part, "model": model, "forecast": round(total, 2), "plan": plan,
                "gap": round(total-plan, 2), "gapPct": round((total-plan)/plan*100, 2) if plan else None,
                "reviewCount": sum(r["review"] for r in rows), "count": len(rows), "rows": rows,
                "trend": sorted(trend.values(), key=lambda r: r["date"]),
                "partInfo": self.parts.get(part), "version": self.manifest["dataVersion"]}

    def maintenance(self, run="WeldingTest_04_NG", supervised=None, unsupervised=None):
        meta = self.meta()
        supervised = supervised or meta["defaultSupervised"]
        unsupervised = unsupervised or meta["defaultUnsupervised"]
        if run not in meta["runs"] or supervised not in meta["supervised"] or unsupervised not in meta["unsupervised"]:
            raise ValueError("시험 파일 또는 모델을 확인하세요.")
        prefix = f"{self.base}/maintenanceRuns/{run}/measurementChunks/"
        points = sorted([p for k, v in self.docs.items() if k.startswith(prefix) for p in v["points"]], key=lambda p: p["sourceRow"])
        rows = []
        for p in points:
            a, b = p["models"][supervised], p["models"][unsupervised]
            ratio_a = a["score"]/max(a["threshold"], 1e-12)
            ratio_b = b["score"]/max(b["threshold"], 1e-12)
            rows.append({"row": p["sourceRow"], "cycle": p["cycle"], "page": p["pageNo"],
                         "time": p["workingTime"], "power": p["signals"]["realPower"],
                         "expected": p["normalReference"]["expectedPower"], "supRatio": ratio_a, "unsupRatio": ratio_b,
                         "risk": max(ratio_a, ratio_b), "prediction": int(a["prediction"] or b["prediction"]), "label": p["actualLabel"]})
        mask = np.array([r["prediction"] for r in rows], bool)
        starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
        ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
        events = []
        for i, (a, b) in enumerate(zip(starts, ends), 1):
            segment = rows[a:b+1]
            events.append({"id": f"{run}:{supervised}:{unsupervised}:{a}-{b}", "event": f"EVT-{i:03d}",
                           "type": "연속 이상" if b-a+1 >= 39 else "고립 이상", "start": int(a), "end": int(b),
                           "rows": int(b-a+1), "maxRisk": round(max(r["risk"] for r in segment), 2),
                           "severity": "위험" if b-a+1 >= 39 else "주의", "run": run})
        return {"run": run, "supervised": supervised, "unsupervised": unsupervised, "points": rows,
                "events": events, "anomalyRows": int(mask.sum()), "cycles": len({r["cycle"] for r in rows}),
                "maxRisk": round(max(r["risk"] for r in rows), 2), "version": self.manifest["dataVersion"]}

    @lru_cache(maxsize=5)
    def quality_raw(self, test):
        if test not in self.meta()["tests"]:
            raise ValueError("시험 ID를 확인하세요.")
        path = quality_path(test)
        raw = pd.read_csv(path)
        cols = [c for c in raw if __import__('re').fullmatch(r"M\d+(?:CV|T)\d+", c)]
        clean = raw[cols].apply(pd.to_numeric, errors="coerce")
        for kind, lo, hi in [("CV", 2, 5), ("T", -20, 100)]:
            cs = [c for c in cols if ("CV" in c) == (kind == "CV")]
            clean[cs] = clean[cs].where(clean[cs].ge(lo) & clean[cs].le(hi))
        clean = clean.interpolate(limit_direction="both").ffill().bfill()
        for group in [[c for c in cols if "CV" in c], [c for c in cols if "CV" not in c]]:
            good = [c for c in group if not clean[c].isna().all()]
            for c in set(group)-set(good):
                clean[c] = clean[good].median(axis=1)
        return clean

    def quality(self, test="Test07_NG_dchg", cell="M02CV01", progress=100):
        doc = dict(self.get("qualityTests", test))
        raw = self.quality_raw(test)
        if cell not in raw or "CV" not in cell:
            raise ValueError("셀 ID를 확인하세요.")
        limit = max(1, int(len(raw)*progress/100))
        indexes = np.unique(np.linspace(0, limit-1, min(300, limit), dtype=int))
        module = [c for c in raw if c.startswith(cell[:3]) and "CV" in c]
        doc["selectedCell"] = cell
        doc["progress"] = progress
        doc["cellSeries"] = [{"index": int(i), "progressPct": round(i/max(len(raw)-1, 1)*100, 2),
                              "cell": float(raw.iloc[i][cell]), "module": float(raw.iloc[i][module].mean())} for i in indexes]
        doc["series"] = [r for r in doc["series"] if r["index"] < limit]
        return doc

    def validation(self, track):
        if track == "demand":
            config = self.get("demandConfig")
            return {"rows": config["walkForwardMetrics"], "holdout": config["modelMetrics"],
                    "extra": self.artifact("demand_folds"), "partErrors": self.artifact("demand_errors")}
        if track == "maintenance":
            return {"rows": self.collection("maintenanceModels"), "extra": self.artifact("maintenance_features")}
        if track == "quality":
            config = self.get("qualityConfig")
            return {"rows": self.collection("qualityModels"), "extra": config["crossValidation"],
                    "fileResults": config["testResults"], "features": self.artifact("quality_features")}
        raise ValueError("분석 영역을 확인하세요.")

    def data_quality(self, track):
        if track == "demand":
            return self.get("demandConfig")["dataQuality"]
        if track in ["maintenance", "quality"]:
            return self.collection(f"{track}DataQuality")
        raise ValueError("분석 영역을 확인하세요.")

    @staticmethod
    @lru_cache(maxsize=12)
    def artifact(name):
        # Explicit allowlist: no client-controlled filesystem paths.
        if name not in ARTIFACTS:
            raise ValueError("지원하지 않는 결과입니다.")
        return records(pd.read_csv(artifact_path(name)))
