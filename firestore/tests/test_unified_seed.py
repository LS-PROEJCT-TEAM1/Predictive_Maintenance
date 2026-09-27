import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SEED = ROOT / "firestore" / "seed"
spec = importlib.util.spec_from_file_location("seed_builder", ROOT / "firestore" / "build_unified_seed.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def read_seed(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    docs = []
    for item in manifest["files"]:
        path = directory / item["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == item["documents"]
        docs.extend(rows)
    assert len(docs) == manifest["totalDocuments"]
    return manifest, {doc["path"]: doc["data"] for doc in docs}


class UnifiedSeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.docs = read_seed(SEED)
        cls.base = cls.manifest["rootDocument"]

    def test_archive_and_current_integrity(self):
        old, old_docs = read_seed(ROOT / "firestore" / "versions" / "seed-v1")
        self.assertEqual(old["schemaVersion"], 1)
        self.assertEqual(len(old_docs), 259)
        self.assertEqual(self.manifest["schemaVersion"], 3)
        previous, _ = read_seed(ROOT / "firestore/versions/seed-2026-09-26.v2")
        self.assertEqual(previous["totalDocuments"], 266)
        self.assertEqual(self.manifest["deploymentStatus"], "local_only_pending_upload")
        self.assertEqual(len(self.docs), self.manifest["totalDocuments"])
        builder.validate_documents([{"path": p, "data": d} for p, d in self.docs.items()])
        for artifact in self.manifest["sourceArtifacts"]:
            self.assertEqual(hashlib.sha256((ROOT / artifact["path"]).read_bytes()).hexdigest(), artifact["sha256"])

    def test_demand_summary_uses_same_policy_and_date_as_parts(self):
        summary = self.docs[f"{self.base}/demandOverview/current"]
        total = plan = count = reviews = 0
        for path, part in self.docs.items():
            if "/demandParts/" not in path:
                continue
            if part["forecasts"]:
                self.assertEqual(part["recommended_model"], summary["operatingModel"])
            for forecast in part["forecasts"]:
                if forecast["target_date"] != summary["targetDate"]:
                    continue
                prediction, reference = forecast[summary["operatingModel"]], forecast["D+3 Plan Reference"]
                total += prediction
                plan += reference
                count += 1
                reviews += abs(prediction-reference) >= max(10, reference*.2)
                self.assertAlmostEqual(part["latestAlert"]["recommended_forecast"], prediction)
        self.assertAlmostEqual(summary["recommendedForecast"], round(total, 2))
        self.assertAlmostEqual(summary["planReference"], round(plan, 2))
        self.assertEqual(summary["forecastPartCount"], count)
        self.assertEqual(summary["reviewCount"], reviews)

    def test_quality_counts_and_metrics_match_full_saved_predictions(self):
        scores = pd.read_csv(ROOT / "배터리 품질보증 모델/output/models/model_C_pca_t2_spe_이상점수.csv")
        for test_id, group in scores.groupby("파일"):
            doc = self.docs[f"{self.base}/qualityTests/{test_id}"]
            mask = group["예측"].to_numpy(bool)
            expected = int(np.sum(mask & ~np.r_[False, mask[:-1]]))
            self.assertEqual(doc["abnormalSegmentCount"], expected)
            self.assertEqual(doc["abnormalPointCount"], int(mask.sum()))
            self.assertEqual(sum(s["rows"] for s in doc["anomalySegments"]), int(mask.sum()))
            self.assertEqual(doc["decision"], "pending")
            for point in doc["series"]:
                original = group.iloc[point["index"]]
                self.assertEqual(point["pcaPrediction"], original["예측"])
                self.assertEqual(point["pcaQ"], original["SPE"])
        model = next(d for p, d in self.docs.items() if "/qualityModels/" in p and d["모델"] == "PCA (T²·SPE)" and d["구분"] == "테스트")
        y, pred = scores["label"].to_numpy(bool), scores["예측"].to_numpy(bool)
        self.assertEqual(model["TP"], int((y & pred).sum()))
        self.assertEqual(model["FP"], int((~y & pred).sum()))
        self.assertEqual(model["FN"], int((y & ~pred).sum()))

    def test_operational_quality_evidence_does_not_depend_on_ground_truth(self):
        scores = pd.read_csv(ROOT / "배터리 품질보증 모델/output/models/model_C_pca_t2_spe_이상점수.csv")
        sample = scores[scores["파일"].eq("Test07_NG_dchg")].copy()
        thresholds = self.docs[f"{self.base}/qualityConfig/current"]["pcaThresholds"]
        first = builder.quality_test_payload("Test07_NG_dchg", sample, thresholds)
        sample["label"] = 1 - sample["label"]
        second = builder.quality_test_payload("Test07_NG_dchg", sample, thresholds)
        for field in ["suspectedCells", "cellHeatmap", "decision", "aiStatus", "abnormalSegmentCount", "representativeIndex"]:
            self.assertEqual(first[field], second[field])

    def test_overview_tracks_match_their_domain_documents(self):
        overview = self.docs[f"{self.base}/overview/current"]
        for track in ["demand", "maintenance", "quality"]:
            full = self.docs[f"{self.base}/{track}Overview/current"]
            for field, value in overview[track].items():
                self.assertEqual(value, full[field])
        maintenance = overview["maintenance"]
        events = [p for p in self.docs if f"/maintenanceRuns/{maintenance['sourceFile']}/events/" in p]
        self.assertEqual(maintenance["unconfirmedCount"], len(events))


if __name__ == "__main__":
    unittest.main()
