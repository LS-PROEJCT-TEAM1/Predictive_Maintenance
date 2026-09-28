import json
import unittest

import numpy as np
import pandas as pd

from backend.data import Repository
from backend.quality_evidence import snapshot
from frontend.charts import quality_heat
from frontend.quality_analysis import analysis


class QualityEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Repository()

    def test_snapshot_matches_original_at_selected_time(self):
        for test in self.repo.meta()["tests"]:
            for progress in (1, 50, 100):
                result = self.repo.quality(test, "M07CV06", progress, "raw")
                snap = result["snapshot"]
                original = self.repo.quality_original(test)
                self.assertEqual(snap["index"], int((len(original)-1)*progress/100))
                self.assertEqual(len(snap["cells"]), 176)
                self.assertEqual(len(snap["temperatures"]), 32)
                values = original.iloc[snap["index"]][[p["id"] for p in snap["cells"]]]
                self.assertAlmostEqual(snap["voltage"]["range"], values.max()-values.min())
                self.assertEqual(result["cellSeries"][-1]["cell"], original.iloc[snap["index"]]["M07CV06"])
                json.dumps(result, allow_nan=False)

    def test_sensor_evidence_survives_clean_display(self):
        raw = self.repo.quality("Test06_NG_chg", "M16CV11", 50, "raw")
        clean = self.repo.quality("Test06_NG_chg", "M16CV11", 50, "clean")
        original = next(p for p in raw["snapshot"]["cells"] if p["id"] == "M16CV11")
        corrected = next(p for p in clean["snapshot"]["cells"] if p["id"] == "M16CV11")
        self.assertGreater(original["value"], 5)
        self.assertTrue(corrected["invalid"])
        self.assertEqual(corrected["raw"], original["raw"])
        self.assertLessEqual(corrected["value"], 5)
        self.assertEqual(raw["abnormalPointCount"], clean["abnormalPointCount"])

    def test_missing_and_nonfinite_values_remain_explicit(self):
        raw = pd.DataFrame({"M01CV01": [np.inf], "M01T01": [np.nan]})
        clean = pd.DataFrame({"M01CV01": [3.7], "M01T01": [25.]})
        result = snapshot(raw, clean, 0, "raw")
        self.assertIsNone(result["voltage"]["range"])
        self.assertEqual(result["temperature"]["invalidCount"], 1)
        json.dumps(result, allow_nan=False)

    def test_source_labels_are_not_model_predictions(self):
        data = self.repo.quality("Test07_NG_dchg")
        wire = next(p for p in data["defectEvidence"] if p["id"] == "wire")
        self.assertEqual(wire["status"], "자료 확인 필요")
        normal = self.repo.quality("Test03_OK_chg")
        self.assertFalse(any(p["sourceMatch"] for p in normal["defectEvidence"]))

    def test_map_coordinates_and_rendering(self):
        data = self.repo.quality("Test08_NG_chg", "M10CV10", 80)
        heat = quality_heat(data)
        self.assertEqual(len(heat.data[0].z), 11)
        self.assertEqual(len(heat.data[0].z[0]), 16)
        self.assertEqual(heat.data[0].customdata[9][9][0], "M10CV10")
        self.assertEqual(heat.data[-1].x[0], "M10")
        self.assertEqual(heat.data[-1].y[0], "CV10")
        self.assertEqual(len(quality_heat(data, "temperature").data[0].z), 2)
        self.assertIsNotNone(analysis(data, "미확정", "cell", "capacity"))


if __name__ == "__main__":
    unittest.main()
