import unittest

import app
from components.charts import evidence_figure, location_figure


class QualityDashboardTest(unittest.TestCase):
    def test_locked_test_defaults_use_real_outputs(self):
        scores = app.test_scores("Test07_NG_dchg")
        ranks = app.test_ranks("Test07_NG_dchg")
        self.assertEqual(len(scores), 4594)
        self.assertEqual(app.anomaly_regions(scores["예측"]), 1)
        self.assertEqual(ranks.iloc[0]["셀"], "M02CV01")
        self.assertAlmostEqual(float(ranks.iloc[0]["이상점수"]), 5.853, places=3)

    def test_figures_render_for_cell_and_temperature_modes(self):
        raw = app.load_raw("Test07_NG_dchg")
        scores = app.test_scores("Test07_NG_dchg")
        ranks = app.test_ranks("Test07_NG_dchg")
        evidence = evidence_figure(raw, scores, "M02CV01")
        cell_map = location_figure(ranks, raw, "cell", "M02CV01")
        temp_map = location_figure(ranks, raw, "temperature", "M02CV01")
        self.assertGreaterEqual(len(evidence.data), 20)
        self.assertEqual(cell_map.data[0].z.shape, (16, 11))
        self.assertEqual(temp_map.data[0].z.shape, (16, 2))

    def test_dash_callbacks_are_registered(self):
        self.assertGreaterEqual(len(app.app.callback_map), 5)


if __name__ == "__main__":
    unittest.main()
