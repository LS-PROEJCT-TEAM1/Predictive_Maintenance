from __future__ import annotations

import unittest

import app


class OperationalDashboardTests(unittest.TestCase):
    def test_all_replay_files_are_available(self) -> None:
        self.assertEqual(
            set(app.source_files),
            {
                "WeldingTest_01_OK",
                "WeldingTest_02_OK",
                "WeldingTest_03_NG",
                "WeldingTest_04_NG",
            },
        )

    def test_model_families_are_separated(self) -> None:
        self.assertIn("LogisticCurrent", app.supervised_models)
        self.assertIn("RobustPhaseZ", app.unsupervised_models)
        self.assertFalse(set(app.supervised_models) & set(app.unsupervised_models))

    def test_ng04_continuous_event_is_summarized(self) -> None:
        frame = app.combined_rows("WeldingTest_04_NG", "LogisticCurrent", "RobustPhaseZ")
        events = app.events_for(frame)
        continuous = events[
            events["event_type"].eq("연속 저출력 이상")
            & events["actual_ng_overlap"].eq("예")
        ]
        self.assertEqual(len(continuous), 1)
        self.assertEqual(int(continuous.iloc[0]["duration_rows"]), 273)
        self.assertEqual(continuous.iloc[0]["pages"], "1~39 (39곳)")

    def test_every_tab_renders(self) -> None:
        for tab in ["overview", "map", "events", "models", "quality"]:
            rendered = app.render_tab(
                tab,
                "WeldingTest_04_NG",
                "LogisticCurrent",
                "RobustPhaseZ",
            )
            self.assertIsNotNone(rendered, tab)

    def test_http_entrypoint(self) -> None:
        response = app.app.server.test_client().get("/")
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
