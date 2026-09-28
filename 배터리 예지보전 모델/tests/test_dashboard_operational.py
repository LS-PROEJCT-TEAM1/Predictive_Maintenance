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
        self.assertIn("LightGBMResidual", app.unsupervised_models)
        self.assertFalse(set(app.supervised_models) & set(app.unsupervised_models))

    def test_ng04_continuous_event_is_summarized(self) -> None:
        frame = app.combined_rows("WeldingTest_04_NG", "LogisticCurrent", "RobustPhaseZ")
        events = app.events_for(
            frame,
            "WeldingTest_04_NG",
            "LogisticCurrent",
            "RobustPhaseZ",
            {},
        )
        continuous = events[events["event_type"].eq("연속 저출력 이상")]
        self.assertGreaterEqual(len(continuous), 1)
        self.assertEqual(int(events.loc[events.start_row.ge(78), 'duration_rows'].sum()), 273)
        for event in events.itertuples():
            segment = frame.iloc[event.start_row:event.end_row+1]
            gaps = __import__('pandas').to_datetime(segment.WorkingTime, format='ISO8601').diff().dt.total_seconds().dropna()
            self.assertTrue(gaps.le(120).all())

    def test_dashboard_evidence_and_operational_data_render(self) -> None:
        frame = app.combined_rows("WeldingTest_04_NG", "LogisticCurrent", "RobustPhaseZ")
        events = app.events_for(
            frame,
            "WeldingTest_04_NG",
            "LogisticCurrent",
            "RobustPhaseZ",
            {},
        )
        self.assertEqual(int(events.duration_rows.sum()), int(frame.combined_prediction.sum()))
        self.assertEqual(len(app.quality_summary("WeldingTest_04_NG")), 6)
        self.assertGreater(len(app.power_figure(frame).data), 3)
        self.assertEqual(len(app.score_figure(frame, "LogisticCurrent", "RobustPhaseZ").data), 2)

    def test_callbacks_are_registered(self) -> None:
        app.app._setup_server()
        self.assertGreaterEqual(len(app.app.callback_map), 7)

    def test_http_entrypoint(self) -> None:
        response = app.app.server.test_client().get("/")
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
