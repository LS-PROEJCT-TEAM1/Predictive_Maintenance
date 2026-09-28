import unittest
from datetime import datetime, timedelta
from backend.maintenance_policy import alarm_regions
from backend.data import Repository


class MaintenancePolicyTests(unittest.TestCase):
    def test_alarm_starts_immediately_clears_and_splits_gaps(self):
        start = datetime(2022,1,1)
        seconds = [0,3,6,9,200,203,206]
        pred = [0,1,1,0,1,1,0]
        rows = [dict(row=i,time=str(start+timedelta(seconds=s)),prediction=p) for i,(s,p) in enumerate(zip(seconds,pred))]
        self.assertEqual(alarm_regions(rows), [(1,2),(4,5)])
        rows[3]['prediction'] = 1
        self.assertEqual(alarm_regions(rows), [(1,3),(4,5)])
        # A cycle boundary does not break adjacent alarms. A missing raw row does.
        rows[5]['row'] = 99
        self.assertEqual(alarm_regions(rows), [(1,3),(4,4),(5,5)])

    def test_runtime_evaluation_matches_default_cycle_counts(self):
        repo = Repository()
        groups = [g for name in ['WeldingTest_02_OK','WeldingTest_04_NG'] for g in repo.maintenance(name)['cycleRows']]
        result = repo.maintenance()['operatingEvaluation']
        for key, label, predicted in [('cycle_tp',1,1),('cycle_fp',0,1),('cycle_fn',1,0),('cycle_tn',0,0)]:
            self.assertEqual(result[key], sum(g['label']==label and g['prediction']==predicted for g in groups))

    def test_all_model_combinations_have_evidence(self):
        repo = Repository(); meta = repo.meta()
        for a in meta['supervised']:
            for b in meta['unsupervised']:
                result = repo.maintenance(supervised=a,unsupervised=b)
                self.assertIsNotNone(result['operatingEvaluation'])
                self.assertEqual(sum(e['rows'] for e in result['events']),result['anomalyRows'])
                self.assertEqual(result['alertCycles'],sum(c['prediction'] for c in result['cycleRows']))

    def test_time_validation_is_explicitly_separate(self):
        result = Repository().validation('maintenance')
        self.assertFalse(result['config']['externalValidationCompleted'])
        for row in result['forward']:
            self.assertLess(row['calibration_end'], row['test_start'])
            self.assertFalse(row['used_for_selection'])
