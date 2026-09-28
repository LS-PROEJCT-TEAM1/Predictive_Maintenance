import io
import unittest
from unittest.mock import patch
import pandas as pd
from backend.demand_workspace import diagnostics, plan_history, inspect_csv, evidence
from backend.data import Repository


class DemandWorkspaceTests(unittest.TestCase):
    def setUp(self):
        diagnostics.cache_clear()

    def tearDown(self):
        diagnostics.cache_clear()

    def test_calendar_alignment_not_adjacent_rows(self):
        daily = [{'part_number':'Part 1','date':f'2021-10-{day}','actual_d':9,
                  **{f'plan_d{h}':day*100+h for h in range(1,6)}} for day in range(27,32)]
        source = {'daily':daily,'cv':[{'part_number':'Part 1','origin_date':'2021-10-29','target_date':'2021-11-01','actual':3000,'Fold':'f'}],
                  'quality':{},'importance':[],'importanceMethod':'test','version':'test'}
        with patch('backend.demand_workspace.evidence',return_value=source):
            rows = plan_history(None,{'date':'2021-11-01','rows':[{'part':'Part 1'}]})
            self.assertEqual([r['계획량'] for r in rows],[2705,2804,2903,3002,3101])
            result = diagnostics('Part 1')['metrics']
            self.assertEqual([r['Bias'] for r in result],[-295,-196,-97,2,101])
            self.assertTrue(all(r['상관계수'] is None for r in result))
            rows = plan_history(None,{'date':'2021-11-01','rows':[{'part':'Part 1'},{'part':'Part 2'}]})
            self.assertTrue(all(r['계획량'] is None for r in rows))

    def test_evidence_cohort_and_importance(self):
        data = diagnostics()
        self.assertEqual(data['correlationRows'],1289)
        self.assertEqual(len({r['행 수'] for r in data['metrics']}),1)
        self.assertAlmostEqual(sum(r['percent'] for r in evidence()['importance']),100)
        self.assertNotIn('Part 21',{r['part_number'] for r in evidence()['daily']})
        self.assertEqual(diagnostics('Part 21')['correlationRows'],0)

    def test_plan_reference_matches_current_seed(self):
        repo = Repository()
        for part in ['ALL','Part 92','Part 60']:
            data = repo.demand(part=part)
            row = plan_history(repo,data)[2]
            self.assertAlmostEqual(row['계획량'],data['plan'])

    def test_input_checks_zero_unsorted_invalid_and_quarantine(self):
        rows = [{'part_number':'Part 1','date':f'2021-10-{d}','actual_d':0,'plan_d3':0,'plan_d4':0,'plan_d5':0} for d in [29,27,28]]
        def check(rs):
            return inspect_csv(pd.DataFrame(rs).to_csv(index=False).encode(),['Part 21','Part 26'])
        self.assertTrue(check(rows)['valid'])
        self.assertFalse(check(rows+[rows[0]])['valid'])
        self.assertFalse(check([{**r,'part_number':'Part 21'} for r in rows])['valid'])
        self.assertFalse(check([{**r,'actual_d':-1} for r in rows])['valid'])
        self.assertFalse(check([{**r,'plan_d3':float('inf')} for r in rows])['valid'])
        self.assertFalse(inspect_csv(b'wrong\n1',[])['valid'])
        self.assertFalse(inspect_csv(b'x'*1_000_001,[])['valid'])


if __name__=='__main__':
    unittest.main()
