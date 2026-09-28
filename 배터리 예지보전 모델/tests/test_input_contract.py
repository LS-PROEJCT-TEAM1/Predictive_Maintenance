import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pdm_contract import validate_sequence
import track_b_final_v2 as pipeline


class InputContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pipeline.read_signal(pipeline.DATA_ROOT/'raw_data/test/WeldingTest_02_OK.csv').iloc[:78].copy()

    def test_valid_complete_cycle(self):
        self.assertEqual(len(validate_sequence(self.raw)), 78)
        mixed = self.raw.copy()
        mixed['WorkingTime'] = mixed.WorkingTime.astype(str)
        mixed.loc[mixed.index[1], 'WorkingTime'] = str(self.raw.WorkingTime.iloc[1].floor('s'))
        self.assertEqual(len(validate_sequence(mixed)), 78)

    def test_invalid_sensor_and_page_rejected(self):
        for col, value in [('RealPower', np.nan), ('RealPower', np.inf), ('RealPower', -1),
                           ('RealPower', 'broken'), ('PageNo', 999), ('PageNo', 1.5),
                           ('SetDuty', 101), ('WorkingTime', None)]:
            with self.subTest(col=col, value=value):
                frame = self.raw.copy().astype({col:object})
                frame.loc[frame.index[0], col] = value
                with self.assertRaises(ValueError):
                    validate_sequence(frame)

    def test_partial_shuffled_and_reversed_cycles_rejected(self):
        cases = [self.raw.iloc[1:], self.raw.iloc[:-1], self.raw.iloc[::-1], self.raw.iloc[[]]]
        duplicate_time = self.raw.copy()
        duplicate_time.iloc[1, duplicate_time.columns.get_loc('WorkingTime')] = duplicate_time.iloc[0].WorkingTime
        cases.append(duplicate_time)
        for frame in cases:
            with self.assertRaises(ValueError):
                validate_sequence(frame)

    def test_feature_builder_cannot_hide_nonfinite_power(self):
        identity = pipeline.add_identity(self.raw, 'probe')
        ref = pipeline.phase_reference(identity)
        for value in [np.nan, np.inf]:
            broken = identity.copy(); broken['RealPower'] = value
            with self.assertRaises(ValueError):
                pipeline.make_features(broken, ref)

    def test_historical_quarantine_preserves_original_ids(self):
        raw = pipeline.require_complete_cycles(pipeline.add_identity(
            pipeline.read_signal(pipeline.DATA_ROOT/'raw_data/train/Training_Data.csv'), 'Training_Data'))
        train, cal, info = pipeline.split_historical_normal(raw)
        self.assertEqual(info['excluded_time_cycles'], 1)
        self.assertEqual(info['excluded_zero_power_cycles'], 1)
        self.assertLess(train.WorkingTime.max(), cal.WorkingTime.min())
        self.assertFalse(set(train.group_id) & set(cal.group_id))
        kept = pd.concat([train, cal])
        self.assertEqual(len(kept), len(raw)-78)
        for _, group in kept.groupby('group_id'):
            self.assertEqual(group.PageNo.tolist(), list(range(1,40)))
            self.assertTrue((group.source_row.diff().dropna() == 1).all())

    def test_winner_independent_of_test_and_wall_time(self):
        metrics = pd.DataFrame([dict(model=n,split='validation',missed_events=0,fn=0,false_alarm_events=0,f1=1.,fp=0,training_seconds=i)
                                for i,n in enumerate(pipeline.COMPLEXITY_ORDER[::-1])])
        self.assertEqual(pipeline.choose_winner(metrics), 'RobustPhaseZ')
        metrics['training_seconds'] = metrics.training_seconds[::-1].values
        test = metrics.copy(); test['split']='locked_test'; test['f1']=0.
        self.assertEqual(pipeline.choose_winner(pd.concat([metrics,test])), 'RobustPhaseZ')
