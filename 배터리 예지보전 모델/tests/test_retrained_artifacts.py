import json
import sys
import unittest
import subprocess
import tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import track_b_final_v2 as m


class RetrainedArtifactTests(unittest.TestCase):
    def test_seed_predictions_reproduce_from_raw_and_current_models(self):
        points = [json.loads(line)['data'] for line in (ROOT/'firestore/seed/measurements.jsonl').read_text(encoding='utf-8').splitlines()]
        models = []
        for path in m.MODEL_DIR.glob('*.joblib'):
            payload = joblib.load(path)
            models.append(m.Detector(**{k:payload[k] for k in ['name','family','model','threshold','feature_names','reference','threshold_source']},training_seconds=0.))
        for source in sorted({p['sourceFile'] for p in points}):
            raw = m.load_test_file(source)
            stored = sorted([p for p in points if p['sourceFile']==source],key=lambda p:p['sourceRow'])
            self.assertEqual(len(raw),len(stored))
            self.assertEqual(raw.source_row.tolist(),[p['sourceRow'] for p in stored])
            self.assertEqual(raw.label.tolist(),[p['actualLabel'] for p in stored])
            for model in models:
                score = model.score(raw)
                np.testing.assert_allclose(score,[p['models'][model.name]['score'] for p in stored],rtol=1e-12,atol=1e-12)
                np.testing.assert_array_equal(score >= model.threshold,[p['models'][model.name]['prediction'] for p in stored])

    def test_cli_rejects_missing_power_without_prediction_file(self):
        frame = m.read_signal(m.DATA_ROOT/'raw_data/test/WeldingTest_02_OK.csv').iloc[:39].copy()
        frame['RealPower'] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            src, dst = Path(directory)/'bad.csv',Path(directory)/'predictions.csv'
            frame.to_csv(src,index=False)
            result = subprocess.run([sys.executable,'-B',str(ROOT/'src/predict_track_b.py'),str(src),str(dst)],capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse(dst.exists())
