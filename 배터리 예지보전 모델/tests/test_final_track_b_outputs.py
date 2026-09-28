"""Regression checks replacing the removed legacy final_track_b artifact suite."""
import json
import hashlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'outputs/track_b_final_v2'

class CurrentArtifactContractTests(unittest.TestCase):
    def test_source_model_hashes_and_pre_test_selection(self):
        manifest = json.loads((OUTPUT/'run_manifest.json').read_text(encoding='utf-8'))
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        for path, expected in manifest['source_sha256'].items():
            self.assertEqual(digest(ROOT.parent/path), expected)
        for path, expected in manifest['model_sha256'].items():
            self.assertEqual(digest(OUTPUT/'models'/path), expected)
        self.assertLessEqual(manifest['selection']['selected_at'], manifest['test_evaluation_started_at'])
        self.assertFalse(manifest['external_validation_completed'])
        self.assertEqual(manifest['evaluation_protocol'], 'retrospective_reused_file_holdout')
