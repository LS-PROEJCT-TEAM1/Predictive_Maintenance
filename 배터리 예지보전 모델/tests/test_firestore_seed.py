import hashlib
import json
import unittest
from pathlib import Path

import app
from scripts.import_firestore_seed import validate


ROOT = Path(__file__).resolve().parents[1]
SEED_DIR = ROOT / "firestore" / "seed"


class FirestoreSeedTest(unittest.TestCase):
    def test_manifest_counts_and_hashes(self):
        manifest = json.loads((SEED_DIR / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(5273, manifest["totalDocuments"])
        self.assertEqual(manifest["totalDocuments"], validate(SEED_DIR))
        for item in manifest["files"]:
            path = SEED_DIR / item["file"]
            self.assertEqual(item["bytes"], path.stat().st_size)
            self.assertEqual(item["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_measurement_seed_can_rebuild_dashboard_replay(self):
        replay = app.load_replay_from_firestore_seed(SEED_DIR / "measurements.jsonl")
        self.assertEqual(5226 * 5, len(replay))
        self.assertEqual(set(app.supervised_models + app.unsupervised_models), set(replay["model"]))
        self.assertEqual(set(app.source_files), set(replay["source_file"]))

    def test_no_credential_fields_are_exported(self):
        forbidden = {"private_key", "privateKey", "client_email", "clientEmail", "apiKey"}
        for path in SEED_DIR.glob("*.jsonl"):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    document = json.loads(line)
                    self.assertTrue(forbidden.isdisjoint(document["data"]))


if __name__ == "__main__":
    unittest.main()
