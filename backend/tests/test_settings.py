import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.settings import settings, resolve_mode


class PortableSettingsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.root_patch = patch('backend.settings.ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def env(self, extra=''):
        (self.root / '.env').write_text(
            'FIREBASE_PROJECT_ID=test-project\nFIREBASE_WEB_API_KEY=fake-web\n'
            'GEMINI_API_KEY=fake-gemini\n' + extra, encoding='utf-8-sig')

    def sdk(self, name='project-firebase-adminsdk-test.json', project='test-project'):
        path = self.root / name
        path.write_text(json.dumps(dict(type='service_account', project_id=project,
            client_email='fake@example.com', private_key='test-only', token_uri='https://oauth2.googleapis.com/token')))
        return path

    def test_no_files_selects_demo(self):
        mode, missing = resolve_mode()
        self.assertEqual(mode, 'demo')
        self.assertEqual(len(missing), 3)

    def test_drop_in_files_select_connected_from_any_cwd(self):
        self.env(); sdk = self.sdk()
        self.assertEqual(resolve_mode(), ('connected', []))
        self.assertEqual(Path(settings()['service_account_file']), sdk)

    def test_partial_setup_selects_demo_with_missing_names(self):
        self.env()
        self.assertEqual(resolve_mode(), ('demo', ['Firebase SDK JSON']))

    def test_forced_connected_never_falls_back(self):
        with self.assertRaisesRegex(ValueError, '설정 누락'):
            resolve_mode('connected')

    def test_ambiguous_sdk_requires_explicit_selection(self):
        self.env(); self.sdk(); second = self.sdk('other-firebase-adminsdk.json')
        with self.assertRaisesRegex(ValueError, '여러 개'):
            resolve_mode()
        self.env('FIREBASE_SERVICE_ACCOUNT_FILE=other-firebase-adminsdk.json\n')
        self.assertEqual(Path(settings()['service_account_file']), second)
        self.assertEqual(resolve_mode()[0], 'connected')

    def test_wrong_project_is_error_not_demo(self):
        self.env(); self.sdk(project='other')
        with self.assertRaisesRegex(ValueError, '서로 다릅니다'):
            resolve_mode()

    def test_malformed_sdk_does_not_disclose_contents(self):
        self.env()
        (self.root / 'firebase-adminsdk.json').write_text('do-not-disclose-this')
        with self.assertRaises(ValueError) as caught:
            resolve_mode()
        self.assertNotIn('do-not-disclose-this', str(caught.exception))
        self.assertEqual(resolve_mode('demo'), ('demo', []))

    def test_root_env_prevents_silent_legacy_credentials(self):
        local = self.root / '.local'; local.mkdir()
        (local / 'settings.json').write_text(json.dumps({'firebase_web_key': 'legacy-web', 'connections_file': 'legacy.env'}))
        (self.root / 'legacy.env').write_text('GEMINI_API_KEY=legacy-gemini\n')
        self.assertEqual(settings()['gemini_key'], 'legacy-gemini')
        (self.root / '.env').write_text('BATTERYFLOW_MODE=auto\n')
        self.assertEqual(settings()['gemini_key'], '')
        self.assertEqual(settings()['firebase_web_key'], '')
        self.assertEqual(resolve_mode()[0], 'demo')

    def test_process_environment_overrides_env(self):
        self.env()
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'override'}):
            self.assertEqual(settings()['gemini_key'], 'override')

    def test_demo_is_explicit_and_invalid_mode_is_rejected(self):
        self.env('BATTERYFLOW_MODE=demo\n')
        self.assertEqual(resolve_mode(), ('demo', []))
        self.env('BATTERYFLOW_MODE=typo\n')
        with self.assertRaisesRegex(ValueError, 'BATTERYFLOW_MODE'):
            resolve_mode()


if __name__ == '__main__':
    unittest.main()
