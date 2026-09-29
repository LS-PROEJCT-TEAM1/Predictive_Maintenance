"""Portable root .env configuration; never print credential values."""
import json
import os
from pathlib import Path
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


def _path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def settings():
    root_env = ROOT / '.env'
    candidates = sorted(ROOT.glob('*firebase-adminsdk*.json'))
    # Portable setups must not silently inherit another PC's legacy credentials.
    portable = root_env.is_file() or bool(candidates)
    config, private = {}, {}
    if not portable:
        path = _path(os.environ.get('MANUFACTURING_SETTINGS', '.local/settings.json'))
        if path.is_file():
            try:
                config = json.loads(path.read_text(encoding='utf-8-sig'))
                if not isinstance(config, dict):
                    raise ValueError()
            except (ValueError, OSError):
                raise ValueError('기존 연결 설정 파일을 읽을 수 없습니다.') from None
        if config.get('connections_file'):
            private = dotenv_values(_path(config['connections_file']), encoding='utf-8-sig')
    if root_env.is_file():
        private = dotenv_values(root_env, encoding='utf-8-sig', interpolate=False)

    def value(name, legacy=None, default=''):
        return str(os.environ.get(name, private.get(name, config.get(legacy, default))) or '').strip()

    sdk = value('FIREBASE_SERVICE_ACCOUNT_FILE', 'service_account_file')
    if not sdk:
        if len(candidates) > 1:
            raise ValueError('SDK 파일이 여러 개입니다. .env의 FIREBASE_SERVICE_ACCOUNT_FILE에 사용할 파일명을 지정하세요.')
        sdk = str(candidates[0]) if candidates else ''
    return {
        **config,
        'project_id': value('FIREBASE_PROJECT_ID', 'project_id', 'ls-proejct-team1'),
        'service_account_file': str(_path(sdk)) if sdk else '',
        'firebase_web_key': value('FIREBASE_WEB_API_KEY', 'firebase_web_key'),
        'gemini_key': value('GEMINI_API_KEY'),
        'gemini_model': value('GEMINI_MODEL', 'gemini_model', 'gemini-2.5-flash'),
        'startup_mode': value('BATTERYFLOW_MODE', default='auto').lower(),
        'embedding_model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        'rag_dir': ROOT / '.local/rag',
    }


def connection_missing(config):
    missing = []
    sdk = config.get('service_account_file')
    if not sdk or not Path(sdk).is_file():
        missing.append('Firebase SDK JSON')
    else:
        try:
            account = json.loads(Path(sdk).read_text(encoding='utf-8-sig'))
            valid = isinstance(account, dict) and account.get('type') == 'service_account' and all(
                account.get(k) for k in ('project_id', 'client_email', 'private_key', 'token_uri'))
        except (ValueError, OSError):
            valid = False
        if not valid:
            raise ValueError('Firebase SDK JSON 형식이 잘못되었습니다. 관리자 인증 파일을 확인하세요.')
        if account['project_id'] != config.get('project_id'):
            raise ValueError('Firebase SDK와 FIREBASE_PROJECT_ID가 서로 다릅니다.')
    for key, label in (('firebase_web_key', 'FIREBASE_WEB_API_KEY'), ('gemini_key', 'GEMINI_API_KEY')):
        if not config.get(key):
            missing.append(label)
    return missing


def resolve_mode(requested=None):
    if requested == 'demo':
        return 'demo', []
    config = settings()
    mode = requested or config['startup_mode']
    if mode not in ('auto', 'connected', 'demo'):
        raise ValueError('BATTERYFLOW_MODE는 auto, connected, demo 중 하나여야 합니다.')
    if mode == 'demo':
        return 'demo', []
    missing = connection_missing(config)
    if missing and mode == 'connected':
        raise ValueError('실제 연결 설정 누락: ' + ', '.join(missing))
    return ('demo' if missing else 'connected'), missing
