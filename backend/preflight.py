"""Fail early with actionable setup errors without printing secrets."""
import importlib.util
import sys
from pathlib import Path
from backend.data import Repository
from backend.runtime_assets import ROOT, verify_runtime


def check(connected=False):
    if sys.version_info[:2] != (3, 12):
        raise SystemExit('Python 3.12로 setup_local.cmd를 실행하세요.')
    for module in ('dash', 'fastapi', 'xgboost', 'sklearn', 'joblib', 'torch', 'cryptography'):
        if importlib.util.find_spec(module) is None:
            raise SystemExit('필수 라이브러리 누락: ' + module + '. setup_local.cmd를 실행하세요.')
    try:
        repo = Repository()
        manifest = verify_runtime()
        if repo.manifest['dataVersion'] != manifest['dataVersion']:
            raise ValueError('공식 시드와 runtime 자료 버전이 다릅니다.')
        for name in ('frontend/assets/ci_img20.png', 'frontend/assets/ci_img02.png', '발주량 예측 모델/src/inference.py'):
            if not (ROOT / name).is_file():
                raise ValueError('필수 파일 누락: ' + name)
        if connected:
            from backend.settings import settings, connection_missing
            from backend.copilot import SOURCES
            config = settings()
            missing = connection_missing(config)
            if missing:
                raise ValueError('실제 연결 설정 누락: ' + ', '.join(missing))
            from firebase_admin import credentials
            try:
                credentials.Certificate(config['service_account_file'])
            except Exception:
                raise ValueError('Firebase SDK 인증 키를 읽을 수 없습니다. JSON 파일을 확인하세요.') from None
            from backend.auth_log import PiiCipher
            if PiiCipher.from_config(config) is None:
                print('안내: LOG_ENC_KEY/LOG_HMAC_KEY가 없어 로그인 기록을 저장하지 않습니다. '
                      '.venv\\Scripts\\python.exe -m backend.auth_log init-keys 로 만들 수 있습니다.', flush=True)
            for _, relative in SOURCES.values():
                if not (ROOT / relative).is_file():
                    raise ValueError('Copilot 문서 누락: ' + relative)
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from None
    print(f"Ready: {repo.manifest['dataVersion']} / {len(repo.docs)} documents / {'connected' if connected else 'demo'}")
