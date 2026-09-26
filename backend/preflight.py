"""Fail early with actionable setup errors without printing secrets."""
import importlib.util
import sys
from pathlib import Path
from backend.data import Repository
from backend.runtime_assets import ROOT, verify_runtime


def check(connected=False):
    if sys.version_info[:2] != (3, 12):
        raise SystemExit('Python 3.12로 setup_local.cmd를 실행하세요.')
    for module in ('dash', 'fastapi', 'xgboost', 'sklearn', 'joblib'):
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
            from backend.settings import settings
            from backend.copilot import SOURCES
            config = settings()
            if not Path(config.get('service_account_file') or '__missing__').is_file() or not config.get('firebase_web_key'):
                raise ValueError('Firebase 연결 설정이 필요합니다. TEAM_SETUP.md를 참고하거나 start_demo.cmd를 실행하세요.')
            for _, relative in SOURCES.values():
                if not (ROOT / relative).is_file():
                    raise ValueError('Copilot 문서 누락: ' + relative)
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from None
    print(f"Ready: {repo.manifest['dataVersion']} / {len(repo.docs)} documents / {'connected' if connected else 'demo'}")
