"""Only paths/public configuration live here; private credentials stay outside Git."""
import json
import os
from pathlib import Path
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


def settings():
    path = Path(os.environ.get('MANUFACTURING_SETTINGS', ROOT / '.local/settings.json'))
    config = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    env_path = config.get('connections_file')
    private = dotenv_values(env_path) if env_path else {}
    return {
        **config,
        'project_id': config.get('project_id', 'ls-proejct-team1'),
        'gemini_key': os.environ.get('GEMINI_API_KEY') or private.get('GEMINI_API_KEY', ''),
        'gemini_model': os.environ.get('GEMINI_MODEL') or config.get('gemini_model', 'gemini-2.5-flash'),
        'embedding_model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        'rag_dir': ROOT / '.local/rag',
    }
