"""대시보드 데이터 읽기 창구.

화면 코드는 파일 경로나 Firestore 를 직접 부르지 않고 load(table) 하나만 사용한다.

    DATA_BACKEND="csv"       (기본) 프로젝트루트/outputs/lightgbm/{table}.csv
    DATA_BACKEND="firestore" 컬렉션 {table}, 문서 1개 = 1행, 필드명 = CSV 컬럼명

읽은 결과는 lru_cache 로 캐시한다. 업로드·재분석 뒤에는 load.cache_clear() 를 호출한다.
반환된 DataFrame 은 캐시와 공유되므로, 컬럼을 추가·수정할 때는 .copy() 후에 작업한다.
"""

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

DATA_BACKEND = os.getenv("DATA_BACKEND", "csv").lower()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = PROJECT_ROOT / "outputs" / "lightgbm"


def _load_csv(table):
    file_path = CSV_DIR / f"{table}.csv"
    return pd.read_csv(file_path, encoding="utf-8-sig")


def _load_firestore(table):
    # firebase-admin 은 Firestore 를 쓸 때만 필요하므로 여기서 불러온다.
    # 자격증명: 환경변수 GOOGLE_APPLICATION_CREDENTIALS (서비스 계정 JSON 경로)
    import firebase_admin
    from firebase_admin import firestore

    if not firebase_admin._apps:
        firebase_admin.initialize_app()

    db = firestore.client()
    rows = [doc.to_dict() for doc in db.collection(table).stream()]
    return pd.DataFrame(rows)


@lru_cache(maxsize=None)
def load(table):
    """table 이름(예: "predictions")에 해당하는 표를 DataFrame 으로 돌려준다."""
    if DATA_BACKEND == "csv":
        return _load_csv(table)
    if DATA_BACKEND == "firestore":
        return _load_firestore(table)
    raise ValueError(f"알 수 없는 DATA_BACKEND: {DATA_BACKEND}")
