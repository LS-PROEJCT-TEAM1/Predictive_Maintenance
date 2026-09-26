# -*- coding: utf-8 -*-
"""
설비 예지보전(LightGBM) → Firestore 업로드

extract_pdm.py 가 만든 output/firebase/pdm/ 의 JSON을 올린다.
배터리 모듈과 컬렉션이 겹치지 않도록 접두사를 pdm_ 으로 쓴다.

사전 준비
    1) python src/models/extract_pdm.py
    2) serviceAccount.json 이 프로젝트 루트에 있을 것
       (이 파일은 비밀번호와 같다. 절대 GitHub에 올리지 말 것)
    3) pip install firebase-admin

실행
    python src/models/firebase_upload_pdm.py --dry-run    # 문서 수만 확인
    python src/models/firebase_upload_pdm.py              # 업로드

컬렉션 구조
    pdm_meta/config             임계값, 분할 정보, 설비 정보, 변수 중요도
    pdm_files/{fileId}          파일 마스터   → 드롭다운, KPI 카드
      └ series/predictions      행 단위 배열  → 실제vs예측 / 이상점수 그래프
      └ modules/judgement       모듈 판정 배열 → 모듈 판정 표
    pdm_segments/{fileId}       이상 구간     → 점검 목록, 우선순위
    pdm_models/lightgbm         성능 지표
    pdm_eda/{docId}             EDA / 참고 표 6종
"""

import os
import sys
import json
import glob
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "output", "firebase", "pdm")
DEFAULT_KEY = os.path.join(ROOT, "serviceAccount.json")

DOC_LIMIT = 1_000_000          # Firestore 문서 1 MiB


def load(path):
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=DEFAULT_KEY, help="서비스 계정 키 JSON 경로")
    ap.add_argument("--dry-run", action="store_true", help="업로드 없이 점검만")
    ap.add_argument("--prefix", default="pdm_", help="컬렉션 이름 접두사")
    a = ap.parse_args()

    if not os.path.isdir(SRC):
        sys.exit(f"[오류] {SRC} 가 없습니다.\n"
                 "       python src/models/extract_pdm.py 를 먼저 실행하세요.")

    files = load(os.path.join(SRC, "files.json"))
    meta = load(os.path.join(SRC, "meta.json"))
    models = load(os.path.join(SRC, "models.json"))
    eda = load(os.path.join(SRC, "eda.json"))
    series_files = sorted(glob.glob(os.path.join(SRC, "series", "*.json")))
    module_files = sorted(glob.glob(os.path.join(SRC, "modules", "*.json")))
    seg_files = sorted(glob.glob(os.path.join(SRC, "segments", "*.json")))

    n_docs = 1 + len(files) + len(series_files) + len(module_files) \
        + len(seg_files) + 1 + len(eda)
    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(SRC) for f in fs)

    print("=" * 62)
    print("  설비 예지보전 업로드 대상")
    print("=" * 62)
    print(f"  파일 마스터   {len(files):5d}개")
    print(f"  시계열        {len(series_files):5d}개")
    print(f"  모듈 판정     {len(module_files):5d}개")
    print(f"  이상 구간     {len(seg_files):5d}개")
    print(f"  EDA / 참고표  {len(eda):5d}개")
    print(f"  모델 지표         1개")
    print(f"  ----------------------------")
    print(f"  총 문서       {n_docs:5d}개   (무료 쓰기 한도 20,000/일)")
    print(f"  총 용량       {size/1024/1024:5.2f} MB (무료 저장 한도 1 GiB)")
    print("=" * 62)

    over = [(os.path.relpath(f, SRC), os.path.getsize(f))
            for f in series_files + module_files + seg_files
            if os.path.getsize(f) > 0.9 * DOC_LIMIT]
    if over:
        print("\n[경고] 1 MiB에 근접한 문서:")
        for n, s in over:
            print(f"  {n} {s/1024:.0f} KB")
        print("  extract_pdm.py --max-points 를 줄여 다시 만드세요.")

    if a.dry_run:
        print("\n--dry-run 이므로 업로드하지 않고 종료합니다.")
        return

    if not os.path.exists(a.key):
        sys.exit(f"\n[오류] 서비스 계정 키가 없습니다: {a.key}")

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
    except ImportError:
        sys.exit("[오류] firebase-admin 이 없습니다.  pip install firebase-admin")

    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(a.key))
    db = firestore.client()
    P = a.prefix

    print("\n업로드 시작...")

    db.collection(f"{P}meta").document("config").set(meta)
    print("  메타 1건")

    for f in files:
        db.collection(f"{P}files").document(f["file_id"]).set(f)
    print(f"  파일 마스터 {len(files)}건")

    for f in series_files:
        d = load(f)
        db.collection(f"{P}files").document(d["file_id"]) \
          .collection("series").document("predictions").set(d)
    print(f"  시계열 {len(series_files)}건")

    for f in module_files:
        d = load(f)
        db.collection(f"{P}files").document(d["file_id"]) \
          .collection("modules").document("judgement").set(d)
    print(f"  모듈 판정 {len(module_files)}건")

    for f in seg_files:
        d = load(f)
        db.collection(f"{P}segments").document(d["file_id"]).set(d)
    print(f"  이상 구간 {len(seg_files)}건")

    db.collection(f"{P}models").document("lightgbm").set(models)
    print("  모델 지표 1건")

    for key, val in eda.items():
        db.collection(f"{P}eda").document(key).set(val)
    print(f"  EDA / 참고표 {len(eda)}건")

    print("\n완료. Firebase 콘솔 → Firestore Database 에서 확인하세요.")
    print("점검: python src/models/firebase_check.py --prefix pdm_")


if __name__ == "__main__":
    main()
