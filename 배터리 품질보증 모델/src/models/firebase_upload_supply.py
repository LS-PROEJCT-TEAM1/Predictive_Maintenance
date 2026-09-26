# -*- coding: utf-8 -*-
"""
공급망 발주량 예측 → Firestore 업로드

담당: 김유빈 / DB화: 최우찬
입력: data/team/dashboard_data/firestore/*.jsonl

유빈님이 이미 Firestore 형식(JSON Lines, 한 줄이 문서 하나)으로 내보냈으므로
변환하지 않고 그대로 올린다. 문서 ID는 각 줄의 document_id 를 쓴다.

컬렉션 이름
    원본은 접두사가 없어 parts / alerts 처럼 일반적인 이름이다.
    다른 모듈과 섞이지 않도록 supply_ 를 붙여 올린다.
    (원래 이름 그대로 올리려면 --prefix "" )

        supply_parts                부품 마스터 117      → 드롭다운
        supply_daily_history        일자별 실적 5,478    → 실적 추이 그래프
        supply_forecasts            예측 결과 764        → 예측 vs 실제
        supply_part_metrics         부품별 성능 666
        supply_alerts               경보 111             → 알림 목록
        supply_daily_summary        일자 합계 7          → KPI
        supply_model_metrics        모델 성능 6
        supply_walk_forward_metrics 워크포워드 18
        supply_data_quality         데이터 검증 23
        supply_dashboard_config     대시보드 설정 1

사전 준비
    1) data/team/dashboard_data/firestore/ 에 jsonl 파일이 있을 것
    2) serviceAccount.json 이 프로젝트 루트에 있을 것
    3) pip install firebase-admin

실행
    python src/models/firebase_upload_supply.py --dry-run
    python src/models/firebase_upload_supply.py
    python src/models/firebase_upload_supply.py --only parts,forecasts
"""

import os
import sys
import json
import glob
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "data", "team", "dashboard_data", "firestore")
DEFAULT_KEY = os.path.join(ROOT, "serviceAccount.json")

BATCH = 400                 # Firestore 배치 한도는 500. 여유를 둔다.
WRITE_LIMIT = 20_000        # 무료 쓰기 한도 (하루)


def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as fp:
        for i, line in enumerate(fp, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                sys.exit(f"[오류] {os.path.basename(path)} {i}번째 줄을 읽을 수 없습니다: {e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=DEFAULT_KEY, help="서비스 계정 키 JSON 경로")
    ap.add_argument("--dry-run", action="store_true", help="업로드 없이 점검만")
    ap.add_argument("--prefix", default="supply_", help='컬렉션 접두사 (원본 그대로 쓰려면 "")')
    ap.add_argument("--only", default="", help="올릴 컬렉션만 쉼표로 지정")
    a = ap.parse_args()

    if not os.path.isdir(SRC):
        sys.exit(f"[오류] {SRC} 가 없습니다.\n"
                 "       발주량 예측 모델/outputs/dashboard_data 를 data/team/ 에 넣으세요.")

    manifest_path = os.path.join(SRC, "manifest.json")
    manifest = json.load(open(manifest_path, encoding="utf-8")) if os.path.exists(manifest_path) else None

    files = sorted(glob.glob(os.path.join(SRC, "*.jsonl")))
    if not files:
        sys.exit(f"[오류] {SRC} 에 jsonl 파일이 없습니다.")

    want = {s.strip() for s in a.only.split(",") if s.strip()}
    plan, total_docs, total_bytes = [], 0, 0
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        if want and name not in want:
            continue
        rows = read_jsonl(f)
        missing = sum(1 for r in rows if not r.get("document_id"))
        plan.append((name, rows, missing))
        total_docs += len(rows)
        total_bytes += os.path.getsize(f)

    print("=" * 66)
    print("  공급망 발주량 예측 업로드 대상")
    print("=" * 66)
    for name, rows, missing in plan:
        warn = f"   [경고] document_id 없는 줄 {missing}개" if missing else ""
        print(f"  {a.prefix + name:32s} {len(rows):6,d}건{warn}")
    print(f"  {'-'*62}")
    print(f"  {'총 문서':32s} {total_docs:6,d}건   (무료 쓰기 한도 {WRITE_LIMIT:,}/일)")
    print(f"  {'총 용량':32s} {total_bytes/1024/1024:6.2f} MB")
    print("=" * 66)

    if manifest:
        counts = {c["collection"]: c["document_count"] for c in manifest["collections"]}
        bad = [(n, len(r), counts.get(n)) for n, r, _ in plan
               if n in counts and counts[n] != len(r)]
        if bad:
            print("\n[경고] manifest.json 의 문서 수와 다릅니다:")
            for n, got, exp in bad:
                print(f"  {n}: 파일 {got}건 / manifest {exp}건")

    if total_docs > WRITE_LIMIT:
        print(f"\n[중단] 문서 {total_docs:,}건은 하루 쓰기 한도 {WRITE_LIMIT:,}건을 넘습니다.")
        print("       --only 로 나눠서 올리세요.")
        if not a.dry_run:
            return

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

    print("\n업로드 시작...")
    done = 0
    for name, rows, _ in plan:
        col = db.collection(a.prefix + name)
        batch, n = db.batch(), 0
        for r in rows:
            doc_id = r.get("document_id")
            batch.set(col.document(doc_id) if doc_id else col.document(), r)
            n += 1
            if n % BATCH == 0:
                batch.commit()
                batch = db.batch()
                print(f"    {name} {n:,}/{len(rows):,}")
        batch.commit()
        done += len(rows)
        print(f"  {a.prefix + name:32s} {len(rows):6,d}건 완료   (누적 {done:,})")

    print(f"\n완료. 총 {done:,}건.")
    print(f"점검: python src/models/firebase_check.py --prefix {a.prefix}")


if __name__ == "__main__":
    main()
