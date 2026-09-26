# -*- coding: utf-8 -*-
"""
Firestore 점검 — 무엇이 올라가 있는지 한 번에 확인

세 모듈(배터리 품질 / 설비 예지보전 / 공급망 예측)이 모두 올라갔는지,
문서 수가 기대값과 맞는지 본다. 읽기만 하므로 데이터를 바꾸지 않는다.

실행
    python src/models/firebase_check.py                 # 전체 요약
    python src/models/firebase_check.py --deep          # 하위 문서까지
    python src/models/firebase_check.py --prefix pdm_   # 한 모듈만
    python src/models/firebase_check.py --key 경로.json

읽기 비용
    문서 수를 셀 때 집계 쿼리(count)를 쓰므로 읽기 1건만 든다.
    집계를 못 쓰는 환경이면 문서를 훑으며 세고, 그만큼 읽기가 발생한다.
"""

import os
import re
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_KEY = os.path.join(ROOT, "serviceAccount.json")

TEST_RE = re.compile(r"^Test\d+_", re.IGNORECASE)

# 모듈별 기대 문서 수 (추출 스크립트가 찍어준 값)
EXPECTED = {
    "배터리 품질": {
        "battery_meta": 2, "battery_packs": 102,
        "battery_segments": 41, "battery_models": 5,
    },
    "설비 예지보전": {
        "pdm_meta": 1, "pdm_files": 4, "pdm_segments": 3,
        "pdm_models": 1, "pdm_eda": 6,
    },
    "공급망 예측": {
        "supply_parts": 117, "supply_daily_history": 5478,
        "supply_forecasts": 764, "supply_part_metrics": 666,
        "supply_alerts": 111, "supply_data_quality": 23,
        "supply_walk_forward_metrics": 18, "supply_daily_summary": 7,
        "supply_model_metrics": 6, "supply_dashboard_config": 1,
    },
}


def count(col):
    """집계 쿼리로 문서 수를 센다. 안 되면 훑어서 센다."""
    try:
        return int(col.count().get()[0][0].value), True
    except Exception:
        return sum(1 for _ in col.stream()), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=DEFAULT_KEY, help="서비스 계정 키 JSON 경로")
    ap.add_argument("--prefix", default="", help="이 접두사로 시작하는 컬렉션만")
    ap.add_argument("--deep", action="store_true", help="팩/파일별 하위 문서까지 확인 (느림)")
    a = ap.parse_args()

    if not os.path.exists(a.key):
        sys.exit(f"[오류] 서비스 계정 키가 없습니다: {a.key}")
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
    except ImportError:
        sys.exit("[오류] firebase-admin 이 없습니다.  pip install firebase-admin")

    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(a.key))
    db = firestore.client()

    cols = sorted(c.id for c in db.collections())
    if a.prefix:
        cols = [c for c in cols if c.startswith(a.prefix)]
    if not cols:
        sys.exit("[오류] 컬렉션이 하나도 없습니다. 업로드를 먼저 하세요.")

    counts, agg_ok = {}, True
    for c in cols:
        n, ok = count(db.collection(c))
        counts[c] = n
        agg_ok = agg_ok and ok

    print("=" * 64)
    print("  Firestore 점검")
    print("=" * 64)

    seen, total = set(), 0
    for module, exp in EXPECTED.items():
        rows = [(c, counts[c], exp.get(c)) for c in cols if c in exp]
        if not rows:
            print(f"\n  [{module}]  아직 올라가지 않았습니다.")
            continue
        missing = [c for c in exp if c not in counts]
        got = sum(n for _, n, _ in rows)
        total += got
        seen.update(c for c, _, _ in rows)
        print(f"\n  [{module}]  {got:,}건")
        for c, n, e in rows:
            mark = "" if e is None or n == e else f"   <- 기대 {e:,}건"
            print(f"    {c:26s} {n:6,d}건{mark}")
        if missing:
            print(f"    빠진 컬렉션: {', '.join(missing)}")

    extra = [c for c in cols if c not in seen]
    if extra:
        print("\n  [그 외]")
        for c in extra:
            total += counts[c]
            print(f"    {c:26s} {counts[c]:6,d}건")

    # 배터리: 시험 팩이 남아 있는지
    if "battery_packs" in counts:
        left = []
        for col in ("battery_packs", "battery_segments"):
            if col in counts:
                left += [d.id for d in db.collection(col).stream() if TEST_RE.match(d.id)]
        print("\n" + "-" * 64)
        if left:
            print("  배터리 시험 팩이 남아 있습니다:")
            for i in sorted(set(left)):
                print(f"    {i}")
        else:
            print("  배터리 시험 팩: 남아 있지 않습니다.")

    # 하위 문서
    if a.deep:
        print("\n" + "-" * 64)
        if "battery_packs" in counts:
            ids = [d.id for d in db.collection("battery_packs").stream()]
            s = sum(1 for i in ids if db.collection("battery_packs").document(i)
                    .collection("series").document("score").get().exists)
            h = sum(1 for i in ids if db.collection("battery_packs").document(i)
                    .collection("heatmap").document("cells").get().exists)
            print(f"  battery_packs  series {s}/{len(ids)}  heatmap {h}/{len(ids)}")
        if "pdm_files" in counts:
            ids = [d.id for d in db.collection("pdm_files").stream()]
            s = sum(1 for i in ids if db.collection("pdm_files").document(i)
                    .collection("series").document("predictions").get().exists)
            m = sum(1 for i in ids if db.collection("pdm_files").document(i)
                    .collection("modules").document("judgement").get().exists)
            print(f"  pdm_files      series {s}/{len(ids)}  modules {m}/{len(ids)}")

    done = sum(1 for exp in EXPECTED.values() if any(c in counts for c in exp))
    print("\n" + "=" * 64)
    print(f"  총 {total:,}건 / 컬렉션 {len(cols)}개 / 모듈 {done}/3 올라감")
    if not agg_ok:
        print("  (집계 쿼리를 쓸 수 없어 문서를 훑어 셌습니다. 그만큼 읽기가 발생했습니다.)")
    print("=" * 64)


if __name__ == "__main__":
    main()
