# -*- coding: utf-8 -*-
"""
Firestore 업로드

extract_firebase.py가 만든 output/firebase/ 의 JSON을 Firestore에 올린다.
extract_dashboard.py를 함께 돌렸다면 dashboard.json의 내용(모듈 온도 채널,
4대 불량 유형 지표)도 같은 문서에 얹어 올린다. 문서를 새로 만들지 않고
기존 문서에 필드를 더하는 방식이라 문서 수와 화면당 읽기 횟수는 그대로다.

사전 준비
    1) Firebase 콘솔 → 프로젝트 설정 → 서비스 계정 → 새 비공개 키 생성
    2) 내려받은 JSON을 프로젝트 루트에 serviceAccount.json 으로 저장
       (이 파일은 비밀번호와 같다. 절대 GitHub에 올리지 말 것)
    3) pip install firebase-admin

실행
    python src/models/firebase_upload.py                 # 전체 업로드
    python src/models/firebase_upload.py --dry-run       # 올리지 않고 점검만
    python src/models/firebase_upload.py --key 경로.json # 키 위치 지정

컬렉션 구조
    battery_meta/config            메타 (셀 ID, 관리한계, 모델 정보)
    battery_meta/dashboard         4대 불량 유형 임계값, 온도 센서 ID   ← dashboard.json
    battery_packs/{packId}         팩 마스터  → 드롭다운, KPI 카드
                                   + metrics/flags (4대 불량 유형)      ← dashboard.json
    battery_packs/{packId}/series/score     시계열 배열 → 이상 점수 그래프
    battery_packs/{packId}/heatmap/cells    셀 스냅샷   → Heatmap
                                   + temp_snapshots (모듈 온도 채널)    ← dashboard.json
    battery_segments/{packId}      이상 구간 목록
    battery_models/{modelId}       모델 성능지표
"""

import os
import sys
import json
import glob
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "output", "firebase")
DEFAULT_KEY = os.path.join(ROOT, "serviceAccount.json")


def load(path):
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=DEFAULT_KEY, help="서비스 계정 키 JSON 경로")
    ap.add_argument("--dry-run", action="store_true", help="업로드 없이 점검만")
    ap.add_argument("--prefix", default="battery_", help="컬렉션 이름 접두사")
    ap.add_argument("--include-test", action="store_true",
                    help="모델 검증용 시험 팩(Test01~09)도 함께 올린다")
    a = ap.parse_args()

    if not os.path.isdir(SRC):
        sys.exit(f"[오류] {SRC} 가 없습니다. 먼저 extract_firebase.py 를 실행하세요.")

    packs = load(os.path.join(SRC, "packs.json"))
    meta = load(os.path.join(SRC, "meta.json"))
    segments = load(os.path.join(SRC, "segments.json"))
    models = load(os.path.join(SRC, "models.json"))
    series_files = sorted(glob.glob(os.path.join(SRC, "series", "*.json")))
    heat_files = sorted(glob.glob(os.path.join(SRC, "heatmap", "*.json")))

    # 시험 팩은 모델을 검증하려고 쓴 라벨 데이터라 대시보드에 올리지 않는다.
    if not a.include_test:
        drop = {p["pack_id"] for p in packs if p["source"] == "test"}
        packs = [p for p in packs if p["pack_id"] not in drop]
        segments = [s for s in segments if s["pack_id"] not in drop]
        series_files = [f for f in series_files
                        if os.path.splitext(os.path.basename(f))[0] not in drop]
        heat_files = [f for f in heat_files
                      if os.path.splitext(os.path.basename(f))[0] not in drop]
        if drop:
            print(f"시험 팩 {len(drop)}개 제외 (--include-test 로 포함 가능)\n")

    # dashboard.json — 있으면 얹고, 없으면 그냥 건너뛴다 (선택 단계)
    dash_path = os.path.join(SRC, "dashboard.json")
    dash = load(dash_path) if os.path.exists(dash_path) else None
    if dash:
        for p in packs:                                  # 팩 문서에 4대 불량 유형 붙이기
            d = dash["packs"].get(p["pack_id"])
            if d:
                p["metrics"] = d["metrics"]
                p["flags"] = d["flags"]
                p["t_peak"] = d["t_peak"]
                p["t_last"] = d["t_last"]

    n_docs = 1 + len(packs) + len(series_files) + len(heat_files) \
        + len({s["pack_id"] for s in segments}) + len(models) + (1 if dash else 0)
    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(SRC) for f in fs)

    print("=" * 62)
    print("  업로드 대상")
    print("=" * 62)
    print(f"  팩 마스터     {len(packs):5d}개")
    print(f"  시계열        {len(series_files):5d}개")
    print(f"  Heatmap      {len(heat_files):5d}개")
    print(f"  이상 구간     {len(segments):5d}건")
    print(f"  모델 지표     {len(models):5d}개")
    if dash:
        hit = {k: sum(1 for d in dash["packs"].values() if d["flags"][k])
               for k in dash["thresholds"]}
        print(f"  대시보드      온도 채널 + 4대 불량 유형")
        print(f"                {', '.join(f'{k} {v}팩' for k, v in hit.items())}")
    else:
        print(f"  대시보드      없음 (extract_dashboard.py 미실행)")
    print(f"  ----------------------------")
    print(f"  총 문서       {n_docs:5d}개   (무료 쓰기 한도 20,000/일)")
    print(f"  총 용량       {size/1024/1024:5.1f} MB (무료 저장 한도 1 GiB)")
    print("=" * 62)

    # 문서 크기 한도(1 MiB) 사전 점검
    over = [(os.path.basename(f), os.path.getsize(f))
            for f in series_files + heat_files if os.path.getsize(f) > 1_000_000]
    if over:
        print("\n[경고] 1 MiB에 근접한 문서:")
        for n, s in over:
            print(f"  {n} {s/1024:.0f} KB")

    if a.dry_run:
        print("\n--dry-run 이므로 업로드하지 않고 종료합니다.")
        return

    if not os.path.exists(a.key):
        sys.exit(f"\n[오류] 서비스 계정 키가 없습니다: {a.key}\n"
                 "       Firebase 콘솔 → 프로젝트 설정 → 서비스 계정 →\n"
                 "       새 비공개 키 생성 → serviceAccount.json 으로 저장")

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
    except ImportError:
        sys.exit("[오류] firebase-admin 이 없습니다.  pip install firebase-admin")

    firebase_admin.initialize_app(credentials.Certificate(a.key))
    db = firestore.client()
    P = a.prefix

    print("\n업로드 시작...")

    # --- 메타 ---
    db.collection(f"{P}meta").document("config").set(meta)
    print("  메타 1건")

    if dash:
        db.collection(f"{P}meta").document("dashboard").set({
            "thresholds": dash["thresholds"],
            "temp_ids": dash["temp_ids"],
            "n_normal_ref": dash["n_normal_ref"],
        })
        print("  대시보드 메타 1건")

    # --- 팩 마스터 + 하위 문서 (배치 500건 단위) ---
    batch, n = db.batch(), 0
    for p in packs:
        batch.set(db.collection(f"{P}packs").document(p["pack_id"]), p)
        n += 1
        if n % 400 == 0:
            batch.commit(); batch = db.batch()
    batch.commit()
    print(f"  팩 마스터 {len(packs)}건")

    for i, f in enumerate(series_files, 1):
        d = load(f)
        db.collection(f"{P}packs").document(d["pack_id"]) \
          .collection("series").document("score").set(d)
        if i % 25 == 0:
            print(f"    시계열 {i}/{len(series_files)}")
    print(f"  시계열 {len(series_files)}건")

    for i, f in enumerate(heat_files, 1):
        d = load(f)
        if dash:                                   # 모듈 온도 채널을 같은 문서에 얹는다
            extra = dash["packs"].get(d["pack_id"])
            if extra and extra.get("temp"):
                d["temp_snapshots"] = extra["temp"]
        db.collection(f"{P}packs").document(d["pack_id"]) \
          .collection("heatmap").document("cells").set(d)
        if i % 25 == 0:
            print(f"    Heatmap {i}/{len(heat_files)}")
    print(f"  Heatmap {len(heat_files)}건")

    # --- 이상 구간: 팩 단위로 묶어 문서 1개 ---
    by_pack = {}
    for s in segments:
        by_pack.setdefault(s["pack_id"], []).append(s)
    batch, n = db.batch(), 0
    for pid, segs in by_pack.items():
        batch.set(db.collection(f"{P}segments").document(pid),
                  {"pack_id": pid, "n": len(segs), "items": segs})
        n += 1
        if n % 400 == 0:
            batch.commit(); batch = db.batch()
    batch.commit()
    print(f"  이상 구간 {len(by_pack)}건")

    # --- 모델 지표 ---
    for m in models:
        db.collection(f"{P}models").document(
            m["model"].replace("/", "_").replace(" ", "_")).set(m)
    print(f"  모델 지표 {len(models)}건")

    print("\n완료. Firebase 콘솔 → Firestore Database 에서 확인하세요.")


if __name__ == "__main__":
    main()
