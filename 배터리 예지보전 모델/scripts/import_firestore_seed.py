from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED_DIR = ROOT / "firestore" / "seed"


def iter_documents(seed_dir: Path):
    manifest = json.loads((seed_dir / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        with (seed_dir / item["file"]).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                document = json.loads(line)
                if set(document) != {"path", "data"}:
                    raise ValueError(f"{item['file']}:{line_number}: path/data 형식이 아닙니다.")
                yield document


def validate(seed_dir: Path) -> int:
    count = 0
    seen = set()
    for document in iter_documents(seed_dir):
        path = document["path"]
        if path in seen:
            raise ValueError(f"중복 문서 경로: {path}")
        if len(path.split("/")) % 2:
            raise ValueError(f"Firestore 문서 경로가 아닙니다: {path}")
        if len(json.dumps(document["data"], ensure_ascii=False).encode("utf-8")) >= 1_000_000:
            raise ValueError(f"1 MiB에 근접하거나 초과한 문서: {path}")
        seen.add(path)
        count += 1
    return count


def import_seed(seed_dir: Path, project_id: str | None) -> int:
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
    except ImportError as exc:
        raise SystemExit(
            "firebase-admin이 필요합니다. requirements-firestore.txt를 설치하세요."
        ) from exc

    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.ApplicationDefault(), {"projectId": project_id} if project_id else None)
    database = firestore.client()
    batch = database.batch()
    pending = 0
    total = 0
    for document in iter_documents(seed_dir):
        batch.set(database.document(document["path"]), document["data"])
        pending += 1
        total += 1
        if pending == 400:
            batch.commit()
            batch = database.batch()
            pending = 0
    if pending:
        batch.commit()
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Firestore JSONL 시드 검증 및 일괄 적재")
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--project-id")
    parser.add_argument("--apply", action="store_true", help="실제 Firestore에 적재")
    args = parser.parse_args()
    count = validate(args.seed_dir)
    if not args.apply:
        print(f"검증 완료: {count:,}개 문서 (변경 없음)")
        return
    imported = import_seed(args.seed_dir, args.project_id)
    print(f"Firestore 적재 완료: {imported:,}개 문서")


if __name__ == "__main__":
    main()
