"""List/copy the exact clone-test payload; excludes credentials and research archives."""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.copilot import SOURCES


def files():
    selected = set()
    for directory in ('backend', 'frontend', 'runtime', 'firestore/seed'):
        for path in (ROOT / directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.log'):
                selected.add(path)
    for relative in ('README.md', 'DESIGN.md', 'AGENTS.md', '.gitignore', '.gitattributes', 'LOCAL_RUN.md', 'TEAM_SETUP.md',
                     'requirements-local.txt', 'run_local.py', 'setup_local.cmd', 'start_demo.cmd',
                     'start_local.cmd', 'start_local.ps1', 'scripts/team_bundle.py',
                     'scripts/package_runtime.py', '발주량 예측 모델/src/inference.py'):
        selected.add(ROOT / relative)
    for _, relative in SOURCES.values():
        selected.add(ROOT / relative)
    for path in sorted(selected):
        if not path.is_file():
            raise SystemExit('Required file missing: ' + str(path.relative_to(ROOT)))
        if path.stat().st_size >= 100 * 1024**2:
            raise SystemExit('File exceeds GitHub regular Git limit: ' + str(path.relative_to(ROOT)))
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--copy-to', type=Path, help='New empty directory for clean-clone verification')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    selected = files()
    if args.copy_to:
        destination = args.copy_to.resolve()
        if destination.exists():
            raise SystemExit('Use a new empty destination; existing files are never overwritten.')
        destination.mkdir(parents=True)
        for path in selected:
            target = destination / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    if args.list:
        for path in selected:
            print(path.relative_to(ROOT).as_posix())
    print(f'{len(selected)} files / {sum(p.stat().st_size for p in selected)/1048576:.2f} MiB')


if __name__ == '__main__':
    main()
