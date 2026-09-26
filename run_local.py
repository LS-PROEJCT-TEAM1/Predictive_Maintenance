"""Run the unified local-only FastAPI + Dash application."""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="BatteryFlow AI 운영센터 로컬 통합 화면")
    parser.add_argument("--port", type=int, default=8070)
    parser.add_argument('--demo', action='store_true', help='계정·외부 연결 없는 로컬 분석 체험')
    parser.add_argument('--check', action='store_true', help='필수 파일과 실행 환경만 검사')
    args = parser.parse_args()
    # Real authentication remains the default; demo must be explicitly selected.
    os.environ['MANUFACTURING_MODE'] = 'demo' if args.demo else 'connected'
    from backend.preflight import check
    check(connected=not args.demo)
    if args.check:
        return
    os.environ["MANUFACTURING_API_URL"] = f"http://127.0.0.1:{args.port}"
    import uvicorn
    print(f"Dashboard: http://127.0.0.1:{args.port}", flush=True)
    print(f"API docs: http://127.0.0.1:{args.port}/docs", flush=True)
    uvicorn.run("backend.main:app", host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
