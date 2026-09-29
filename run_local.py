"""Run the unified local-only FastAPI + Dash application."""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="BatteryFlow AI 운영센터 로컬 통합 화면")
    parser.add_argument("--port", type=int, default=8070)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--demo', action='store_true', help='계정·외부 연결 없는 로컬 분석 체험')
    modes.add_argument('--connected', action='store_true', help='실제 연결 필수; 설정 누락 시 중단')
    parser.add_argument('--check', action='store_true', help='필수 파일과 실행 환경만 검사')
    args = parser.parse_args()
    from backend.settings import resolve_mode
    try:
        mode, missing = resolve_mode('demo' if args.demo else 'connected' if args.connected else None)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    os.environ['MANUFACTURING_MODE'] = mode
    if missing:
        print('로컬 체험 모드: 연결 설정 누락 - ' + ', '.join(missing), flush=True)
        print('실제 로그인·Firestore 저장·Gemini는 비활성화됩니다. .env와 SDK를 루트에 넣고 재시작하세요.', flush=True)
    from backend.preflight import check
    check(connected=mode == 'connected')
    if args.check:
        return
    os.environ["MANUFACTURING_API_URL"] = f"http://127.0.0.1:{args.port}"
    import uvicorn
    print(f"Dashboard: http://127.0.0.1:{args.port}", flush=True)
    print(f"API docs: http://127.0.0.1:{args.port}/docs", flush=True)
    uvicorn.run("backend.main:app", host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
