"""Run the unified local-only FastAPI + Dash application."""
import argparse
import os
import socket
import threading
import time
import webbrowser
from urllib.error import URLError
from urllib.request import urlopen


def login_ready(url, mode):
    try:
        with urlopen(url, timeout=2) as response:
            page = response.read(1_000_000).decode('utf-8')
        marker = 'type="password"' if mode == 'connected' else '체험 시작'
        return 'BatteryFlow' in page and marker in page
    except (OSError, URLError, UnicodeError):
        return False


def open_when_ready(url, mode):
    for _ in range(120):
        if login_ready(url, mode):
            webbrowser.open(url)
            return
        time.sleep(0.5)
    print(f'화면이 준비되면 브라우저에서 {url} 을 열어 주세요.', flush=True)


def lan_address():
    """Best-effort IPv4 of the active network adapter (no packets are sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 1))
            return probe.getsockname()[0]
    except OSError:
        return None


def main():
    parser = argparse.ArgumentParser(description="BatteryFlow AI 운영센터 로컬 통합 화면")
    parser.add_argument("--port", type=int, default=8070)
    parser.add_argument("--host", default="127.0.0.1",
                        help="접속을 받을 주소. 기본 127.0.0.1(이 PC만). 같은 Wi-Fi 공개는 0.0.0.0")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--demo', action='store_true', help='계정·외부 연결 없는 로컬 분석 체험')
    modes.add_argument('--connected', action='store_true', help='실제 연결 필수; 설정 누락 시 중단')
    parser.add_argument('--check', action='store_true', help='필수 파일과 실행 환경만 검사')
    parser.add_argument('--open-browser', action='store_true', help='준비되면 로그인 화면 열기')
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
    login_url = f'http://127.0.0.1:{args.port}/login'
    if args.open_browser:
        with socket.socket() as probe:
            probe.settimeout(1)
            occupied = probe.connect_ex(('127.0.0.1', args.port)) == 0
        if occupied:
            if login_ready(login_url, mode):
                print(f'이미 실행 중인 화면을 엽니다: {login_url}', flush=True)
                webbrowser.open(login_url)
                return
            raise SystemExit(f'{args.port} 포트에서 다른 서비스 또는 다른 모드가 실행 중입니다. 기존 서버를 종료한 뒤 다시 실행하세요.')
        threading.Thread(target=open_when_ready, args=(login_url, mode), daemon=True).start()
    # Dash talks to the API inside this PC, whatever address the server listens on.
    os.environ["MANUFACTURING_API_URL"] = f"http://127.0.0.1:{args.port}"
    import uvicorn
    print(f"Dashboard: http://127.0.0.1:{args.port}", flush=True)
    print(f"API docs: http://127.0.0.1:{args.port}/docs", flush=True)
    if args.host != "127.0.0.1":
        address = lan_address()
        print(f"같은 Wi-Fi 공개 모드: 다른 기기는 http://{address or '이PC의IP'}:{args.port} 로 접속합니다.", flush=True)
        print(f"http(암호화 없음)입니다. 신뢰할 수 있는 네트워크에서만 사용하고, 방화벽 {args.port} 허용과 허용 주소 설정이 필요합니다.", flush=True)
    uvicorn.run("backend.main:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
