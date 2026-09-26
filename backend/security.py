from __future__ import annotations

import secrets
import time
from collections import defaultdict, deque
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

COOKIE = 'manufacturing_session'
CSRF = 'manufacturing_csrf'
ASSETS = Path(__file__).resolve().parents[1] / 'frontend' / 'assets'
BRAND_FONTS = {'notokr-demilight.woff', 'notokr-medium.woff', 'notokr-bold.woff'}
PUBLIC_BRAND = {'/auth/ci_img02.png', *('/auth/fonts/'+name for name in BRAND_FONTS)}


class Login(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)


def install_security(app, service):
    attempts = defaultdict(deque)
    attempt_lock = Lock()

    @app.middleware('http')
    async def identity(request: Request, call_next):
        path = request.url.path
        try:
            if request.method not in ('GET', 'HEAD', 'OPTIONS'):
                origin = request.headers.get('origin', '')
                parsed = urlsplit(origin)
                if parsed.netloc != request.headers.get('host') or parsed.scheme not in ('http', 'https'):
                    raise HTTPException(403, '요청 출처를 확인할 수 없습니다. 로그인 화면을 다시 열어 주세요.')
                if not path.startswith('/_dash-'):
                    cookie, header = request.cookies.get(CSRF, ''), request.headers.get('x-csrf-token', '')
                    if not cookie or not secrets.compare_digest(cookie, header):
                        raise HTTPException(403, '요청 확인이 만료되었습니다. 화면을 새로고침하세요.')
            public = path in ('/login', '/auth/login', '/auth/theme.css', '/api/health') or path in PUBLIC_BRAND
            if not public:
                user = await run_in_threadpool(service.verify, request.cookies.get(COOKIE),
                    fresh=request.method not in ('GET', 'HEAD') and not path.startswith('/_dash-'))
                request.state.user = user
            response = await call_next(request)
        except HTTPException as exc:
            if exc.status_code == 401 and request.method == 'GET' and 'text/html' in request.headers.get('accept', ''):
                response = RedirectResponse('/login', status_code=303)
            else:
                response = JSONResponse({'detail': exc.detail}, status_code=exc.status_code)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Referrer-Policy'] = 'same-origin'
        return response

    @app.get('/login', response_class=HTMLResponse, include_in_schema=False)
    def login_page():
        page = 'demo.html' if getattr(service, 'mode', None) == 'demo' else 'login.html'
        response = HTMLResponse((ASSETS.parent / page).read_text(encoding='utf-8'))
        response.set_cookie(CSRF, secrets.token_urlsafe(32), samesite='strict', max_age=28800)
        return response

    @app.get('/auth/theme.css', include_in_schema=False)
    def theme():
        return FileResponse(ASSETS / 'theme.css', media_type='text/css')

    @app.get('/auth/ci_img02.png', include_in_schema=False)
    def brand_logo():
        return FileResponse(ASSETS / 'ci_img02.png', media_type='image/png')

    @app.get('/auth/fonts/{filename}', include_in_schema=False)
    def brand_font(filename: str):
        if filename not in BRAND_FONTS:
            raise HTTPException(404, '글꼴을 찾을 수 없습니다.')
        return FileResponse(ASSETS / 'fonts' / filename, media_type='font/woff')

    @app.post('/auth/login', include_in_schema=False)
    def login(body: Login, request: Request):
        host = request.client.host if request.client else 'unknown'
        with attempt_lock:
            queue = attempts[host]
            while queue and queue[0] < time.monotonic()-60:
                queue.popleft()
            if len(queue) >= 10:
                raise HTTPException(429, '로그인 시도가 많습니다. 1분 후 다시 시도하세요.')
            queue.append(time.monotonic())
        cookie, user = service.login(body.email.strip(), body.password)
        response = JSONResponse(user)
        response.set_cookie(COOKIE, cookie, max_age=28800, httponly=True, samesite='strict',
                            secure=request.url.scheme == 'https')
        response.set_cookie(CSRF, secrets.token_urlsafe(32), max_age=28800, samesite='strict',
                            secure=request.url.scheme == 'https')
        return response

    @app.get('/api/me')
    def me(request: Request):
        return request.state.user

    @app.post('/auth/logout')
    def logout(request: Request):
        service.logout(request.cookies.get(COOKIE))
        response = JSONResponse({'ok': True})
        response.delete_cookie(COOKIE)
        response.delete_cookie(CSRF)
        return response
