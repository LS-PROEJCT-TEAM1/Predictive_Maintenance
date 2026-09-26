from __future__ import annotations

import io
import json
import os
import sys
import subprocess
import tempfile
import secrets
import time
from collections import OrderedDict
from uuid import uuid4
from threading import Lock
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd
from a2wsgi import WSGIMiddleware
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from backend.data import ROOT, Repository, records
from backend.runtime_assets import quality_path, RUNTIME
from backend.firebase_service import FirebaseService
from backend.security import install_security
from backend.copilot import Copilot
from google.api_core.exceptions import ResourceExhausted


class PreviewDecision(BaseModel):
    track: Literal["demand", "maintenance", "quality"]
    target: str = Field(min_length=1, max_length=250)
    decision: Literal["reviewed", "clear", "retest", "hold"]
    note: str = Field(default="", max_length=2000)
    date: str | None = None
    model: str | None = None
    source: Literal['screen', 'copilot'] = 'screen'
    conversationId: str | None = Field(None, pattern=r'^[a-f0-9]{32}$')


class ExportRequest(BaseModel):
    rows: list[dict] = Field(max_length=20000)
    name: str = Field(pattern=r"^[a-z_]{1,40}$")


class SaveDecision(PreviewDecision):
    requestId: str = Field(pattern=r'^[a-zA-Z0-9_-]{16,80}$')
    expectedRevision: int = Field(ge=0)


class ScreenContext(BaseModel):
    track: Literal['overview', 'demand', 'maintenance', 'quality'] = 'overview'
    date: str | None = Field(None, max_length=20)
    part: str = Field('ALL', max_length=60)
    model: str = Field('3-day Moving Average', max_length=80)
    run: str = Field('WeldingTest_04_NG', max_length=80)
    supervised: str | None = Field(None, max_length=80)
    unsupervised: str | None = Field(None, max_length=80)
    test: str = Field('Test07_NG_dchg', max_length=80)
    cell: str = Field('M02CV01', max_length=20)
    progress: int = Field(100, ge=1, le=100)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    threadId: str | None = Field(None, pattern=r'^[a-f0-9]{32}$')
    revision: int = Field(0, ge=0)
    context: ScreenContext = Field(default_factory=ScreenContext)


class DraftRequest(BaseModel):
    threadId: str = Field(pattern=r'^[a-f0-9]{32}$')
    target: str | None = Field(None, max_length=250)


def create_api(mount_ui=True, service=None, copilot_service=None):
    repo = Repository()
    demo = os.environ.get('MANUFACTURING_MODE') == 'demo' and service is None
    if demo:
        from backend.demo_service import DemoService, DemoCopilot
        service, copilot_service = DemoService(), DemoCopilot()
    service = service if service is not None else FirebaseService()
    copilot = copilot_service if copilot_service is not None else Copilot(repo)
    chat_inflight = set()
    chat_last = {}
    chat_lock = Lock()
    exports = OrderedDict()
    export_lock = Lock()
    state_cache = {'until': 0, 'value': None, 'unavailable': False}
    state_lock = Lock()
    app = FastAPI(title="BatteryFlow AI · Local API", version="0.2.0", description="공식 로컬 v2 시드 · Firebase 직원 인증/기록 · Gemini RAG Copilot")
    install_security(app, service)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    @app.exception_handler(ValueError)
    async def value_error(_, exc):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def service_error(_, exc):
        # Do not return SDK exception strings containing credential paths or tokens.
        return JSONResponse(status_code=503, content={'detail': '서비스 연결을 확인하지 못했습니다. 잠시 후 다시 시도하세요.'})

    @app.exception_handler(ResourceExhausted)
    async def quota_error(_, exc):
        return JSONResponse(status_code=503, content={'detail': 'Firestore 무료 할당량을 초과했습니다. 기록을 저장하거나 조회하지 못했습니다. 할당량 초기화 후 다시 시도하세요.'})

    @app.get("/api/health", tags=["상태"])
    def health():
        return {"status": "ok", "mode": "demo" if demo else "local-seed-with-cloud-records", "dataVersion": repo.manifest["dataVersion"], "documents": len(repo.docs), "firebase": "disabled" if demo else "configured", "copilot": "disabled" if demo else "gemini-local-faiss"}

    @app.get("/api/meta", tags=["상태"])
    def meta():
        return {**repo.meta(), 'mode': 'demo' if demo else 'local-seed-with-cloud-records'}

    @app.get("/api/overview", tags=["통합"])
    def overview():
        return {**repo.get("overview"), "actions": repo.collection("actions"), "trend": repo.demand()["trend"]}

    @app.get("/api/demand", tags=["공급망"])
    def demand(date: str | None = None, part: str = "ALL", model: str = "3-day Moving Average"):
        return repo.demand(date, part, model)

    @app.get("/api/maintenance", tags=["예지보전"])
    def maintenance(run: str = "WeldingTest_04_NG", supervised: str | None = None, unsupervised: str | None = None):
        return repo.maintenance(run, supervised, unsupervised)

    @app.get("/api/quality", tags=["품질"])
    def quality(test: str = "Test07_NG_dchg", cell: str = "M02CV01", progress: int = Query(100, ge=1, le=100)):
        return repo.quality(test, cell, progress)

    @app.get("/api/validation/{track}", tags=["검증"])
    def validation(track: Literal["demand", "maintenance", "quality"]):
        return repo.validation(track)

    @app.get("/api/data-quality/{track}", tags=["검증"])
    def data_quality(track: Literal["demand", "maintenance", "quality"]):
        return {"rows": repo.data_quality(track)}

    @app.get("/api/results/{track}", tags=["통합"])
    def results(track: Literal["demand", "maintenance", "quality"], kind: Literal["prediction", "evaluation", "raw", "processed"] = "prediction"):
        if kind == "evaluation":
            rows = repo.validation(track)["rows"]
        elif track == "demand":
            if kind == "prediction":
                rows = repo.demand()["rows"]
            elif kind == "processed":
                rows = [{"part": p["part_number"], **f} for p in repo.parts.values() for f in p["forecasts"]]
            else:
                rows = [{"part": p["part_number"], **h} for p in repo.parts.values() for h in p["history"]]
        elif track == "maintenance":
            data = repo.maintenance()
            rows = data["events"] if kind == "prediction" else data["points"]
            if kind == "raw":
                rows = [{k: row[k] for k in ["row", "cycle", "page", "time", "power", "label"]} for row in rows]
        else:
            data = repo.quality()
            if kind == "raw":
                frame = pd.read_csv(quality_path('Test07_NG_dchg'))
                indexes = np.unique(np.linspace(0, len(frame)-1, min(300, len(frame)), dtype=int))
                frame.insert(0, "sourceRow", np.arange(len(frame)))
                rows = records(frame.iloc[indexes])
            else:
                rows = data["series"] if kind == "prediction" else data["cellHeatmap"]
        return {"rows": rows, "dataVersion": repo.manifest["dataVersion"], "generatedAt": repo.manifest["generatedAt"],
                "notice": "공급망·예지보전 원본 유형은 시드의 측정·이력, 품질 원본은 Test07_NG_dchg CSV의 최대 300개 표본입니다. 가공 유형은 각각 목표일 정렬표·모델 점수·셀 위치 점수입니다."}

    @app.post("/api/preview/decision", tags=["체험"])
    def preview_decision(body: PreviewDecision):
        if body.track == "demand" and body.target not in repo.parts:
            raise ValueError("부품을 확인하세요.")
        if body.track == "demand":
            if not body.date or not body.model or not repo.demand(body.date, body.target, body.model)["count"]:
                raise ValueError("해당 날짜·모델의 예측이 없습니다.")
        if body.track == "quality" and body.target not in repo.meta()["tests"]:
            raise ValueError("시험 ID를 확인하세요.")
        if body.track == "maintenance":
            run, sup, unsup, _ = body.target.split(":", 3)
            if body.target not in {e["id"] for e in repo.maintenance(run, sup, unsup)["events"]}:
                raise ValueError("이벤트를 확인하세요.")
        if (body.track == "quality") == (body.decision == "reviewed"):
            raise ValueError("대상에 맞는 판정을 선택하세요.")
        key = f"{body.track}:{body.target}"
        if body.track == "demand":
            key += f":{body.date}:{body.model}"
        return {**body.model_dump(), "key": key, "persistence": "preview-only"}

    @app.get('/api/records/state', tags=['기록'])
    def record_state():
        with state_lock:
            if time.monotonic() < state_cache['until']:
                if state_cache['unavailable']:
                    raise HTTPException(503, 'Firestore 할당량 초과로 공유 기록을 조회하지 못했습니다. 화면에는 분석 결과만 표시됩니다.')
                return {'records': state_cache['value']}
            try:
                state_cache.update(value=service.states(), until=time.monotonic()+15, unavailable=False)
            except ResourceExhausted:
                state_cache.update(until=time.monotonic()+60, unavailable=True)
                raise
            return {'records': state_cache['value']}

    @app.get('/api/records', tags=['기록'])
    def record_history(key: str | None = Query(None, max_length=400)):
        return {'records': service.history(key)}

    @app.post('/api/records', tags=['기록'])
    def save_record(body: SaveDecision, request: Request):
        validated = preview_decision(body)
        if body.source == 'copilot':
            if not body.conversationId:
                raise ValueError('초안 대화를 확인하세요.')
            service.conversation(request.state.user['uid'], body.conversationId)
        result = service.save_record(body.model_dump(), validated['key'], request.state.user, repo.manifest['dataVersion'])
        with state_lock:
            state_cache['until'] = 0
        return result

    @app.get('/api/copilot/conversations', tags=['Copilot'])
    def conversations(request: Request):
        return {'threads': service.list_conversations(request.state.user['uid'])}

    @app.get('/api/copilot/conversations/{thread_id}', tags=['Copilot'])
    def conversation(thread_id: str, request: Request):
        if len(thread_id) != 32 or any(c not in '0123456789abcdef' for c in thread_id):
            raise HTTPException(404, '대화를 찾을 수 없습니다.')
        return service.conversation(request.state.user['uid'], thread_id)

    @app.get('/api/copilot/sources/{source_id}', tags=['Copilot'])
    def source(source_id: str):
        item = copilot.source(source_id)
        return Response(item['text'], media_type='text/plain; charset=utf-8')

    @app.post('/api/copilot/draft', tags=['Copilot'])
    def draft(body: DraftRequest, request: Request):
        thread = service.conversation(request.state.user['uid'], body.threadId)
        assistant = next((m for m in reversed(thread['messages']) if m['role'] == 'assistant'), None)
        if not assistant or assistant.get('status') != 'answered':
            raise ValueError('저장된 AI 답변이 필요합니다.')
        facts = assistant['context']
        track = facts['track']
        if track == 'quality':
            record = PreviewDecision(track=track, target=facts['testId'], decision='retest')
        elif track == 'demand' and facts.get('part') != 'ALL':
            record = PreviewDecision(track=track, target=facts['part'], date=facts['date'], model=facts['model'], decision='reviewed')
        elif track == 'maintenance' and body.target in {e['id'] for e in facts.get('events', [])}:
            record = PreviewDecision(track=track, target=body.target, decision='reviewed')
        else:
            raise ValueError('품질 시험 또는 단일 부품을 선택해 질문하세요. 예지보전은 답변에 포함된 이벤트를 먼저 선택하세요.')
        record.note = assistant['text'][:2000]
        record.source, record.conversationId = 'copilot', body.threadId
        return preview_decision(record)

    @app.post('/api/copilot/ask', tags=['Copilot'])
    def ask(body: ChatRequest, request: Request):
        uid = request.state.user['uid']
        if not body.question.strip():
            raise ValueError('질문을 입력하세요.')
        with chat_lock:
            if uid in chat_inflight or time.monotonic()-chat_last.get(uid, 0) < 3:
                raise HTTPException(429, '진행 중인 응답을 기다린 뒤 다시 질문하세요.')
            chat_inflight.add(uid)
            chat_last[uid] = time.monotonic()
        try:
            # Check storage availability before spending the user's Gemini quota.
            if not body.threadId:
                service.list_conversations(uid)
            history = service.conversation(uid, body.threadId) if body.threadId else {'messages': [], 'revision': 0}
            if history['revision'] != body.revision:
                raise HTTPException(409, '대화가 변경되었습니다. 이력을 다시 선택하세요.')
            if history['revision'] >= 20:
                raise HTTPException(422, '새 대화를 시작하세요. 대화당 최대 20회 질문할 수 있습니다.')
            result = copilot.answer(body.question.strip(), body.context.model_dump(), history['messages'])
            return service.save_conversation(uid, body.threadId or uuid4().hex, body.revision, body.question.strip(), result)
        finally:
            with chat_lock:
                chat_inflight.discard(uid)

    @app.get("/api/demand/template", tags=["추론"])
    def template():
        data = "part_number,date,actual_d,plan_d3,plan_d4,plan_d5\nPart 1,2026-09-24,100,120,130,140\nPart 1,2026-09-25,110,125,135,145\nPart 1,2026-09-26,120,130,140,150\n"
        return Response(data.encode("utf-8-sig"), media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="demand_input.csv"'})

    @app.post("/api/exports", tags=["체험"])
    def prepare_export(body: ExportRequest, request: Request):
        frame = pd.DataFrame(body.rows)
        for col in frame.select_dtypes(include=["object"]):
            frame[col] = frame[col].map(lambda x: "'"+x if isinstance(x, str) and x.lstrip().startswith(("=", "+", "-", "@")) else x)
        content = frame.to_csv(index=False).encode("utf-8-sig")
        if len(content) > 5_000_000:
            raise HTTPException(413, "필터를 좁혀 5MB 이하로 내보내세요.")
        token = secrets.token_urlsafe(24)
        with export_lock:
            for key in list(exports):
                if exports[key][0] < time.monotonic():
                    exports.pop(key)
            while len(exports) >= 20:
                exports.popitem(last=False)
            exports[token] = (time.monotonic()+900, body.name+".csv", content, request.state.user['uid'])
        return {"url": f"/api/exports/{token}", "count": len(frame)}

    @app.get("/api/exports/{token}", tags=["체험"])
    def download_export(token: str, request: Request):
        with export_lock:
            item = exports.get(token)
        if item is None or item[0] < time.monotonic() or item[3] != request.state.user['uid']:
            raise HTTPException(404, "내보내기 링크가 만료되었습니다. 화면에서 다시 내보내세요.")
        return Response(item[2], media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{item[1]}"', "Cache-Control": "no-store"})

    @app.post("/api/demand/infer", tags=["추론"])
    async def inference(file: UploadFile = File(...)):
        contents = await file.read(1_000_001)
        if len(contents) > 1_000_000:
            raise HTTPException(413, "CSV는 1MB 이하로 업로드하세요.")
        try:
            frame = pd.read_csv(io.BytesIO(contents))
        except Exception:
            raise ValueError("UTF-8 CSV 파일을 확인하세요.") from None
        required = ["part_number", "date", "actual_d", "plan_d3", "plan_d4", "plan_d5"]
        if not set(required).issubset(frame.columns) or len(frame) != 3:
            raise ValueError("템플릿의 필수 6개 열과 동일 부품의 연속 3일 자료가 필요합니다.")
        numbers = frame[required[2:]].apply(pd.to_numeric, errors="coerce")
        dates = pd.to_datetime(frame["date"], errors="coerce").sort_values()
        if (not np.isfinite(numbers.to_numpy()).all() or (numbers < 0).any().any() or dates.isna().any()
                or frame["part_number"].nunique() != 1 or frame["part_number"].isna().any()
                or not dates.diff().dropna().eq(pd.Timedelta(days=1)).all()):
            raise ValueError("부품·날짜·수량을 확인하세요. 연속 3일, 유한한 0 이상 수량이 필요합니다.")
        # One environment for the API and existing model inference; no training.
        interpreter = sys.executable
        script = ROOT / "발주량 예측 모델/src/inference.py"
        import asyncio
        def predict():
            with tempfile.TemporaryDirectory() as directory:
                input_path = __import__('pathlib').Path(directory) / "input.csv"
                input_path.write_bytes(contents)
                result = subprocess.run([str(interpreter), "-B", str(script), str(input_path)], capture_output=True,
                                        text=True, encoding="utf-8", timeout=45, env={**os.environ, "PYTHONIOENCODING": "utf-8", "MANUFACTURING_DEMAND_MODEL_DIR": str(RUNTIME / 'demand')},
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                if result.returncode:
                    raise ValueError("모델 추론에 실패했습니다. 입력 형식과 기존 모델 환경을 확인하세요.")
                return json.loads(result.stdout)
        try:
            return await asyncio.to_thread(predict)
        except subprocess.TimeoutExpired:
            raise HTTPException(504, "추론 시간이 초과되었습니다.") from None

    if mount_ui:
        from frontend.app import create_dashboard
        dash_app = create_dashboard({**repo.meta(), 'demo': demo})
        app.mount("/", WSGIMiddleware(dash_app.server))
    return app


app = create_api()
