"""FastAPI application for the Energytool consultant prototype."""

from __future__ import annotations

import threading
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from integration.interface import EnergytoolAI
from memory.conversation_state import ConversationState


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "web" / "static"

app = FastAPI(
    title="Energytool AI Consultant",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url=None,
)

_ai: EnergytoolAI | None = None
_ai_lock = threading.Lock()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    debug: bool = False


def get_ai() -> EnergytoolAI:
    """Build the retrieval index only on the first chat request."""
    global _ai
    if _ai is None:
        with _ai_lock:
            if _ai is None:
                _ai = EnergytoolAI()
    return _ai


def validate_session_id(session_id: str) -> str:
    try:
        return str(UUID(session_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Некорректный идентификатор сессии.") from exc


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "energytool-ai"}


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict:
    try:
        result = await run_in_threadpool(get_ai().chat, request.message, request.session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Не удалось обработать запрос. Проверьте настройки модели и повторите попытку.",
        ) from exc
    if not request.debug:
        result["debug"] = {"elapsed_ms": result.get("debug", {}).get("elapsed_ms")}
    return result


@app.post("/api/sessions")
def create_session() -> dict:
    state = ConversationState()
    state.save()
    return {"session_id": state.session_id, "created_at": state.data["created_at"]}


@app.get("/api/sessions")
def list_sessions(limit: int = Query(default=100, ge=1, le=500)) -> dict:
    return {"sessions": ConversationState.list_sessions(limit)}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    session_id = validate_session_id(session_id)
    state = ConversationState(session_id)
    if not state.path.exists():
        raise HTTPException(status_code=404, detail="Сессия не найдена.")
    return state.load().data


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
