from __future__ import annotations

import os
import time
import uuid
import threading
import asyncio
from typing import Annotated, List, Optional

from operator import add
from typing_extensions import TypedDict

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Config
# -----------------------------
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://postgres:postgres@localhost:5433/postgres")

BASE_DIR = os.path.dirname(__file__)
MOCK_FINAL_TEXT_PATH = os.path.join(BASE_DIR, "mock_long_text_es.txt")

if not os.path.exists(MOCK_FINAL_TEXT_PATH):
    raise RuntimeError(f"Missing mock file: {MOCK_FINAL_TEXT_PATH}")

with open(MOCK_FINAL_TEXT_PATH, "r", encoding="utf-8") as f:
    MOCK_FINAL_TEXT = f.read()


def _tokenize(s: str) -> List[str]:
    # tokenización simple estilo demo: palabras + espacio
    out: List[str] = []
    for w in s.split(" "):
        out.append(w + " ")
    return out


TARGET_TOKENS = _tokenize(MOCK_FINAL_TEXT)

# -----------------------------
# LangGraph state (persisted)
# -----------------------------
class ChatState(TypedDict):
    prompt: str
    tokens: Annotated[List[str], add]
    done: bool


# -----------------------------
# Nodes
# -----------------------------
def generate_one_token(state: ChatState) -> dict:
    # simulación lenta para que se note el "disconnect/reconnect"
    time.sleep(0.6)

    already = len(state.get("tokens", []))
    if already >= len(TARGET_TOKENS):
        return {"done": True}

    next_tok = TARGET_TOKENS[already]
    return {
        "tokens": [next_tok],
        "done": (already + 1) >= len(TARGET_TOKENS),
    }


def should_continue(state: ChatState) -> str:
    return "loop" if not state.get("done", False) else "finish"


def build_graph(checkpointer: PostgresSaver):
    g = StateGraph(ChatState)
    g.add_node("gen", generate_one_token)
    g.add_edge(START, "gen")
    g.add_conditional_edges(
        "gen",
        should_continue,
        {"loop": "gen", "finish": END},
    )
    return g.compile(checkpointer=checkpointer)


# -----------------------------
# "Server" background generation
# -----------------------------
def run_generation_job(thread_id: str, prompt: str):
    """
    Corre en background y escribe checkpoints en Postgres.
    Importante: abre su propia conexión/contexto para que sea independiente del request.
    """
    config = {"configurable": {"thread_id": thread_id}}

    with PostgresSaver.from_conn_string(POSTGRES_URL) as checkpointer:
        # Asegura tablas (idempotente)
        checkpointer.setup()

        graph = build_graph(checkpointer)

        # Si ya existe estado, NO lo re-inicializamos (evita pisar tokens)
        snapshot = graph.get_state(config)
        values = snapshot.values or {}
        if values:
            # Si ya estaba done, no hacemos nada
            if bool(values.get("done", False)):
                return
            # Si no estaba done, continuamos desde checkpoint
            graph.invoke({}, config)
            return

        # Nuevo thread: inicializa y deja que el grafo corra hasta END
        graph.invoke({"prompt": prompt, "tokens": [], "done": False}, config)


# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="LangGraph + Postgres checkpoint streaming demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ajusta en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    # crea tablas una sola vez al arrancar
    with PostgresSaver.from_conn_string(POSTGRES_URL) as checkpointer:
        checkpointer.setup()


@app.post("/threads")
def create_thread(payload: dict):
    """
    Crea un thread y arranca la generación en background.
    Body: {"prompt": "..."}
    """
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    thread_id = payload.get("thread_id") or f"thread-{uuid.uuid4().hex}"

    # Arranca background "server" (demo: thread dentro del proceso)
    t = threading.Thread(target=run_generation_job, args=(thread_id, prompt), daemon=True)
    t.start()

    return {"thread_id": thread_id}


@app.get("/threads/{thread_id}")
def get_thread_state(thread_id: str):
    """
    Endpoint auxiliar: devuelve el texto completo y si terminó.
    """
    config = {"configurable": {"thread_id": thread_id}}
    with PostgresSaver.from_conn_string(POSTGRES_URL) as checkpointer:
        graph = build_graph(checkpointer)
        snapshot = graph.get_state(config)
        values = snapshot.values or {}
        if not values:
            raise HTTPException(status_code=404, detail="thread not found")
        tokens = values.get("tokens", []) or []
        done = bool(values.get("done", False))
        return {"thread_id": thread_id, "text": "".join(tokens), "done": done, "tokens_count": len(tokens)}


@app.get("/threads/{thread_id}/stream")
async def stream_thread(thread_id: str, request: Request, from_index: Optional[int] = None):
    """
    SSE streaming:
    - usa persistencia en Postgres
    - soporta reconexión: EventSource envía header Last-Event-ID automáticamente
      si nosotros mandamos 'id:' en cada evento.
    """
    # Prioridad: Last-Event-ID (SSE) > from_index query param > 0
    last_event_id = request.headers.get("last-event-id")
    try:
        if last_event_id is not None:
            start_index = int(last_event_id) + 1
        elif from_index is not None:
            start_index = max(0, int(from_index))
        else:
            start_index = 0
    except ValueError:
        start_index = 0

    async def event_gen():
        config = {"configurable": {"thread_id": thread_id}}

        # Mantén este contexto abierto durante el streaming
        with PostgresSaver.from_conn_string(POSTGRES_URL) as checkpointer:
            graph = build_graph(checkpointer)

            seen = start_index

            while True:
                if await request.is_disconnected():
                    # el cliente se fue; cortamos el stream
                    return

                # get_state es sync -> lo mandamos a threadpool
                snapshot = await asyncio.to_thread(graph.get_state, config)
                values = snapshot.values or {}

                # Si aún no existe, hacemos keepalive y esperamos
                if not values:
                    yield ": waiting-for-thread\n\n"
                    await asyncio.sleep(0.5)
                    continue

                tokens = values.get("tokens", []) or []
                done = bool(values.get("done", False))

                # enviar delta desde "seen"
                if len(tokens) > seen:
                    delta = tokens[seen:]
                    for i, tok in enumerate(delta, start=seen):
                        # SSE: id + data
                        # id permite que EventSource reintente y mande Last-Event-ID
                        yield f"id: {i}\n"
                        # data NO debe tener saltos de línea sin prefijo "data:"
                        safe = tok.replace("\n", "\\n")
                        yield f"data: {safe}\n\n"
                    seen = len(tokens)

                if done:
                    # evento final
                    yield "event: done\n"
                    yield "data: true\n\n"
                    return

                # keepalive/poll
                yield ": keep-alive\n\n"
                await asyncio.sleep(0.25)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        # si estás detrás de nginx, esto ayuda:
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=headers)


# Ejecutar:
# uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
