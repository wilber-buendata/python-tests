from __future__ import annotations

import os
import re
import time
import queue
import threading
from dataclasses import dataclass
from typing import Annotated, Dict, List, Optional, Tuple
from operator import add
from typing_extensions import TypedDict

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from google import genai
from google.genai import types

load_dotenv()

# -----------------------------
# Config
# -----------------------------
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
POSTGRES_URL = os.getenv("POSTGRES_URL")

STEP_DELAY_SECONDS = float(os.getenv("STEP_DELAY_SECONDS", "0.05"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "2048"))

# Para asegurar un stream "largo" en demo (evita que termine en 1 frase)
MIN_CHARS = int(os.getenv("MIN_CHARS", "2200"))
MAX_CONTINUATIONS = int(os.getenv("MAX_CONTINUATIONS", "3"))

SENTINEL = object()


# -----------------------------
# 1) State (persistido)
# -----------------------------
class ChatState(TypedDict):
    thread_id: str
    prompt: str
    tokens: Annotated[List[str], add]
    done: bool


# -----------------------------
# 2) Utilidades: tokenización + delta
# -----------------------------
TOKEN_RE = re.compile(r"\S+|\s+")


def split_preserve_whitespace(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


def compute_delta(chunk_text: str, accumulated: str) -> Tuple[str, str]:
    # Normaliza streaming que pueda venir como "texto acumulado" o "delta"
    if chunk_text.startswith(accumulated):
        delta = chunk_text[len(accumulated):]
        accumulated = chunk_text
        return delta, accumulated

    accumulated = accumulated + chunk_text
    return chunk_text, accumulated


def safe_chunk_text(chunk) -> str:
    text = getattr(chunk, "text", "") or ""
    return text


# -----------------------------
# 3) GenAI client + Chat factory
# -----------------------------
def build_genai_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)
    return genai.Client()


def build_chat(client: genai.Client) -> object:
    cfg = types.GenerateContentConfig(
        temperature=0.7,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        system_instruction=(
            "Responde en español, con secciones y ejemplos. "
            f"Extiende la respuesta hasta superar ~{MIN_CHARS} caracteres. "
            "No termines prematuramente."
        ),
    )
    return client.chats.create(model=MODEL, config=cfg)


# -----------------------------
# 4) Producer: Gemini -> queue (micro-tokens)
# -----------------------------
@dataclass
class ProducerState:
    thread_id: str
    prompt: str
    q: "queue.Queue[object]"
    accumulated: str = ""
    full_text: str = ""


def enqueue_micro_tokens(q: "queue.Queue[object]", delta: str) -> None:
    for tok in split_preserve_whitespace(delta):
        q.put(tok)


def send_message_and_enqueue(chat, st: ProducerState, message: str) -> None:
    for chunk in chat.send_message_stream(message):
        chunk_text = safe_chunk_text(chunk)
        delta, st.accumulated = compute_delta(chunk_text, st.accumulated)
        st.full_text = st.full_text + delta
        enqueue_micro_tokens(st.q, delta)


def producer_main(st: ProducerState) -> None:
    try:
        client = build_genai_client()
        chat = build_chat(client)

        send_message_and_enqueue(chat, st, st.prompt)

        for _ in range(MAX_CONTINUATIONS):
            if len(st.full_text) >= MIN_CHARS:
                break
            send_message_and_enqueue(
                chat,
                st,
                "Continúa EXACTAMENTE donde quedaste. No repitas nada. "
                "Agrega más detalle técnico y ejemplos."
            )

    except Exception as e:
        st.q.put(f"\n\n[server] ❌ error: {type(e).__name__}: {e}\n")

    st.q.put(SENTINEL)


# -----------------------------
# 5) Registry: queue + producer thread por thread_id
# -----------------------------
@dataclass
class StreamHandle:
    q: "queue.Queue[object]"
    t: threading.Thread


class StreamRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._streams: Dict[str, StreamHandle] = {}

    def get_or_start(self, thread_id: str, prompt: str) -> "queue.Queue[object]":
        with self._lock:
            handle = self._streams.get(thread_id)

            if handle is None:
                q: "queue.Queue[object]" = queue.Queue()
                st = ProducerState(thread_id=thread_id, prompt=prompt, q=q)
                t = threading.Thread(target=producer_main, args=(st,), daemon=True)
                self._streams[thread_id] = StreamHandle(q=q, t=t)
                t.start()
                return q

            if not handle.t.is_alive():
                # Si el thread murió (por error), reinicia para demo
                q2: "queue.Queue[object]" = queue.Queue()
                st2 = ProducerState(thread_id=thread_id, prompt=prompt, q=q2)
                t2 = threading.Thread(target=producer_main, args=(st2,), daemon=True)
                self._streams[thread_id] = StreamHandle(q=q2, t=t2)
                t2.start()
                return q2

            return handle.q

    def cleanup(self, thread_id: str) -> None:
        with self._lock:
            self._streams.pop(thread_id, None)


REGISTRY = StreamRegistry()


# -----------------------------
# 6) LangGraph node: 1 micro-token por step
# -----------------------------
def generate_one_token(state: ChatState) -> dict:
    if state.get("done", False):
        return {"done": True}

    thread_id = state["thread_id"]
    prompt = state["prompt"]

    q = REGISTRY.get_or_start(thread_id, prompt)
    item = q.get()

    if item is SENTINEL:
        REGISTRY.cleanup(thread_id)
        return {"done": True}

    if STEP_DELAY_SECONDS > 0:
        time.sleep(STEP_DELAY_SECONDS)

    return {"tokens": [str(item)], "done": False}


def should_continue(state: ChatState) -> str:
    if state.get("done", False):
        return "finish"
    return "loop"


def build_graph(checkpointer: PostgresSaver):
    g = StateGraph(ChatState)
    g.add_node("gen", generate_one_token)
    g.add_edge(START, "gen")
    g.add_conditional_edges("gen", should_continue, {"loop": "gen", "finish": END})
    return g.compile(checkpointer=checkpointer)


# -----------------------------
# 7) Server runner (graph invoke)
# -----------------------------
def run_server_job(graph, thread_id: str, prompt: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    graph.invoke({"thread_id": thread_id, "prompt": prompt, "tokens": [], "done": False}, config)


# -----------------------------
# 8) Client: polling, disconnect/reconnect
# -----------------------------
def poll_stream_from_postgres(
    graph,
    thread_id: str,
    seen: int,
    stop_after_seconds: Optional[float],
    stop_after_new_tokens: Optional[int],
) -> int:
    config = {"configurable": {"thread_id": thread_id}}
    start = time.time()
    printed = 0

    while True:
        snapshot = graph.get_state(config)
        values = snapshot.values or {}
        tokens = values.get("tokens", []) or []
        done = bool(values.get("done", False))

        if len(tokens) > seen:
            delta = tokens[seen:]
            print("".join(delta), end="", flush=True)
            seen = len(tokens)
            printed = printed + len(delta)

        if done:
            print("\n[client] ✅ done (según estado persistido)")
            return seen

        if stop_after_new_tokens is not None and printed >= stop_after_new_tokens:
            print("\n[client] 🔌 disconnected (simulado por tokens)")
            return seen

        if stop_after_seconds is not None and (time.time() - start) >= stop_after_seconds:
            print("\n[client] 🔌 disconnected (simulado por tiempo)")
            return seen

        time.sleep(0.20)


def read_full_progress_once(graph, thread_id: str) -> int:
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    values = snapshot.values or {}
    tokens = values.get("tokens", []) or []
    done = bool(values.get("done", False))

    print("[client] Reconnected. Último texto persistido hasta ahora:\n")
    print("".join(tokens))
    print(f"\n[client] done={done}\n")
    return len(tokens)


# -----------------------------
# 9) Main
# -----------------------------
def main() -> None:
    if not POSTGRES_URL:
        raise SystemExit("Missing POSTGRES_URL env var.")

    thread_id = os.getenv("THREAD_ID", f"demo-thread-{int(time.time())}")

    prompt = (
        "Explica en detalle cómo diseñar streaming + reconexión en chatbots con LLMs. "
        "Incluye arquitectura, persistencia, idempotencia, manejo de fallos, y ejemplos."
    )

    with PostgresSaver.from_conn_string(POSTGRES_URL) as checkpointer:
        checkpointer.setup()
        graph = build_graph(checkpointer)

        server = threading.Thread(target=run_server_job, args=(graph, thread_id, prompt), daemon=True)
        server.start()

        seen = 0

        print("[client] connecting and streaming (1)...\n")
        seen = poll_stream_from_postgres(graph, thread_id, seen, None, 180)

        print("[client] (offline 1) ...server keeps generating...\n")
        time.sleep(2.5)

        print("[client] reconnecting (1)...\n")
        seen = read_full_progress_once(graph, thread_id)

        print("[client] continuing stream (2)...\n")
        seen = poll_stream_from_postgres(graph, thread_id, seen, None, 180)

        print("[client] (offline 2) ...server keeps generating...\n")
        time.sleep(2.5)

        print("[client] reconnecting (2)...\n")
        seen = read_full_progress_once(graph, thread_id)

        print("[client] continuing stream until done (3)...\n")
        seen = poll_stream_from_postgres(graph, thread_id, seen, None, None)

        server.join(timeout=15.0)
        print("[main] finished")


if __name__ == "__main__":
    main()
