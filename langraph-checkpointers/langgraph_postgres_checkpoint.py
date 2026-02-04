"""
LangGraph + Postgres checkpointer demo (single-file script)

What this simulates:
- A "server" thread is generating a chatbot response token-by-token (mocked).
- The "client" starts streaming those tokens by polling the persisted state in Postgres.
- Mid-response, the client "disconnects" (stops polling).
- The server continues and may finish.
- When the client "reconnects", it reads the latest checkpoint from Postgres and:
    - prints everything generated so far (even if the server already finished), and/or
    - keeps streaming the remaining tokens until done.

Requirements (install):
    pip install langgraph langgraph-checkpoint-postgres "psycopg[binary]"

Run:
    export POSTGRES_URL='postgresql://user:password@localhost:5432/postgres'
    python demo_langgraph_pg_checkpoint_resume.py

Notes:
- This is pure mock data; no LLM calls.
- The "resume" behavior here is *client reconnection* to persisted progress.
  (If your server process crashes, you’d typically restart execution from the last checkpoint —
   that’s a different failure mode.)
"""

from __future__ import annotations

import os
import time
import threading
from typing import Annotated, List
from typing_extensions import TypedDict
from operator import add

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# 1) Graph state (persisted)
# -----------------------------
class ChatState(TypedDict):
    prompt: str
    # We append tokens over time; reducer "add" lets LangGraph merge lists across steps.
    tokens: Annotated[List[str], add]
    done: bool


BASE_DIR = os.path.dirname(__file__)
MOCK_FINAL_TEXT_PATH = os.path.join(BASE_DIR, "mock_long_text_es.txt")

with open(MOCK_FINAL_TEXT_PATH, "r", encoding="utf-8") as f:
    MOCK_FINAL_TEXT = f.read()


def _tokenize(s: str) -> List[str]:
    # simple word-ish tokenization; keep spaces to make printing nicer
    out: List[str] = []
    for w in s.split(" "):
        out.append(w + " ")
    return out


TARGET_TOKENS = _tokenize(MOCK_FINAL_TEXT)


# -----------------------------
# 2) Nodes
# -----------------------------
def generate_one_token(state: ChatState) -> dict:
    """
    Produces exactly ONE token per graph step and persists it via checkpointer.
    This gives you durable, incremental progress.
    """
    # slow down so we can "disconnect" mid-response varias veces
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


# -----------------------------
# 3) Build the looping graph
# -----------------------------
def build_graph(checkpointer: PostgresSaver):
    g = StateGraph(ChatState)
    g.add_node("gen", generate_one_token)

    g.add_edge(START, "gen")
    g.add_conditional_edges(
        "gen",
        should_continue,
        {
            "loop": "gen",
            "finish": END,
        },
    )
    return g.compile(checkpointer=checkpointer)


# -----------------------------
# 4) "Server" worker
# -----------------------------
def run_server_job(graph, thread_id: str, prompt: str):
    """
    Runs the graph to completion, writing checkpoints in Postgres each step.
    """
    config = {"configurable": {"thread_id": thread_id}}

    # First user message initializes the thread state
    graph.invoke(
        {"prompt": prompt, "tokens": [], "done": False},
        config,
    )

    # Now it will keep looping (token by token) until done.
    # Because the graph is compiled with a Postgres checkpointer,
    # each step is persisted.
    graph.invoke({}, config)


# -----------------------------
# 5) "Client" that can disconnect/reconnect
# -----------------------------
def poll_stream_from_postgres(graph, thread_id: str, *, stop_after_seconds: float | None = None):
    """
    Polls persisted state from Postgres and prints only newly-seen tokens.
    If stop_after_seconds is set, it simulates a disconnect (stops polling).
    """
    config = {"configurable": {"thread_id": thread_id}}
    seen = 0
    start = time.time()

    while True:
        snapshot = graph.get_state(config)
        values = snapshot.values or {}
        tokens = values.get("tokens", []) or []
        done = bool(values.get("done", False))

        # Print newly generated tokens
        if len(tokens) > seen:
            delta = tokens[seen:]
            print("".join(delta), end="", flush=True)
            seen = len(tokens)

        if done:
            print("\n[client] ✅ done (as per persisted state)")
            return

        if stop_after_seconds is not None and (time.time() - start) >= stop_after_seconds:
            print("\n[client] 🔌 disconnected (simulated)")
            return

        time.sleep(0.25)


def read_full_progress_once(graph, thread_id: str):
    """
    On reconnect, fetch the latest state and show what is already available,
    even if the server finished while we were away.
    """
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    values = snapshot.values or {}
    tokens = values.get("tokens", []) or []
    done = bool(values.get("done", False))

    print("[client] Reconnected. Latest persisted text so far:")
    print("".join(tokens))
    print(f"[client] done={done}\n")


# -----------------------------
# 6) Main
# -----------------------------
def main():
    postgres_url = "postgresql://postgres:postgres@localhost:5433/postgres"
    if not postgres_url:
        raise SystemExit(
            "Missing POSTGRES_URL env var.\n"
            "Example:\n"
            "  export POSTGRES_URL='postgresql://user:password@localhost:5432/postgres'\n"
        )

    thread_id = "demo-thread-123"
    prompt = "Hola, simula una respuesta larga y persistente."

    # IMPORTANT: PostgresSaver.from_conn_string is a contextmanager
    # (so connections are cleaned up properly).
    with PostgresSaver.from_conn_string(postgres_url) as checkpointer:
        # Creates/checks required tables
        checkpointer.setup()

        graph = build_graph(checkpointer)

        # Start "server" generation in background
        server = threading.Thread(
            target=run_server_job, args=(graph, thread_id, prompt), daemon=True
        )
        server.start()

        # Primera conexión: el cliente se desconecta a los pocos segundos
        print("[client] connecting and streaming (1)...\n")
        poll_stream_from_postgres(graph, thread_id, stop_after_seconds=3.0)

        print("[client] (offline 1) ...server keeps generating...\n")
        time.sleep(3.0)

        print("[client] reconnecting (1)...\n")
        read_full_progress_once(graph, thread_id)

        # Segunda conexión: vuelve a desconectarse
        print("[client] continuing stream (2)...\n")
        poll_stream_from_postgres(graph, thread_id, stop_after_seconds=3.0)

        print("[client] (offline 2) ...server keeps generating...\n")
        time.sleep(3.0)

        print("[client] reconnecting (2)...\n")
        read_full_progress_once(graph, thread_id)

        # Tercera conexión: ahora sí, hasta el final
        print("[client] continuing stream until done (3)...\n")
        poll_stream_from_postgres(graph, thread_id, stop_after_seconds=None)

        # Join server (should already be done)
        server.join(timeout=1.0)
        print("[main] finished")

if __name__ == "__main__":
    main()
