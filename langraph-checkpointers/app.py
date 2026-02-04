# main.py
from __future__ import annotations

import os
import uuid
import aiofiles
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from google import genai
from langgraph.checkpoint.postgres import PostgresSaver
from agent import create_graph

from dotenv import load_dotenv

load_dotenv()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Very simple in-memory registry (use Redis/DB in production)
UPLOAD_REGISTRY: dict[str, str] = {}  # upload_id -> file_name (Gemini Files API name)


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        return JSONResponse({"error": "Only PDF files are allowed"}, status_code=400)

    upload_id = str(uuid.uuid4())
    tmp_path = f"/tmp/{upload_id}.pdf"

    async with aiofiles.open(tmp_path, "wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            await out.write(chunk)

    client = genai.Client()
    # Upload to Gemini Files API
    uploaded = client.files.upload(file=tmp_path)  # :contentReference[oaicite:11]{index=11}

    # Store the returned file "name" for later retrieval
    UPLOAD_REGISTRY[upload_id] = uploaded.name

    # Optional: cleanup tmp file
    try:
        os.remove(tmp_path)
    except OSError:
        pass

    return {"upload_id": upload_id}


@app.get("/stream")
async def stream_plan(
    upload_id: str,
    user_instructions: str = "",
):
    if upload_id not in UPLOAD_REGISTRY:
        return JSONResponse({"error": "Invalid upload_id"}, status_code=404)

    file_name = UPLOAD_REGISTRY[upload_id]

    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        return JSONResponse({"error": "POSTGRES_URL not configured"}, status_code=500)

    def event_generator():
        # Build graph with Postgres checkpointer and stream with per-upload thread_id
        with PostgresSaver.from_conn_string(postgres_url) as checkpointer:
            checkpointer.setup()
            graph = create_graph(checkpointer)

            config = {"configurable": {"thread_id": upload_id}}

            for mode, chunk in graph.stream(
                {
                    "upload_file_name": file_name,
                    "user_instructions": user_instructions,
                    "model": os.getenv("GEMINI_MODEL"),
                },
                config=config,
                stream_mode=["custom"],
            ):
                if mode != "custom":
                    continue

                # chunk is what we wrote via writer({...})
                if chunk.get("type") == "token":
                    yield {"event": "token", "data": chunk["text"]}
                elif chunk.get("type") == "done":
                    yield {"event": "done", "data": ""}

    return EventSourceResponse(event_generator())


# uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1