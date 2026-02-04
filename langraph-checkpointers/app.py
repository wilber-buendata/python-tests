# app.py
from __future__ import annotations

import os
import uuid
import json
import aiofiles
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from google import genai
from langgraph.checkpoint.postgres import PostgresSaver

# Importamos ambas funciones de creación de grafos
from agent import create_plan_graph, create_content_graph

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

# In-memory storage simulado (usar Redis/DB en prod)
# upload_id -> gemini_file_name
UPLOAD_REGISTRY: dict[str, str] = {}
# upload_id -> markdown_plan_text (Guardamos el plan generado para usarlo luego)
PLAN_STORAGE: dict[str, str] = {} 
# upload_id -> final_json_content (Guardamos el JSON final generado)
CONTENT_STORAGE: dict[str, list] = {}

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
    uploaded = client.files.upload(file=tmp_path)
    
    UPLOAD_REGISTRY[upload_id] = uploaded.name
    
    try:
        os.remove(tmp_path)
    except OSError:
        pass

    return {"upload_id": upload_id}

# --- ENDPOINT 1: PLANIFICACIÓN (Genera Markdown) ---
@app.get("/stream-plan")
async def stream_plan(upload_id: str, user_instructions: str = ""):
    if upload_id not in UPLOAD_REGISTRY:
        return JSONResponse({"error": "Invalid upload_id"}, status_code=404)

    file_name = UPLOAD_REGISTRY[upload_id]
    postgres_url = os.getenv("POSTGRES_URL")

    def event_generator():
        full_markdown = ""
        with PostgresSaver.from_conn_string(postgres_url) as checkpointer:
            checkpointer.setup()
            graph = create_plan_graph(checkpointer)
            config = {"configurable": {"thread_id": f"plan_{upload_id}"}}

            for mode, chunk in graph.stream(
                {
                    "upload_file_name": file_name,
                    "user_instructions": user_instructions,
                    "model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
                },
                config=config,
                stream_mode=["custom"],
            ):
                if mode == "custom":
                    if chunk.get("type") == "token":
                        text = chunk["text"]
                        full_markdown += text
                        yield {"event": "token", "data": text}
                    elif chunk.get("type") == "done":
                        yield {"event": "done", "data": ""}
        
        # Guardamos el plan generado en memoria para el siguiente paso
        PLAN_STORAGE[upload_id] = full_markdown

    return EventSourceResponse(event_generator())

# --- ENDPOINT 2: GENERACIÓN DE CONTENIDO (Genera JSON Complejo) ---
@app.get("/stream-content-generation")
async def stream_content(upload_id: str):
    """
    Toma el plan ya generado para ese upload_id y empieza a construir
    los JSONs de Moodle iterativamente.
    """
    if upload_id not in UPLOAD_REGISTRY:
        return JSONResponse({"error": "Invalid upload_id"}, status_code=404)
    
    if upload_id not in PLAN_STORAGE:
        return JSONResponse({"error": "Plan not found. Generate plan first."}, status_code=400)

    file_name = UPLOAD_REGISTRY[upload_id]
    markdown_plan = PLAN_STORAGE[upload_id]
    postgres_url = os.getenv("POSTGRES_URL")

    def event_generator():
        with PostgresSaver.from_conn_string(postgres_url) as checkpointer:
            checkpointer.setup()
            graph = create_content_graph(checkpointer)
            # Usamos un thread_id diferente para no mezclar con el plan
            config = {"configurable": {"thread_id": f"content_{upload_id}"}}

            # Ejecutamos el grafo
            # Nota: El parseo inicial y cada actividad emitirán eventos
            final_state = None
            
            for mode, chunk in graph.stream(
                {
                    "upload_file_name": file_name,
                    "markdown_plan": markdown_plan,
                    "model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
                    "activities_queue": [], # Se llenará en el primer nodo
                    "generated_content": []
                },
                config=config,
                stream_mode=["custom", "values"], # 'values' para obtener el estado final
            ):
                # Capturar eventos custom (status updates)
                if mode == "custom":
                    if chunk.get("type") == "status":
                        yield {"event": "status", "data": chunk["text"]}
                
                # Capturar cambios de estado para guardar el resultado final
                if mode == "values":
                    final_state = chunk

            # Al finalizar el stream, guardamos el resultado
            if final_state and "generated_content" in final_state:
                CONTENT_STORAGE[upload_id] = final_state["generated_content"]
                yield {"event": "complete", "data": json.dumps({"count": len(final_state["generated_content"])})}

    return EventSourceResponse(event_generator())

# --- ENDPOINT 3: OBTENER RESULTADO FINAL ---
@app.get("/get-content/{upload_id}")
async def get_content(upload_id: str):
    if upload_id not in CONTENT_STORAGE:
        return JSONResponse({"error": "Content not generated yet"}, status_code=404)
    return JSONResponse(CONTENT_STORAGE[upload_id])