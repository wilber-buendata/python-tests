# agent.py
from __future__ import annotations

import asyncio
import json
from typing import TypedDict, List, Any, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from langgraph.checkpoint.postgres import PostgresSaver

from google import genai
from google.genai import types

# Importamos nuestros esquemas
from schemas import (
    QuizActivity, 
    BookActivity, 
    AssignmentActivity, 
    PlanParsingResult, 
    PlanItem
)

# --- SYSTEM PROMPT DEL PLANIFICADOR (Ya existente) ---
SYSTEM_INSTRUCTION_PLAN = """You are a course planning assistant that converts syllabus content into structured learning sections and activities.

INPUT
- You will receive a course syllabus as a PDF file.
- You may also receive additional user instructions as plain text.

PRIORITY OF USER INSTRUCTIONS
- When explicit user instructions specify constraints on the NUMBER of sections or activities, you MUST strictly follow those numerical limits.
- When explicit user instructions specify constraints on the TYPE of activities, you MUST strictly follow those type constraints and MUST NOT add additional activity types beyond what the user allows.
- When explicit user instructions restrict the SCOPE of the content (for example, to certain sections or topics of the syllabus), you MUST restrict the plan to that scope and ignore non-requested parts.

NON-NEGOTIABLE OUTPUT RULES
- Output ONLY Markdown text. Absolutely no JSON, YAML, XML, HTML, or code fences.
- Do NOT include preambles, explanations, warnings, or meta commentary.
- Do NOT include any sections other than the exact headings specified below.
- Do NOT add extra bullet types, tables, or numbering styles outside what is requested.
- Use ONLY the activity types: book, quiz, assign.
- Do NOT invent facts that are not supported by the syllabus. If something essential is missing, write a minimal assumption using the exact prefix: "Assumption: ..."

REQUIRED MARKDOWN STRUCTURE (EXACT)
# COURSE OVERVIEW
- ...

# COURSE SECTIONS
## <Section Title 1>
### SECTION OVERVIEW
- ...

### SECTION ACTIVITIES
- **[book] <Activity Title>**: <What this activity will contain>
- **[quiz] <Activity Title>**: <What this activity will contain>
- **[assign] <Activity Title>**: <What this activity will contain>

## <Section Title 2>
### SECTION OVERVIEW
- ...

### SECTION ACTIVITIES
- **[book] <Activity Title>**: <What this activity will contain>
- **[quiz] <Activity Title>**: <What this activity will contain>
- **[assign] <Activity Title>**: <What this activity will contain>

CONTENT RULES
- COURSE OVERVIEW: summarize course purpose, key topics, and evaluation style only if present in the syllabus.
- Each SECTION OVERVIEW: 2–5 bullets summarizing the section’s learning intent and main topics, consistent with the syllabus.
- SECTION ACTIVITIES: create a coherent sequence. Each activity line MUST state what it will contain (content/instructions/deliverable).
- Keep activities realistic and pedagogically coherent. Avoid redundancy.
- When user instructions limit the number or type of activities, create ONLY activities that satisfy those limits and do not create any additional activities beyond them.

QUALITY BAR
- Titles must be specific and aligned with the syllabus terminology.
- The plan must be internally consistent and cover the requested syllabus scope without adding unrelated content.
"""
# --- ESTADOS DEL GRAFO ---

class PlanState(TypedDict):
    """Estado para la generación del plan (Markdown)."""
    upload_file_name: str
    user_instructions: str
    model: str

class ContentState(TypedDict):
    """Estado para la generación del contenido (JSON)."""
    upload_file_name: str
    model: str
    markdown_plan: str         # El plan completo en texto
    activities_queue: List[PlanItem] # Lista de actividades pendientes por generar
    current_activity: Optional[PlanItem] # Actividad actual siendo procesada
    generated_content: List[dict] # Resultados acumulados

# --- FUNCIONES DE AYUDA ---

def get_client():
    return genai.Client() # Asume GEMINI_API_KEY en env

# --- GRAFO 1: GENERADOR DE PLAN (Markdown) ---
# (Este código es casi idéntico al tuyo original, solo envuelto para exportar)

async def generate_plan_node(state: PlanState) -> dict:
    writer = get_stream_writer()
    client = get_client()
    
    file_obj = client.files.get(name=state["upload_file_name"])
    
    if state.get("user_instructions"):
        prompt_text = (
            "Generate the course plan. "
            "Follow the user's instructions as closely as possible, "
            "as long as they do not conflict with system or safety instructions."
        )
        prompt_text += f"\nUser instructions: {state['user_instructions']}"
    else:
        prompt_text = "Generate the course plan."

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION_PLAN,
        temperature=0.2,
        max_output_tokens=8192,
    )

    async for chunk in await client.aio.models.generate_content_stream(
        model=state.get("model", "gemini-2.0-flash"),
        contents=[file_obj, prompt_text],
        config=config,
    ):
        if chunk.text:
            writer({"type": "token", "text": chunk.text})

    writer({"type": "done"})
    return {}

def create_plan_graph(checkpointer: PostgresSaver):
    g = StateGraph(PlanState)
    g.add_node("generate_plan", lambda s: asyncio.run(generate_plan_node(s)))
    g.add_edge(START, "generate_plan")
    g.add_edge("generate_plan", END)
    return g.compile(checkpointer=checkpointer)


# --- GRAFO 2: GENERADOR DE CONTENIDO (JSON Iterativo) ---

async def parse_plan_node(state: ContentState) -> dict:
    """Lee el plan Markdown y extrae una lista estructurada de actividades."""
    writer = get_stream_writer()
    writer({"type": "status", "text": "Analyzing plan structure..."})
    
    client = get_client()
    
    prompt = f"""
    Analyze the following Course Plan Markdown and extract all activities listed under 'SECTION ACTIVITIES'.
    Return a strict JSON list of objects matching the schema.
    
    MARKDOWN PLAN:
    {state['markdown_plan']}
    """
    
    # Usamos response_json_schema con GenAI SDK
    response = await client.aio.models.generate_content(
        model=state.get("model", "gemini-2.0-flash"),
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": PlanParsingResult.model_json_schema(),
        },
    )
    
    # Parseamos la respuesta con Pydantic
    parsed_result = PlanParsingResult.model_validate_json(response.text)
    
    writer({"type": "status", "text": f"Found {len(parsed_result.activities)} activities to generate."})
    
    return {"activities_queue": parsed_result.activities, "generated_content": []}

def router_node(state: ContentState) -> str:
    """Decide qué nodo ejecutar basándose en la siguiente actividad."""
    queue = state.get("activities_queue", [])
    if not queue:
        return "end"
    
    next_activity = queue[0]
    return f"generate_{next_activity.activity_type}" # generate_quiz, generate_book, generate_assign

async def generate_quiz_node(state: ContentState) -> dict:
    return await _generate_generic_activity(state, QuizActivity, "quiz")

async def generate_book_node(state: ContentState) -> dict:
    return await _generate_generic_activity(state, BookActivity, "book")

async def generate_assign_node(state: ContentState) -> dict:
    return await _generate_generic_activity(state, AssignmentActivity, "assignment")

async def _generate_generic_activity(state: ContentState, schema_class, type_label: str) -> dict:
    """Función genérica para llamar a Gemini con un esquema específico."""
    writer = get_stream_writer()
    client = get_client()
    
    # Sacamos la actividad actual de la cola (sin eliminarla aun del estado global hasta retornar)
    current_activity: PlanItem = state["activities_queue"][0]
    remaining_queue = state["activities_queue"][1:]
    
    # Emitimos evento de progreso
    writer({"type": "status", "text": f"Generating {type_label}: {current_activity.title}..."})
    
    # Obtenemos referencia al archivo PDF para contexto
    file_obj = client.files.get(name=state["upload_file_name"])
    
    prompt = f"""
    Create the detailed content JSON for the activity described below.
    Use the provided Syllabus PDF as the ground truth for the content content.
    
    ACTIVITY DETAILS:
    - Type: {current_activity.activity_type}
    - Title: {current_activity.title}
    - Section: {current_activity.section}
    - Intent: {current_activity.description_intent}
    
    Requirement: Generate fully valid HTML for text fields inside the JSON.
    """
    
    # Llamada a Gemini forzando el esquema JSON de Pydantic
    response = await client.aio.models.generate_content(
        model=state.get("model", "gemini-2.0-flash"),
        contents=[file_obj, prompt],
        config={
            "response_mime_type": "application/json",
            "response_json_schema": schema_class.model_json_schema(),
        },
    )
    
    # Validamos y convertimos a dict
    generated_data = schema_class.model_validate_json(response.text).model_dump()
    
    # Añadimos metadatos para identificarlo luego
    final_record = {
        "activity_id": f"{type_label}_{len(state['generated_content'])}",
        "type": type_label,
        "plan_context": current_activity.model_dump(),
        "data": generated_data
    }
    
    writer({"type": "status", "text": f"Completed {current_activity.title}"})
    
    # Actualizamos estado: añadimos resultado, quitamos de cola
    return {
        "generated_content": [final_record], # LangGraph hace append/update según reducer, por defecto overwrite en simple dict, pero aquí queremos acumular.
        # NOTA: En LangGraph simple dict state, se sobreescribe. Para listas necesitamos un reducer o manejar la lista completa.
        # Aquí reescribiremos la lista completa + el nuevo.
        "generated_content": state["generated_content"] + [final_record],
        "activities_queue": remaining_queue
    }

# Sync wrapper para LangGraph
def run_async_node(node_func, state):
    return asyncio.run(node_func(state))

def create_content_graph(checkpointer: PostgresSaver):
    g = StateGraph(ContentState)
    
    g.add_node("parse_plan", lambda s: run_async_node(parse_plan_node, s))
    g.add_node("generate_quiz", lambda s: run_async_node(generate_quiz_node, s))
    g.add_node("generate_book", lambda s: run_async_node(generate_book_node, s))
    g.add_node("generate_assign", lambda s: run_async_node(generate_assign_node, s))
    
    g.add_edge(START, "parse_plan")
    
    # Router condicional
    g.add_conditional_edges(
        "parse_plan",
        router_node,
        {
            "generate_quiz": "generate_quiz",
            "generate_book": "generate_book",
            "generate_assign": "generate_assign",
            "end": END
        }
    )
    
    # Después de generar cualquiera, volvemos al router para ver si queda algo en la cola
    g.add_conditional_edges(
        "generate_quiz", router_node, 
        {"generate_quiz": "generate_quiz", "generate_book": "generate_book", "generate_assign": "generate_assign", "end": END}
    )
    g.add_conditional_edges(
        "generate_book", router_node,
        {"generate_quiz": "generate_quiz", "generate_book": "generate_book", "generate_assign": "generate_assign", "end": END}
    )
    g.add_conditional_edges(
        "generate_assign", router_node,
        {"generate_quiz": "generate_quiz", "generate_book": "generate_book", "generate_assign": "generate_assign", "end": END}
    )

    return g.compile(checkpointer=checkpointer)