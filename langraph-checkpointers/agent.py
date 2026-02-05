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
    PlanItem,
    # Nuevos imports
    QuizBlueprint,
    TrueFalseQuestion,
    MultichoiceQuestion,
    EssayQuestion
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

class UsageMetrics(TypedDict):
    """Métrica individual de una llamada a la LLM."""
    step: str             # Nombre del paso (ej: "Quiz Blueprint", "Q1 TrueFalse")
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model_name: str

class ContentState(TypedDict):
    """Estado actualizado con log de uso."""
    upload_file_name: str
    model: str
    markdown_plan: str
    activities_queue: List[PlanItem]
    # Nuevo campo para acumular métricas
    usage_log: List[UsageMetrics] 
    generated_content: List[dict]

# --- HELPER PARA EXTRAER TOKENS ---
def log_token_usage(response: Any, step_name: str, model: str) -> UsageMetrics:
    """Extrae la metadata de uso de la respuesta de Gemini."""
    usage = response.usage_metadata
    return {
        "step": step_name,
        "input_tokens": usage.prompt_token_count,
        "output_tokens": usage.candidates_token_count,
        "total_tokens": usage.total_token_count,
        "model_name": model
    }
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
        model=state.get("model", "gemini-2.5-flash"),
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

    current_log = state.get("usage_log", [])
    
    prompt = f"""
    Analyze the following Course Plan Markdown and extract all activities listed under 'SECTION ACTIVITIES'.
    Return a strict JSON list of objects matching the schema.
    
    MARKDOWN PLAN:
    {state['markdown_plan']}
    """
    
    # Usamos response_json_schema con GenAI SDK
    response = await client.aio.models.generate_content(
        model=state.get("model", "gemini-2.5-flash"),
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": PlanParsingResult.model_json_schema(),
        },
    )

    metrics = log_token_usage(response, "Parse Plan Structure", state.get("model"))
    
    # Parseamos la respuesta con Pydantic
    parsed_result = PlanParsingResult.model_validate_json(response.text)
    
    writer({"type": "status", "text": f"Found {len(parsed_result.activities)} activities to generate."})
    
    return {
        "activities_queue": parsed_result.activities, 
        "generated_content": [],
        "usage_log": current_log + [metrics] # Añadimos al historial
    }

def router_node(state: ContentState) -> str:
    """Decide qué nodo ejecutar basándose en la siguiente actividad."""
    queue = state.get("activities_queue", [])
    if not queue:
        return "end"
    
    next_activity = queue[0]
    return f"generate_{next_activity.activity_type}" # generate_quiz, generate_book, generate_assign


async def generate_quiz_node(state: ContentState) -> dict:
    """
    Genera un Quiz en dos fases:
    1. Blueprint: Define qué preguntas hacer.
    2. Construction: Genera cada pregunta individualmente.
    """
    writer = get_stream_writer()
    client = get_client()
    model = state.get("model", "gemini-2.5-flash")
    
    # 1. Obtener contexto actual
    current_activity: PlanItem = state["activities_queue"][0]
    remaining_queue = state["activities_queue"][1:]
    file_obj = client.files.get(name=state["upload_file_name"])
    current_log = state.get("usage_log", [])
    
    writer({"type": "status", "text": f"Planning Quiz: {current_activity.title}..."})

    # 2. FASE 1: Generar el Blueprint (El Arquitecto)
    blueprint_prompt = f"""
    You are an expert Instructional Designer designed to plan a Moodle Quiz.
    
    CONTEXT:
    - Activity Title: {current_activity.title}
    - Section: {current_activity.section}
    - Intent: {current_activity.description_intent}
    
    TASK:
    Create a 'QuizBlueprint' that outlines the questions needed to cover this topic effectively based on the Syllabus PDF.
    - Select a mix of question types (True/False, Multichoice, Essay) appropriate for the difficulty.
    - Do NOT generate the full question content yet, just the topic and type.
    - Plan for approximately 5-10 questions unless the intent implies otherwise.
    """

    bp_response = await client.aio.models.generate_content(
        model=model,
        contents=[file_obj, blueprint_prompt],
        config={
            "response_mime_type": "application/json",
            "response_json_schema": QuizBlueprint.model_json_schema(),
        },
    )
    bp_metrics = log_token_usage(bp_response, f"Quiz Blueprint: {current_activity.title}", model)
    current_log.append(bp_metrics)
    blueprint = QuizBlueprint.model_validate_json(bp_response.text)

    # 3. FASE 2: Generación Iterativa de Preguntas (Los Especialistas)
    generated_questions = []
    total_q = len(blueprint.questions_tasks)

    for idx, task in enumerate(blueprint.questions_tasks):
        writer({"type": "status", "text": f"Generating Q{idx+1}/{total_q}: [{task.question_type}] {task.topic_focus}..."})
        
        # Seleccionar esquema y prompt según el tipo
        if task.question_type == "truefalse":
            target_schema = TrueFalseQuestion
            type_instruction = "Create a Moodle True/False question."
        elif task.question_type == "essay":
            target_schema = EssayQuestion
            type_instruction = "Create a Moodle Essay question."
        else: # multichoice
            target_schema = MultichoiceQuestion
            type_instruction = "Create a Moodle Multi-choice question with single or multiple answers allowed."

        # Prompt específico y ligero para cada pregunta
        # NO pasamos las preguntas anteriores para ahorrar contexto, confiamos en el Blueprint para la variedad.
        question_prompt = f"""
        {type_instruction}
        
        TOPIC FOCUS: {task.topic_focus}
        DIFFICULTY: {task.difficulty}
        CONTEXT: Part of quiz '{blueprint.title}' covering '{current_activity.section}'.
        
        REQUIREMENTS:
        - Use the provided Syllabus PDF as ground truth.
        - Output strict JSON matching the schema.
        - Ensure all text fields use valid HTML.
        - Provide helpful feedback for correct/incorrect answers.
        """

        q_response = await client.aio.models.generate_content(
            model=model,
            contents=[file_obj, question_prompt],
            config={
                "response_mime_type": "application/json",
                "response_json_schema": target_schema.model_json_schema(),
            },
        )
        q_metrics = log_token_usage(q_response, f"Quiz Q{idx+1} ({task.question_type})", model)
        current_log.append(q_metrics)
        
        try:
            # Validamos y añadimos a la lista
            q_data = target_schema.model_validate_json(q_response.text)
            generated_questions.append(q_data)
        except Exception as e:
            print(f"Error parsing question {idx}: {e}")
            # Si falla una, continuamos con las siguientes para no romper todo el proceso
            continue

    # 4. FASE 3: Ensamblaje Final
    writer({"type": "status", "text": f"Assembling Quiz: {current_activity.title}..."})
    
    final_quiz = QuizActivity(
        name=blueprint.title,
        introeditor={"text": blueprint.intro_text},
        mod_settings={"questions": generated_questions}
    )

    final_record = {
        "activity_id": f"quiz_{len(state['generated_content'])}",
        "type": "quiz",
        "plan_context": current_activity.model_dump(),
        "data": final_quiz.model_dump()
    }

    writer({"type": "status", "text": f"Completed Quiz {current_activity.title} ({len(generated_questions)} questions)"})

    return {
        "generated_content": state["generated_content"] + [final_record],
        "activities_queue": remaining_queue,
        "usage_log": current_log
    }

async def generate_book_node(state: ContentState) -> dict:
    return await _generate_generic_activity(state, BookActivity, "book")

async def generate_assign_node(state: ContentState) -> dict:
    return await _generate_generic_activity(state, AssignmentActivity, "assignment")

async def _generate_generic_activity(state: ContentState, schema_class, type_label: str) -> dict:
    """Función genérica para llamar a Gemini con un esquema específico."""
    writer = get_stream_writer()
    client = get_client()
    model = state.get("model", "gemini-2.5-flash")
    current_log = state.get("usage_log", [])
    
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
        model=model,
        contents=[file_obj, prompt],
        config={
            "response_mime_type": "application/json",
            "response_json_schema": schema_class.model_json_schema(),
        },
    )
    metrics = log_token_usage(response, f"Generate {type_label}: {current_activity.title}", model)
    
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
        "activities_queue": remaining_queue,
        "usage_log": current_log + [metrics]
    }

# Sync wrapper para LangGraph
def run_async_node(node_func, state):
    return asyncio.run(node_func(state))

def create_content_graph(checkpointer: PostgresSaver):
    # Create a new state graph that will manage the JSON content generation flow
    g = StateGraph(ContentState)

    # Node 1: parse the Markdown plan and extract a queue of activities
    g.add_node("parse_plan", lambda s: run_async_node(parse_plan_node, s))

    # Nodes 2–4: generate concrete content for each activity type
    # Each node consumes the next activity in the queue and appends the
    # generated JSON to the accumulated "generated_content" list
    g.add_node("generate_quiz", lambda s: run_async_node(generate_quiz_node, s))
    g.add_node("generate_book", lambda s: run_async_node(generate_book_node, s))
    g.add_node("generate_assign", lambda s: run_async_node(generate_assign_node, s))

    # The graph always starts by parsing the full Markdown plan
    g.add_edge(START, "parse_plan")

    # After parsing the plan, we call "router_node" to decide what to do next
    # - If there are no activities left in the queue, it returns "end" and the graph finishes
    # - If there is a next activity, it returns one of: "generate_quiz", "generate_book", "generate_assign"
    g.add_conditional_edges(
        "parse_plan",
        router_node,
        {
            "generate_quiz": "generate_quiz",
            "generate_book": "generate_book",
            "generate_assign": "generate_assign",
            "end": END,
        },
    )

    # After any content node finishes, we go back to the router again
    # The router will look at the remaining queue and:
    # - send the graph to the appropriate generator node for the next activity, or
    # - send the graph to END when there are no more activities to process
    g.add_conditional_edges(
        "generate_quiz",
        router_node,
        {
            "generate_quiz": "generate_quiz",
            "generate_book": "generate_book",
            "generate_assign": "generate_assign",
            "end": END,
        },
    )
    g.add_conditional_edges(
        "generate_book",
        router_node,
        {
            "generate_quiz": "generate_quiz",
            "generate_book": "generate_book",
            "generate_assign": "generate_assign",
            "end": END,
        },
    )
    g.add_conditional_edges(
        "generate_assign",
        router_node,
        {
            "generate_quiz": "generate_quiz",
            "generate_book": "generate_book",
            "generate_assign": "generate_assign",
            "end": END,
        },
    )

    # Finally, compile the graph with the provided Postgres checkpointer
    # so that executions can be persisted and resumed.
    return g.compile(checkpointer=checkpointer)