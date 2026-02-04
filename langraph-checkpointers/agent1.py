# agent.py
from __future__ import annotations

import asyncio
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from langgraph.checkpoint.postgres import PostgresSaver

from google import genai
from google.genai import types


SYSTEM_INSTRUCTION = """You are a course planning assistant that converts syllabus content into structured learning sections and activities.

INPUT
- You will receive a course syllabus as a PDF file.
- You may also receive additional user instructions as plain text.

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

QUALITY BAR
- Titles must be specific and aligned with the syllabus terminology.
- The plan must be internally consistent and cover the syllabus scope without adding unrelated content.
"""


class PlanState(TypedDict):
    upload_file_name: str  # e.g. "files/..."
    user_instructions: str
    model: str


def build_user_prompt(user_instructions: str) -> str:
    base = (
        "Generate the course plan from the provided syllabus PDF.\n"
        "Follow the Output Rules strictly.\n"
    )
    if user_instructions and user_instructions.strip():
        base += "\nAdditional user instructions:\n" + user_instructions.strip() + "\n"
    return base


async def generate_plan_node(state: PlanState) -> dict:
    """
    Calls Gemini with the uploaded PDF file reference + user instructions.
    Streams tokens via LangGraph 'custom' stream.
    """
    writer = get_stream_writer()

    client = genai.Client()  # uses GEMINI_API_KEY / GOOGLE_API_KEY env vars :contentReference[oaicite:8]{index=8}
    model_name = state.get("model")

    # Get the file object reference for contents
    file_obj = client.files.get(name=state["upload_file_name"])

    prompt_text = build_user_prompt(state.get("user_instructions", ""))

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.2,
        max_output_tokens=8192,
    )

    # Streaming generation (async)
    async for chunk in await client.aio.models.generate_content_stream(
        model=model_name,
        contents=[file_obj, prompt_text],
        config=config,
    ):
        if chunk.text:
            print(chunk.text)
            # Emit custom stream events
            writer({"type": "token", "text": chunk.text})

    # (Optional) signal end
    writer({"type": "done"})
    return {}  # we don't need to store the whole plan in state


def generate_plan_node_sync(state: PlanState) -> dict:
    """Synchronous wrapper for LangGraph sync runner.

    LangGraph's `graph.stream` expects sync nodes when using the sync runner.
    We bridge to the existing async implementation with asyncio.run.
    """
    return asyncio.run(generate_plan_node(state))


def create_graph(checkpointer: PostgresSaver):
    g = StateGraph(PlanState)
    g.add_node("generate_plan", generate_plan_node_sync)
    g.add_edge(START, "generate_plan")
    g.add_edge("generate_plan", END)
    return g.compile(checkpointer=checkpointer)

