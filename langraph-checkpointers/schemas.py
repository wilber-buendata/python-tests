# schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# --- Estructuras Genéricas para Moodle ---
class MoodleTextFormat(BaseModel):
    text: str = Field(description="The HTML content string.")
    format: str = Field(default="1", description="1 for HTML format.")
    itemid: Optional[int] = Field(default=0, description="Internal Moodle ID, optional.")

# --- Estructuras para Quiz ---
class QuizQuestion(BaseModel):
    name: str = Field(description="Name of the question.")
    questiontext: MoodleTextFormat
    qtype: Literal["multichoice", "truefalse", "shortanswer", "essay", "calculated"] = Field(description="Type of the question.")
    defaultmark: float = Field(default=1.0)
    generalfeedback: Optional[MoodleTextFormat]
    # Simplified answer structure for the example
    answers: List[MoodleTextFormat] = Field(description="List of possible answer texts.")
    correct_answer_index: int = Field(description="Index of the correct answer in the answers list (0-based).")

class QuizActivity(BaseModel):
    name: str = Field(description="Title of the Quiz.")
    intro: MoodleTextFormat = Field(description="Description/Introduction to the quiz.")
    questions: List[QuizQuestion] = Field(description="List of questions inside this quiz.")

# --- Estructuras para Book (Libro) ---
class BookChapter(BaseModel):
    title: str = Field(description="Chapter title.")
    content: MoodleTextFormat = Field(description="HTML content of the chapter explanations.")

class BookActivity(BaseModel):
    name: str = Field(description="Title of the Book resource.")
    intro: MoodleTextFormat
    chapters: List[BookChapter]

# --- Estructuras para Assignment (Tarea) ---
class AssignmentActivity(BaseModel):
    name: str = Field(description="Title of the assignment.")
    intro: MoodleTextFormat = Field(description="Instructions for the assignment.")
    allowsubmissionsfromdate: int = Field(default=0, description="Timestamp.")
    duedate: int = Field(default=0, description="Timestamp.")

# --- Estructura para el Parseo del Plan ---
class PlanItem(BaseModel):
    section: str = Field(description="The section title this activity belongs to.")
    activity_type: Literal["quiz", "book", "assign"]
    title: str = Field(description="The exact title of the activity.")
    description_intent: str = Field(description="The brief description of what this activity contains according to the plan.")

class PlanParsingResult(BaseModel):
    activities: List[PlanItem]