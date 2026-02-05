# schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union

# --- Estructuras Genéricas para Moodle ---
class MoodleTextFormat(BaseModel):
    text: str = Field(description="The HTML content string.")
    format: str = Field(default="1", description="1 for HTML format.")
    itemid: Optional[int] = Field(default=0, description="Internal Moodle ID, optional.")

# --- Estructuras para Quiz ---


class IntroEditor(BaseModel):
    text: str = Field(
        description=(
            "Contenido en formato HTML seguro."
            "Puedes incluir etiquetas HTML seguras y marcadores {{image: ...}}."
        )
    )


class TrueFalseQuestion(BaseModel):
    qtype: Literal["truefalse"] = Field(description="Debe ser truefalse")
    name: str = Field(description="Nombre visible de la pregunta.")
    defaultmark: float = Field(default=1.0, description="Puntaje asignado, por defecto debe ser 1")
    questiontext: IntroEditor = Field(
        description="Enunciado de la pregunta en formato HTML. Puedes incluir etiquetas HTML seguras."
    )
    generalfeedback: IntroEditor = Field(
        description="Retroalimentación general para la pregunta en formato HTML."
    )
    feedbacktrue: IntroEditor = Field(
        description=(
            "Retroalimentación mostrada cuando la respuesta seleccionada es Verdadero, en formato HTML."
        )
    )
    feedbackfalse: IntroEditor = Field(
        description=(
            "Retroalimentación mostrada cuando la respuesta seleccionada es Falso, en formato HTML."
        )
    )
    correctanswer: Literal[0, 1] = Field(
        description="Respuesta correcta (0 = Falso, 1 = Verdadero)"
    )


class EssayQuestion(BaseModel):
    qtype: Literal["essay"] = Field(description="Debe ser essay")
    category: str = Field(
        description="Identificador de categoría Moodle en formato jerárquico."
    )
    name: str = Field(description="Nombre visible de la pregunta.")
    defaultmark: float = Field(default=1.0, description="Puntaje asignado, por defecto debe ser 1")
    questiontext: IntroEditor = Field(
        description="Enunciado de la pregunta de ensayo en formato HTML."
    )
    generalfeedback: IntroEditor = Field(
        description=(
            "Retroalimentación general que se muestra al estudiante después de responder."
        )
    )
    responsetemplate: IntroEditor = Field(
        description="Plantilla de respuesta sugerida para el estudiante, en formato HTML."
    )
    graderinfo: IntroEditor = Field(
        description=(
            "Información para los calificadores en formato HTML. No visible para el estudiante."
        )
    )
    responserequired: Literal[0, 1] = Field(
        description=(
            "1 para requerir al estudiante que introduzca texto, 0 para no requerirlo"
        )
    )
    responsefieldlines: int = Field(
        description="Número de líneas para el campo de respuesta."
    )
    minwordenabled: Literal[0, 1] = Field(
        description=(
            "1 para activar la validación de mínimo de palabras, 0 para desactivarla"
        )
    )
    minwordlimit: Optional[int] = Field(
        default=None,
        description=(
            "Número mínimo de palabras permitidas en la respuesta, debe ser null si minwordenabled es 0"
        ),
    )
    maxwordenabled: Literal[0, 1] = Field(
        description=(
            "1 para activar la validación de máximo de palabras, 0 para desactivarla"
        )
    )
    maxwordlimit: Optional[int] = Field(
        default=None,
        description=(
            "Número máximo de palabras permitidas en la respuesta, debe ser null si maxwordenabled es 0"
        ),
    )
    attachments: Literal[-1, 0, 1, 2, 3] = Field(
        default=0,
        description=(
            "Número de archivos adjuntos permitidos. -1 para ilimitado. Por defecto 0."
        ),
    )
    attachmentsrequired: Literal[0, 1, 2, 3] = Field(
        description=(
            "Archivos adjuntos requeridos. 0 significa opcional. "
            "Si es diferente de 0 debe ser menor o igual a attachments."
        )
    )
    filetypeslist: str = Field(
        default="document,image,presentation",
        description=(
            "Tipos de archivos permitidos, por defecto document,image,presentation"
        ),
    )


class MultichoiceAnswerFeedback(BaseModel):
    text: str = Field(
        description=(
            "Retroalimentación específica para esta alternativa de respuesta, en formato HTML."
        )
    )


class MultichoiceAnswer(BaseModel):
    text: str = Field(
        description="Texto de la alternativa de respuesta en formato HTML."
    )
    fraction: Union[str, float] = Field(
        description=(
            "Usa 1.0 para correcta, 0.0 para incorrecta y valores intermedios para parcialmente correcta."
        )
    )
    feedback: MultichoiceAnswerFeedback = Field(
        description="Retroalimentación específica para esta alternativa de respuesta."
    )


class MultichoiceQuestion(BaseModel):
    qtype: Literal["multichoice"] = Field(description="Debe ser multichoice")
    category: str = Field(
        description="Identificador de categoría Moodle en formato jerárquico."
    )
    name: str = Field(description="Nombre visible de la pregunta.")
    defaultmark: float = Field(default=1.0, description="Puntaje asignado, por defecto debe ser 1")
    questiontext: IntroEditor = Field(
        description="Enunciado de la pregunta de opción múltiple en formato HTML."
    )
    generalfeedback: IntroEditor = Field(
        description=(
            "Retroalimentación general para la pregunta de opción múltiple en formato HTML."
        )
    )
    single: Union[str, int] = Field(
        description="1 para única respuesta, 0 para múltiples respuestas."
    )
    shuffleanswers: Union[str, int]
    answernumbering: str
    answer: List[MultichoiceAnswer] = Field(
        description="Listado de alternativas de respuesta.",
        min_length=2,
    )
    correctfeedback: IntroEditor = Field(
        description=(
            "Mensaje mostrado cuando la respuesta del estudiante es completamente correcta."
        )
    )
    partiallycorrectfeedback: IntroEditor = Field(
        description=(
            "Mensaje mostrado cuando la respuesta del estudiante es parcialmente correcta."
        )
    )
    incorrectfeedback: IntroEditor = Field(
        description=(
            "Mensaje mostrado cuando la respuesta del estudiante es incorrecta."
        )
    )
    shownumcorrect: Union[str, int]
    hint: List[IntroEditor] = Field(
        description="Listado de pistas para la pregunta."
    )
    penalty: Union[str, float]


QuizQuestion = Union[TrueFalseQuestion, EssayQuestion, MultichoiceQuestion]


class QuizModSettings(BaseModel):
    questions: List[QuizQuestion] = Field(
        description=(
            "Listado de preguntas que la IA desea generar. "
            "El formatter completa los campos faltantes usando plantillas por tipo."
        )
    )

class QuestionTask(BaseModel):
    """Define la intención de una sola pregunta antes de generarla."""
    question_type: Literal["truefalse", "multichoice", "essay"]
    topic_focus: str = Field(description="El concepto específico del syllabus que esta pregunta evaluará.")
    difficulty: Literal["easy", "medium", "hard"]

class QuizBlueprint(BaseModel):
    """El plan maestro del quiz antes de generar el contenido real."""
    title: str
    intro_text: str = Field(description="Texto introductorio HTML para el quiz.")
    questions_tasks: List[QuestionTask] = Field(description="Lista de preguntas planificadas para generar.")

class QuizActivity(BaseModel):
    name: str = Field(description="Título del Quiz")
    introeditor: IntroEditor = Field(
        description=(
            "Descripción del cuestionario. Puedes incluir {{image: ...}} en el HTML."
        )
    )
    mod_settings: QuizModSettings = Field(
        description="Configuración específica del módulo quiz, incluyendo preguntas."
    )

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