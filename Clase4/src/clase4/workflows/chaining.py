"""Workflow 1 — Prompt Chaining.

Patrón: salida del agente A → entrada del agente B.

Caso: dado el nombre de una canción de Carlos Santana, Celia Cruz o Los
Fabulosos Cadillacs, primero un agente investigador genera datos
estructurados (año, género, contexto histórico), luego un agente
redactor produce una reseña periodística usando esos datos.

Esto demuestra:
- **SRP**: cada agente tiene una sola responsabilidad clara.
- **Open/Closed**: cambiar de modelo o de tema no requiere modificar los
  agentes; sólo se inyecta un nuevo LLM o un nuevo prompt.
"""

from __future__ import annotations

from clase4.domain.messages import SystemMessage, UserMessage
from clase4.domain.workflow import WorkflowResult, WorkflowStep
from clase4.ports.llm import LLMClient

RESEARCHER_PROMPT = (
    "Eres un investigador musical especializado en música latinoamericana "
    "(Carlos Santana, Celia Cruz, Los Fabulosos Cadillacs). Cuando te den el "
    "nombre de una canción y un artista, devuelves un informe estructurado "
    "con los siguientes encabezados en Markdown:\n"
    "# DATOS BÁSICOS\n"
    "- Artista, álbum, año, sello, género, duración.\n"
    "# CONTEXTO HISTÓRICO\n"
    "- 2-3 frases sobre el momento en que se grabó.\n"
    "# DATOS MUSICALES\n"
    "- Personal de grabación, instrumentación, tonalidad si la conoces.\n"
    "# IMPACTO CULTURAL\n"
    "- 2-3 frases sobre legado, premios y influencia.\n"
    "Sé conciso, evita relleno."
)

WRITER_PROMPT = (
    "Eres un cronista musical para un suplemento cultural. Recibes una ficha "
    "técnica sobre una canción y debes escribir una reseña periodística de "
    "180-220 palabras, con tono cálido pero riguroso, en español neutro. "
    "Usa los datos provistos, no inventes información adicional."
)


class LyricResearchAgent:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def run(self, song_query: str) -> str:
        messages = [
            SystemMessage(content=RESEARCHER_PROMPT),
            UserMessage(content=f"Investiga la canción: {song_query}"),
        ]
        return self._llm.chat(messages, temperature=0.2).content


class LyricWriterAgent:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def run(self, song_query: str, research_brief: str) -> str:
        messages = [
            SystemMessage(content=WRITER_PROMPT),
            UserMessage(
                content=(
                    f"Canción: {song_query}\n\n"
                    f"Ficha técnica:\n{research_brief}\n\n"
                    "Escribe la reseña periodística."
                )
            ),
        ]
        return self._llm.chat(messages, temperature=0.6).content


class ChainingWorkflow:
    """Encadena un investigador y un redactor."""

    name = "chaining"

    def __init__(self, llm: LLMClient) -> None:
        self._researcher = LyricResearchAgent(llm)
        self._writer = LyricWriterAgent(llm)

    def run(self, user_input: str) -> WorkflowResult:
        research = self._researcher.run(user_input)
        review = self._writer.run(user_input, research)
        
        return WorkflowResult(
            input=user_input,
            output=review,
            steps=[
                WorkflowStep(name="researcher", output=research),
                WorkflowStep(name="writer", output=review),
            ],
        )
