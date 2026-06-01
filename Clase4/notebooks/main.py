import importlib
import subprocess
import sys

def _ensure(pkg: str, import_name: str | None = None) -> None:
    name = import_name or pkg
    try:
        importlib.import_module(name)
    except ImportError:
        print(f"Instalando {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_ensure("langgraph")

# Las versiones "sin LangGraph" vienen del paquete del proyecto.
from clase4.domain.messages import AIMessage, BaseMessage, SystemMessage, UserMessage
from clase4.domain.workflow import WorkflowResult, WorkflowStep
from clase4.ports.llm import LLMClient
from clase4.workflows.chaining import ChainingWorkflow
from clase4.workflows.routing import RoutingWorkflow, build_default_specialists
from clase4.workflows.parallel import ParallelInsightsWorkflow
from clase4.workflows.evaluator_optimizer import EvaluatorOptimizerWorkflow
from clase4.workflows.orchestrator import OrchestratorWorkflow, SubTask

print("OK — paquete clase4 disponible.")


from dataclasses import dataclass, field
from typing import Callable


@dataclass

class ScriptedLLM:
    """LLM determinístico para demos sin red."""

    responder: Callable[[list[BaseMessage]], str]
    model_name_value: str = "scripted"
    calls: list[list[BaseMessage]] = field(default_factory=list)

    @property
    def model_name(self) -> str:
        return self.model_name_value

    def chat(
        self,
        messages: list[BaseMessage],
        *,
        temperature: float | None = None,
    ) -> AIMessage:
        self.calls.append(messages)
        return AIMessage(content=self.responder(messages))


def make_responder(rules: list[tuple[str, str]], default: str = "(sin respuesta)") -> Callable:
    """Devuelve un responder que matchea por substring en el último user message."""

    def responder(messages: list[BaseMessage]) -> str:
        last = messages[-1].content.lower()
        for needle, reply in rules:
            if needle.lower() in last:
                return reply
        return default

    return responder



print('=='*64)
print('Chaining')
print('=='*64)

chaining_llm = ScriptedLLM(
    responder=make_responder(
        rules=[
            (
                "investiga la canción",
                "# DATOS BÁSICOS\n- Año: 1970\n- Álbum: Abraxas\n- Sello: Columbia\n"
                "# CONTEXTO HISTÓRICO\n- Versión latina del clásico de Tito Puente.",
            ),
            (
                "ficha técnica",
                "Reseña: Oye Cómo Va condensa el cruce entre la timba cubana y el rock psicodélico... (180 palabras)",
            ),
        ],
    )
)

chaining = ChainingWorkflow(llm=chaining_llm)
result = chaining.run("Oye Cómo Va — Carlos Santana")

print("=== STEPS ===")
for step in result.steps:
    print(f"\n[{step.name}]")
    print(step.output[:160], "...")
print("\n=== OUTPUT FINAL ===")
print(result.output)


print()
print('=='*64)
print('ChainigLangGraph')
print('=='*64)

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class ChainState(TypedDict):
    song: str
    research: str
    review: str


def researcher_node(state: ChainState) -> dict:
    msgs = [SystemMessage(content="(researcher prompt)"),
            UserMessage(content=f"Investiga la canción: {state['song']}")]
    return {"research": chaining_llm.chat(msgs).content}


def writer_node(state: ChainState) -> dict:
    msgs = [SystemMessage(content="(writer prompt)"),
            UserMessage(content=f"Canción: {state['song']}\nficha técnica:\n{state['research']}")]
    return {"review": chaining_llm.chat(msgs).content}

# Fase 0
builder = StateGraph(ChainState)


# Fase 1 asignar nodos o responsabiliades 
builder.add_node("researcher", researcher_node)
builder.add_node("writer", writer_node)

# Fase2 El mapa 

builder.add_edge(START, "researcher")
builder.add_edge("researcher", "writer")

builder.add_edge("writer", END)
chain_graph = builder.compile()


# Creando el Output con Grafos 
final_state = chain_graph.invoke({"song": "Oye Cómo Va — Carlos Santana"})
print("research:", final_state["research"][:100], "...")
print("review  :", final_state["review"][:100], "...")
print(chain_graph.get_graph().draw_ascii())

######################
# ---- Routing ----- #
######################


print('=='*64)
print('Routing')
print('=='*64)


def routing_responder(messages: list[BaseMessage]) -> str:
    system = messages[0].content.lower()
    question = messages[-1].content.lower()
    if "router de preguntas" in system:
        if "watchmen" in question or "cómic" in question or "comic" in question:
            return "comics_expert"
        if "celia" in question or "salsa" in question:
            return "salsa_expert"
        return "latin_rock_expert"
    if "crítico de cómic" in system:
        return "Watchmen (1986-87) de Alan Moore es una deconstrucción del superhéroe..."
    if "musicólogo" in system and "salsa" in system:
        return "Celia Cruz dialogó con la Fania All-Stars en los 70..."
    if "rock latinoamericano" in system:
        return "Santana fusionó blues, jazz y ritmos afro-caribeños desde 1969..."
    return "(respuesta genérica)"

routing_llm = ScriptedLLM(responder=routing_responder)
routing = RoutingWorkflow(llm=routing_llm, specialists=build_default_specialists(routing_llm))
res = routing.run("Explica qué hace Watchmen tan icónico")
print("Especialista elegido:", res.metadata.get("chosen_specialist"))
print("Respuesta:", res.output[:120], "...")

# ---- Langgraph ---- #


print('='*64)
print('Routing LangGraph')
print('=='*64)

from typing import Literal

class RouteState(TypedDict):
    question: str
    specialist: str
    answer: str


def router_node(state: RouteState) -> dict:
    decision = routing_llm.chat(
        [SystemMessage(content="Eres un router de preguntas. Devolvé el nombre del especialista."),
         UserMessage(content=state["question"])]
    ).content.strip().lower()
    for key in ("comics_expert", "salsa_expert", "latin_rock_expert"):
        if key in decision:
            return {"specialist": key}
    return {"specialist": "comics_expert"}  # fallback


def choose_route(state: RouteState) -> Literal["comics_expert", "salsa_expert", "latin_rock_expert"]:
    return state["specialist"]  # type: ignore[return-value]


SPECIALIST_SYSTEMS = {
    "comics_expert": "Eres un crítico de cómic, riguroso con autores y fechas.",
    "salsa_expert":  "Eres un musicólogo especializado en salsa y son cubano.",
    "latin_rock_expert": "Eres un crítico de rock latinoamericano.",
}




def make_specialist_node(name: str):
    def node(state: RouteState) -> dict:
        ans = routing_llm.chat(
            [SystemMessage(content=SPECIALIST_SYSTEMS[name]),
             UserMessage(content=state["question"])]
        ).content
        return {"answer": ans}
    return node


b = StateGraph(RouteState)
b.add_node("router", router_node)
b.add_node("comics_expert", make_specialist_node("comics_expert"))
b.add_node("salsa_expert", make_specialist_node("salsa_expert"))
b.add_node("latin_rock_expert", make_specialist_node("latin_rock_expert"))

# ----- Arquitectura ------ #
b.add_edge(START, "router")
b.add_conditional_edges("router", choose_route,
                        {"comics_expert": "comics_expert",
                         "salsa_expert": "salsa_expert",
                         "latin_rock_expert": "latin_rock_expert"})
for name in ("comics_expert", "salsa_expert", "latin_rock_expert"):
    b.add_edge(name, END)

route_graph = b.compile()
out = route_graph.invoke({"question": "Explica qué hace Watchmen tan icónico"})
print("Elegido:", out["specialist"])
print("Respuesta:", out["answer"][:120], "...")
print(route_graph.get_graph().draw_ascii())



print('=='*64)
print('Parallel')
print('=='*64)


# --- Sin LangGraph: ParallelInsightsWorkflow ---
# El responder mira el system prompt para distinguir qué especialista llama.

def parallel_responder(messages: list[BaseMessage]) -> str:
    system = messages[0].content.lower() if messages else ""
    if "musicólogo" in system:
        return "Análisis musical: clave Am, montuno cubano, guitarra sobre clave 2-3."
    if "historiador cultural" in system:
        return "Análisis histórico: NY 1970, salsa boom, Fania, contracultura latina."
    if "sociólogo" in system:
        return "Análisis social: la canción atraviesa diásporas caribeñas e identidad afrolatina."
    if "editor cultural" in system:
        return "Síntesis editorial (250 palabras) que cohesiona los tres análisis previos."
    return "(análisis genérico)"

parallel_llm = ScriptedLLM(responder=parallel_responder)
parallel_wf = ParallelInsightsWorkflow(llm=parallel_llm)
res = parallel_wf.run("Oye Cómo Va — Santana")
for step in res.steps:
    print(f"[{step.name}] {step.output[:80]}")
print("\nFINAL:", res.output[:80], "...")


print('=='*64)
print('Parallel LangGraph')
print('=='*64)

from operator import add
from typing import Annotated

class ParallelState(TypedDict):
    topic: str
    analyses: Annotated[list[str], add]   # reducer: concatena listas
    summary: str


def musicologo(state: ParallelState) -> dict:
    out = parallel_llm.chat(
        [SystemMessage(content="Eres un musicólogo. Análisis musical estricto."),
         UserMessage(content=state["topic"])]
    ).content
    return {"analyses": [f"musicologo: {out}"]}


def historiador(state: ParallelState) -> dict:
    out = parallel_llm.chat(
        [SystemMessage(content="Eres un historiador cultural. Contexto histórico."),
         UserMessage(content=state["topic"])]
    ).content
    return {"analyses": [f"historiador: {out}"]}


def sociologo(state: ParallelState) -> dict:
    out = parallel_llm.chat(
        [SystemMessage(content="Eres un sociólogo de la cultura popular latinoamericana."),
         UserMessage(content=state["topic"])]
    ).content
    return {"analyses": [f"sociologo: {out}"]}


#############################
### Very Important LLM #####
###########################

def synthesizer(state: ParallelState) -> dict:
    body = "\n".join(state["analyses"]) #### NO lo pelie , no lo cambie siempre uina estas ideas por mas largas que sean , busque optimizar la memoria nada mas
    out = parallel_llm.chat(
        [SystemMessage(content="Eres un editor cultural. Sintetiza los aportes."),
         UserMessage(content=f"Tema: {state['topic']}\n\nAportes:\n{body}")]
    ).content
    return {"summary": out}


g = StateGraph(ParallelState)
g.add_node("musicologo", musicologo)
g.add_node("historiador", historiador)
g.add_node("sociologo", sociologo)
g.add_node("synthesizer", synthesizer)
for name in ("musicologo", "historiador", "sociologo"):
    g.add_edge(START, name)
    g.add_edge(name, "synthesizer")
g.add_edge("synthesizer", END)
parallel_graph = g.compile()

out = parallel_graph.invoke({"topic": "Oye Cómo Va — Santana", "analyses": []})
print(f"Recolectados {len(out['analyses'])} análisis en paralelo.")
print("Síntesis:", out["summary"][:80], "...")

print(parallel_graph.get_graph().draw_ascii())

print('=='*64)
print('Evaluator Optimizer')
print('=='*64)

class EvalState(TypedDict):
    topic: str
    draft: str
    feedback: str
    attempts: int
    approved: bool


def writer_node_eo(state: EvalState) -> dict:
    prompt = f"Cómic: {state['topic']}\nFeedback previo: {state.get('feedback') or '(ninguno)'}"
    out = eval_llm.chat(
        [SystemMessage(content="Eres un crítico literario que escribe reseñas sin spoilers."),
         UserMessage(content=prompt)]
    ).content
    return {"draft": out, "attempts": state.get("attempts", 0) + 1}


def evaluator_node(state: EvalState) -> dict:
    verdict = eval_llm.chat(
        [SystemMessage(content="Eres un editor de spoilers. Veredicto APROBADO/RECHAZADO."),
         UserMessage(content=state["draft"])]
    ).content
    approved = "APROBADO" in verdict.split("\n", 1)[0].upper()
    feedback = "" if approved else verdict.split("FEEDBACK:", 1)[-1].strip()
    return {"approved": approved, "feedback": feedback}


def should_retry(state: EvalState) -> Literal["writer", "end"]:
    if state["approved"]:
        return "end"
    if state["attempts"] >= 3:
        return "end"
    return "writer"

attempts = {"count": 0}

def writer_then_evaluator(messages: list[BaseMessage]) -> str:
    system = messages[0].content.lower()
    if "crítico literario" in system:               # ← writer
        attempts["count"] += 1
        if attempts["count"] == 1:
            return "La reseña revela el final: Rorschach muere al cierre."   # spoiler
        return "Reseña limpia: foco en tema, contexto, autor y estilo visual."
    if "editor de spoilers" in system:              # ← evaluator
        text = messages[-1].content.lower()
        if "rorschach muere" in text or "al cierre" in text:
            return "VEREDICTO: RECHAZADO\nFEEDBACK: contiene spoilers explícitos del final."
        return "VEREDICTO: APROBADO"
    return "VEREDICTO: APROBADO"

#eval_llm = ScriptedLLM(responder=writer_then_evaluator)

eval_llm = ScriptedLLM(responder=writer_then_evaluator)
eo = EvaluatorOptimizerWorkflow(llm=eval_llm, max_retries=3)
res = eo.run("Reseña de Watchmen sin spoilers")
b = StateGraph(EvalState)
b.add_node("writer", writer_node_eo)
b.add_node("evaluator", evaluator_node)
b.add_edge(START, "writer")
b.add_edge("writer", "evaluator")
b.add_conditional_edges("evaluator", should_retry, {"writer": "writer", "end": END})
eo_graph = b.compile()

attempts["count"] = 0  # reset del contador del responder
out = eo_graph.invoke({"topic": "Watchmen sin spoilers",
                       "draft": "", "feedback": "", "attempts": 0, "approved": False})
print(f"Aprobado: {out['approved']}  Intentos: {out['attempts']}")
print("Final  :", out["draft"])



print('=='*64)
print('Orchestrator')
print('=='*64)

# --- Sin LangGraph: OrchestratorWorkflow ---
# Dispatch por system prompt:
#  * "editor jefe"        → planner JSON
#  * "fact-checker"       → worker datos
#  * "musicólogo"         → worker musical
#  * "historiador de la música popular" → worker legado
#  * "editor de cierre"   → synthesizer

def orch_responder(messages: list[BaseMessage]) -> str:
    system = messages[0].content.lower()
    if "editor jefe" in system:
        return ('''{"analysis": "cobertura editorial multi-ángulo",
                    "tasks": [
                      {"type": "datos",   "description": "ficha técnica"},
                      {"type": "musical", "description": "análisis musical"},
                      {"type": "legado",  "description": "impacto cultural"}
                    ]}''')
    if "fact-checker" in system:
        return "- Artista: Santana\n- Álbum: Abraxas\n- Año: 1970\n- Sello: Columbia"
    if "musicólogo" in system:
        return "Tonalidad Am, montuno cubano, guitarra sobre clave 2-3."
    if "historiador de la música popular" in system:
        return "Influencia decisiva en el latin rock posterior."
    if "editor de cierre" in system:
        return "Ensayo final (~400 palabras) que entrelaza datos, análisis musical y legado cultural."
    return "(worker genérico)"

orch_llm = ScriptedLLM(responder=orch_responder)
orch = OrchestratorWorkflow(llm=orch_llm)
res = orch.run("Oye Cómo Va — Santana")
print("Sub-tareas planeadas:", res.metadata["sub_tasks"])
print("\nEnsayo final:")
print(res.output)




import json
import re

from langgraph.constants import Send


class OrchState(TypedDict):
    topic: str
    tasks: list[dict]
    partials: Annotated[list[str], add]
    final: str


def planner_node(state: OrchState) -> dict:
    raw = orch_llm.chat(
        [SystemMessage(content="Eres un editor jefe. Devolvé JSON con tasks."),
         UserMessage(content=state["topic"])]
    ).content
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    payload = json.loads(m.group(0)) if m else {"tasks": []}
    return {"tasks": payload.get("tasks", [])}


def worker_node(state: dict) -> dict:
    # state acá es el payload de cada Send, no el estado global
    persona = {"datos":   "Eres un fact-checker musical.",
               "musical": "Eres un musicólogo.",
               "legado":  "Eres un historiador de la música popular."}.get(
                  state["type"], "Eres un asistente cultural.")
    out = orch_llm.chat(
        [SystemMessage(content=persona),
         UserMessage(content=f"Tema: {state['topic']}\nSub-tarea: {state['description']}")]
    ).content
    return {"partials": [f"[{state['type']}] {out}"]}


def synth_node(state: OrchState) -> dict:
    body = "\n".join(state["partials"])
    out = orch_llm.chat(
        [SystemMessage(content="Eres un editor de cierre. Sintetizá los aportes."),
         UserMessage(content=f"Tema: {state['topic']}\nAportes:\n{body}")]
    ).content
    return {"final": out}


def fan_out(state: OrchState) -> list:
    return [Send("worker", {**task, "topic": state["topic"]}) for task in state["tasks"]]


b = StateGraph(OrchState)
b.add_node("planner", planner_node)
b.add_node("worker", worker_node)
b.add_node("synth", synth_node)

b.add_edge(START, "planner")
b.add_conditional_edges("planner", fan_out, ["worker"])
b.add_edge("worker", "synth")
b.add_edge("synth", END)
orch_graph = b.compile()

out = orch_graph.invoke({"topic": "Oye Cómo Va — Santana",
                         "tasks": [], "partials": [], "final": ""})
print("Sub-tareas:", [t["type"] for t in out["tasks"]])
print(f"Workers que respondieron: {len(out['partials'])}")
print("\nEnsayo final:")
print(out["final"])
