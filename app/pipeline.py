"""The answer pipeline as a LangGraph state machine.

    START -> route -> graph ----+
                  \\-> vector ---+--> merge -> generate -> validate --(citations invalid)--> generate (retry, max 3)
                                                              \\--(valid or refused)--> END

- route:    small model picks graph | vector | both (low confidence => both, run IN PARALLEL)
- graph:    template-based Cypher retrieval (the model never writes Cypher)
- vector:   pgvector HNSW search; also used as a fallback when the graph returns fewer than 3 facts
- merge:    label graph facts and passages, make every cited chunk id resolvable to real text
- generate: LangChain prompt -> ChatOpenAI answer with chunk-id citations
- validate: every citation must be a retrieved chunk id; otherwise feed the error back and regenerate
Each node returns a partial state update; `trace` records which nodes ran."""
import operator
import re
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from . import config, graph, llm, router, vector

SYSTEM = """Answer the question using ONLY the context. Rules:
- Cite every claim with chunk ids in plain ASCII square brackets [ ] (never 【 】), e.g. [helios:0] or [helios:0, norda:1].
- GRAPH FACTS are verified relationships extracted from the passages; each lists the chunk ids that support it. Cite those ids.
- Multi-hop questions: follow the facts link by link from the entity named in the question to the answer, citing each link. Example: the facts "A partners with B", "C supplies B", "D acquired C", "E is the CEO of D" answer "Who is the CEO of the company that acquired the supplier of the partner of A?" with E.
- Names match even when written differently ("Acme", "Acme Inc." and "ACME CORPORATION" are the same company).
- If the facts form a complete chain to the answer, give the answer. Reply with exactly INSUFFICIENT_CONTEXT only when a link in the chain is genuinely missing, or when the question asks for something the context does not contain.
- Never use outside knowledge."""

ANSWER_PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", "{context}\n\nQuestion: {question}")])
MAX_ATTEMPTS = 3
MIN_GRAPH_FACTS = 3


class State(TypedDict, total=False):
    question: str
    mode: str                     # auto | graph | vector | both
    route: str
    confidence: float
    facts: list[dict]
    passages: list[dict]
    allowed: list[str]
    messages: list
    answer: str
    refused: bool
    valid: bool
    attempts: int
    plan_len: int
    rescued: int
    trace: Annotated[list, operator.add]   # parallel branches append to it safely


def citations(text: str) -> list[str]:
    ids = []
    for grp in re.findall(r"\[([^\[\]]+)\]", text):
        ids += [x.strip() for x in grp.split(",") if re.fullmatch(r"[\w\-.]+:\d+", x.strip())]
    return ids


def clean(text: str) -> str:
    """gpt-oss writes fullwidth brackets and odd spaces; normalize before validating."""
    for bad, good in (("【", "["), ("】", "]"), ("\u00a0", " "), ("\u202f", " "), ("\u2009", " ")):
        text = text.replace(bad, good)
    return text


# ---- nodes ---------------------------------------------------------------------------------------------------
def route_node(state: State) -> dict:
    mode = state.get("mode", "auto")
    d = router.route(state["question"]) if mode == "auto" else {"route": mode, "confidence": 1.0}
    return {"route": d["route"], "confidence": d["confidence"], "trace": ["route"]}


def graph_node(state: State) -> dict:
    r = graph.retrieve(state["question"])
    return {"facts": r["facts"], "plan_len": len(r["plan"]["steps"]), "trace": ["graph"]}


def vector_node(state: State) -> dict:
    return {"passages": vector.search(state["question"], config.TOP_K), "trace": ["vector"]}


def merge_node(state: State) -> dict:
    facts, passages = state.get("facts", []), list(state.get("passages", []))
    have = {p["id"] for p in passages}
    need = sorted({c for f in facts for c in f["chunk_ids"]} - have)
    passages += vector.get_chunks(need)  # every chunk id a graph fact cites must resolve to real text
    ctx = "GRAPH FACTS (from the knowledge graph):\n" + (
        "\n".join(f"- {f['text']} [{', '.join(f['chunk_ids'])}]" for f in facts) or "(none)")
    ctx += "\n\nPASSAGES:\n" + "\n".join(f"[{p['id']}] {p['text']}" for p in passages)
    messages = ANSWER_PROMPT.format_messages(context=ctx, question=state["question"])
    return {"passages": passages, "allowed": [p["id"] for p in passages], "messages": messages, "trace": ["merge"]}


def generate_node(state: State) -> dict:
    out = clean(llm.chat(state["messages"], max_tokens=500).strip())
    return {"answer": out, "attempts": state.get("attempts", 0) + 1, "trace": ["generate"]}


def validate_node(state: State) -> dict:
    out = state["answer"]
    if out.startswith("INSUFFICIENT_CONTEXT"):
        facts = state.get("facts", [])
        plan_len = state.get("plan_len", 0)
        if plan_len >= 2 and len(facts) == plan_len and state.get("rescued", 0) < 2:
            cited = {c for f in facts for c in f["chunk_ids"]}
            ctx = ("COMPLETE CHAIN, in hop order. The last hop contains the answer:\n" +
                   "\n".join(f"{i+1}. {f['text']} [{', '.join(f['chunk_ids'])}]" for i, f in enumerate(facts)) +
                   "\n\nPASSAGES:\n" +
                   "\n".join(f"[{p['id']}] {p['text']}" for p in state["passages"] if p["id"] in cited))
            msgs = ANSWER_PROMPT.format_messages(context=ctx, question=state["question"])
            return {"refused": False, "valid": False, "rescued": state.get("rescued", 0) + 1, "messages": msgs, "trace": ["validate"]}
        if plan_len >= 2 and len(facts) == plan_len:  # complete verified chain: answer from the last hop
            last = facts[-1]
            ans = f"{last['text']} [{', '.join(last['chunk_ids'])}]"
            return {"answer": ans, "refused": False, "valid": True, "trace": ["validate"]}
        return {"refused": True, "valid": True, "trace": ["validate"]}
    cites = citations(out)
    bad = [c for c in cites if c not in set(state["allowed"])]
    if cites and not bad:
        return {"refused": False, "valid": True, "trace": ["validate"]}
    feedback = (f"Citations missing or invalid ({bad or 'none given'}). "
                f"Allowed ids: {sorted(state['allowed'])}. Rewrite with valid citations.")
    return {"refused": False, "valid": False, "trace": ["validate"],
            "messages": state["messages"] + [AIMessage(content=out), HumanMessage(content=feedback)]}


# ---- routing between nodes -----------------------------------------------------------------------------------
def fan_out(state: State) -> list[str]:
    return {"graph": ["graph"], "vector": ["vector"]}.get(state["route"], ["graph", "vector"])  # both = parallel


def after_graph(state: State) -> str:
    too_few = len(state.get("facts", [])) < MIN_GRAPH_FACTS
    return "vector" if state["route"] == "graph" and too_few else "merge"  # graph found little: add vector passages


def should_retry(state: State) -> str:
    return "generate" if not state["valid"] and state["attempts"] < MAX_ATTEMPTS else END


def build():
    g = StateGraph(State)
    for name, fn in (("route", route_node), ("graph", graph_node), ("vector", vector_node),
                     ("merge", merge_node), ("generate", generate_node), ("validate", validate_node)):
        g.add_node(name, fn)
    g.add_edge(START, "route")
    g.add_conditional_edges("route", fan_out, ["graph", "vector"])
    g.add_conditional_edges("graph", after_graph, ["vector", "merge"])
    g.add_edge("vector", "merge")
    g.add_edge("merge", "generate")
    g.add_edge("generate", "validate")
    g.add_conditional_edges("validate", should_retry, ["generate", END])
    return g.compile()


PIPELINE = build()
