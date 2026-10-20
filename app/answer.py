"""Public entry point: ask(question) runs the LangGraph pipeline and returns a JSON-friendly result."""
import time
from . import llm, router
from .pipeline import PIPELINE, citations


def ask(question: str, mode: str = "auto") -> dict:
    t0 = time.perf_counter()
    llm.WAITED[0] = 0.0
    calls0, tokens0 = llm.USAGE["calls"], llm.USAGE["tokens"]
    s = PIPELINE.invoke({"question": question, "mode": mode, "attempts": 0, "trace": []})
    used = set(citations(s["answer"]))
    result = {"answer": s["answer"], "refused": s["refused"], "citations_valid": s["valid"],
              "citation_retries": s["attempts"] - 1, "route": s["route"], "router_confidence": s["confidence"],
              "graph_facts": [f["text"] for f in s.get("facts", [])],
              "citations": [{"id": p["id"], "text": p["text"]} for p in s.get("passages", []) if p["id"] in used],
              "trace": s["trace"],
              "latency_ms": round((time.perf_counter() - t0 - llm.WAITED[0]) * 1000),
              "llm_calls": llm.USAGE["calls"] - calls0, "tokens": llm.USAGE["tokens"] - tokens0}
    router.log_decision(question=question, mode=mode, route=s["route"], confidence=s["confidence"],
                        n_facts=len(s.get("facts", [])), n_passages=len(s.get("passages", [])),
                        refused=s["refused"], latency_ms=result["latency_ms"], tokens=result["tokens"])
    return result
