"""LangGraph orchestration with the retrievers and the LLM mocked: tests the control flow, not the models."""
import pytest
from app import answer, graph, llm, router, vector

FACT = {"text": "A supplies B.", "chunk_ids": ["co01:0"], "evidence": ""}
PASSAGE = {"id": "co02:0", "text": "passage two", "score": 0.9}


@pytest.fixture()
def env(monkeypatch):
    calls = {"graph": 0, "vector": 0, "llm": 0}
    cfg = {"route": "graph", "facts": [FACT] * 4, "replies": ["Answer [co01:0]"]}

    def fake_route(q):
        return {"route": cfg["route"], "confidence": 0.9}

    def fake_graph(q):
        calls["graph"] += 1
        return {"facts": cfg["facts"], "plan": {}}

    def fake_vector(q, k=5):
        calls["vector"] += 1
        return [PASSAGE]

    def fake_chunks(ids):
        return [{"id": i, "text": f"text of {i}", "score": None} for i in ids]

    def fake_chat(messages, **kw):
        reply = cfg["replies"][min(calls["llm"], len(cfg["replies"]) - 1)]
        calls["llm"] += 1
        return reply

    monkeypatch.setattr(router, "route", fake_route)
    monkeypatch.setattr(graph, "retrieve", fake_graph)
    monkeypatch.setattr(vector, "search", fake_vector)
    monkeypatch.setattr(vector, "get_chunks", fake_chunks)
    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(router, "log_decision", lambda **kw: None)
    return calls, cfg


def test_graph_route_with_enough_facts_skips_vector(env):
    calls, cfg = env
    r = answer.ask("q")
    assert calls["vector"] == 0 and r["trace"] == ["route", "graph", "merge", "generate", "validate"]
    assert r["citations_valid"] and r["citations"][0]["id"] == "co01:0"   # graph-cited chunk was fetched


def test_graph_route_with_few_facts_falls_back_to_vector(env):
    calls, cfg = env
    cfg["facts"] = [FACT]
    r = answer.ask("q")
    assert calls["vector"] == 1 and "vector" in r["trace"]


def test_both_route_runs_both_in_parallel_and_merges_once(env):
    calls, cfg = env
    cfg["route"] = "both"
    r = answer.ask("q")
    assert calls["graph"] == 1 and calls["vector"] == 1 and r["trace"].count("merge") == 1


def test_invalid_citation_is_regenerated(env):
    calls, cfg = env
    cfg["replies"] = ["Answer [made:up]", "Answer [co01:0]"]
    r = answer.ask("q")
    assert r["citations_valid"] and r["citation_retries"] == 1 and r["trace"].count("generate") == 2


def test_gives_up_after_three_invalid_attempts(env):
    calls, cfg = env
    cfg["replies"] = ["no citations at all"]
    r = answer.ask("q")
    assert not r["citations_valid"] and r["trace"].count("generate") == 3


def test_refusal_is_valid_and_not_retried(env):
    calls, cfg = env
    cfg["replies"] = ["INSUFFICIENT_CONTEXT"]
    r = answer.ask("q")
    assert r["refused"] and r["citations_valid"] and r["trace"].count("generate") == 1


def test_explicit_mode_bypasses_router(env, monkeypatch):
    calls, cfg = env
    monkeypatch.setattr(router, "route", lambda q: (_ for _ in ()).throw(AssertionError("router must not run")))
    r = answer.ask("q", mode="vector")
    assert r["route"] == "vector" and calls["graph"] == 0
