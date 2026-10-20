"""Phase 3: cheap router. Small model + few-shot -> enum. Low confidence => run BOTH paths and merge."""
import json, time
from . import config, llm

PROMPT = """Decide the best retrieval path for a question over a corpus of company documents.
"graph": relationships between entities, multi-hop chains ("the CEO of the supplier of..."), comparisons across entities, counts/aggregations over relationships.
"vector": definitions, policy lookups, descriptions, single-fact questions about one thing.
Return ONLY JSON: {"route": "graph"|"vector", "confidence": 0-1}
Examples:
Q: Who supplies the company that Acme bought? -> {"route":"graph","confidence":0.95}
Q: Which firms do both Acme and Beta partner with? -> {"route":"graph","confidence":0.9}
Q: What does the data retention policy say about backups? -> {"route":"vector","confidence":0.95}
Q: Describe the Prism controller. -> {"route":"vector","confidence":0.85}
Q: Where is Acme headquartered? -> {"route":"vector","confidence":0.6}"""


def route(question: str) -> dict:
    try:
        out = llm.chat_json([{"role": "system", "content": PROMPT}, {"role": "user", "content": question}],
                            model=config.ROUTER_MODEL, max_tokens=60)
        r, c = out.get("route"), float(out.get("confidence", 0))
    except Exception:
        r, c = None, 0.0
    if r not in ("graph", "vector") or c < config.ROUTER_MIN_CONF:
        return {"route": "both", "confidence": c}  # fallback: run both, merge
    return {"route": r, "confidence": c}


def log_decision(**fields):
    """Phase 5 needs this data; it cannot be reconstructed later."""
    with open(config.LOG_DIR / "routing.jsonl", "a") as f:
        f.write(json.dumps({"ts": time.time(), **fields}) + "\n")
