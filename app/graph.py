"""Graph retrieval. SECURITY CONTROL: the LLM never writes Cypher. It picks a template and fills parameters
(entity names, relationship steps, depth); every value is whitelisted/regex-validated before it touches a query.
v3: chain steps now carry DIRECTION ("in:SUPPLIES" = the other company supplies the current one), because
"the supplier of X" and "the customer of X" are different questions."""
import re
from functools import lru_cache
from . import config, db, llm
from .resolve import EntityResolver
from .schema import REL_TYPES, REL_UNION

_RET = ("UNWIND relationships(p) AS r RETURN DISTINCT startNode(r).name AS s, type(r) AS rel, "
        "endNode(r).name AS o, r.chunk_ids AS cids, r.evidence AS ev LIMIT 100")
TEMPLATES = {
    "connection": f"MATCH p = allShortestPaths((a:Entity {{id:$a}})-[:{REL_UNION}*..4]-(b:Entity {{id:$b}})) {_RET}",
    "neighborhood": f"MATCH p = (a:Entity {{id:$a}})-[:{REL_UNION}*1..__DEPTH__]-(x:Entity) {_RET}",
    "aggregate": "MATCH p = (a:Entity {id:$a})-[:__REL__]-(b:Entity) " + _RET,
    "top_by_rel": ("MATCH (a:Entity)-[r:__REL__]->(b:Entity) RETURN b.name AS name, count(DISTINCT a) AS n, "
                   "collect(DISTINCT a.name) AS who, collect(r.chunk_ids) AS cids ORDER BY n DESC LIMIT 3"),
    "chain": None,  # built per question from validated steps
}
STEP_RE = re.compile(r"^(in|out|any):(" + "|".join(REL_TYPES) + r")$")
_HINTS = "; ".join(f"{k}: {v[1]}" for k, v in REL_TYPES.items())

PLAN_PROMPT = f"""Convert the question into parameters for a fixed query library. Return ONLY JSON:
{{"template": str, "entities": [str], "rel_type": str|null, "depth": int, "steps": [str]}}
Templates:
- chain: ONE starting entity, then follow relationships hop by hop, moving outward from the entity named in the
  question. `steps` is an ordered list; each step is "in:REL", "out:REL" or "any:REL":
    out:REL = the current entity is the SOURCE of REL   (current -[REL]-> next)
    in:REL  = the current entity is the TARGET of REL   (next -[REL]-> current)
    any:REL = direction does not matter (PARTNERS_WITH, COMPETES_WITH)
  Edge direction is source -> target. Direction hints: {_HINTS}
  Examples:
    "the supplier of X" (someone who SUPPLIES X)        -> "in:SUPPLIES"
    "the companies X supplies"                          -> "out:SUPPLIES"
    "the company that acquired S"                       -> "in:ACQUIRED"
    "the CEO of C"                                      -> "in:CEO_OF"
    "the founder of C"                                  -> "out:FOUNDED_BY"
    "the city where C is headquartered"                 -> "out:HEADQUARTERED_IN"
    "the partner of X"                                  -> "any:PARTNERS_WITH"
    "CEO of the company that acquired the supplier of the partner of X"
        -> ["any:PARTNERS_WITH", "in:SUPPLIES", "in:ACQUIRED", "in:CEO_OF"]
  Max 5 steps.
- connection: how two entities relate (2 entities)
- neighborhood: general facts around ONE entity (depth 1-3); use only when no clear chain exists
- aggregate: everything linked to ONE entity via one rel_type (e.g. "which companies does X supply")
- top_by_rel: which entity has the most edges of rel_type (no entities)
rel_type must be one of {list(REL_TYPES)} or null. Copy entity names as written in the question."""


@lru_cache(maxsize=1)
def resolver() -> EntityResolver:
    r = EntityResolver()
    r.load_from_graph()
    return r


def _fact(s, rel, o, cids, ev) -> dict:
    return {"text": f"{s} {REL_TYPES[rel][0]} {o}.", "chunk_ids": cids or [], "evidence": ev or ""}


def _clean_steps(raw) -> list[str]:
    out = []
    for st in raw or []:
        if isinstance(st, str):
            st = st.strip()
            st = f"any:{st}" if st in REL_TYPES else st  # bare rel type = direction unknown
            if STEP_RE.match(st):
                out.append(st)
    return out[:5]


def _chain_pattern(steps: list[str], ignore_direction: bool = False) -> str:
    pat = "(a:Entity {id:$a})"
    for st in steps:  # steps were validated by STEP_RE, so interpolation is safe
        d, rel = st.split(":")
        d = "any" if ignore_direction else d
        pat += {"in": f"<-[:{rel}]-", "out": f"-[:{rel}]->", "any": f"-[:{rel}]-"}[d] + "(:Entity)"
    return pat


def retrieve(question: str) -> dict:
    plan = llm.chat_json([{"role": "system", "content": PLAN_PROMPT}, {"role": "user", "content": question}])
    tpl = plan.get("template") if plan.get("template") in TEMPLATES else "neighborhood"
    rel = plan.get("rel_type") if plan.get("rel_type") in REL_TYPES else None
    steps = _clean_steps(plan.get("steps"))
    ids = [i for i in (resolver().find(n) for n in plan.get("entities") or []) if i]
    if tpl == "chain" and not steps:
        tpl = "neighborhood"
    if tpl == "connection" and len(ids) < 2:
        tpl = "neighborhood"
    if tpl == "aggregate" and not rel:
        tpl = "neighborhood"

    facts, seen = [], set()

    def add(rows):  # dedupe: the same edge can come back from several paths
        for row in rows:
            key = (row["s"], row["rel"], row["o"])
            if key not in seen:
                seen.add(key)
                facts.append(_fact(row["s"], row["rel"], row["o"], row["cids"], row["ev"]))

    if tpl == "top_by_rel":
        if rel:
            for row in db.run_cypher(TEMPLATES[tpl].replace("__REL__", rel)):
                cids = sorted({c for lst in row["cids"] for c in (lst or [])})
                facts.append({"text": f"{row['name']} has {row['n']} incoming {rel} relationships, from: "
                                      f"{', '.join(row['who'])}.", "chunk_ids": cids, "evidence": ""})
    elif ids:
        if tpl == "chain":
            add(db.run_cypher(f"MATCH p = {_chain_pattern(steps)} {_RET}", a=ids[0]))
            if not facts:  # planner got a direction wrong: retry ignoring direction before giving up
                add(db.run_cypher(f"MATCH p = {_chain_pattern(steps, True)} {_RET}", a=ids[0]))
        elif tpl == "connection":
            add(db.run_cypher(TEMPLATES[tpl], a=ids[0], b=ids[1]))
        else:
            add(db.run_cypher(TEMPLATES[tpl].replace("__DEPTH__", "3").replace("__REL__", rel or ""), a=ids[0]))
        if not facts or (len(facts) < 5 and tpl not in ("neighborhood", "chain")):  # too little: widen
            for i in ids[:3]:
                add(db.run_cypher(TEMPLATES["neighborhood"].replace("__DEPTH__", "3"), a=i))
    return {"plan": {"template": tpl, "rel_type": rel, "steps": steps, "entity_ids": ids}, "facts": facts}
