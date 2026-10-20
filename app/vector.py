"""Vector retrieval over pgvector (HNSW, cosine)."""
from . import config, db, llm


def search(query: str, k: int = config.TOP_K) -> list[dict]:
    qv = llm.embed([query])[0]
    conn = db.pg()
    conn.execute("SELECT set_config('hnsw.ef_search', %s, false)", (str(config.EF_SEARCH),))  # recall vs speed
    rows = conn.execute(
        "SELECT id, text, 1 - (embedding <=> %s) AS score FROM chunks ORDER BY embedding <=> %s LIMIT %s",
        (qv, qv, k)).fetchall()
    return [{"id": r[0], "text": r[1], "score": float(r[2])} for r in rows]


def get_chunks(ids: list[str]) -> list[dict]:
    if not ids:
        return []
    rows = db.pg().execute("SELECT id, text FROM chunks WHERE id = ANY(%s)", (list(ids),)).fetchall()
    return [{"id": r[0], "text": r[1], "score": None} for r in rows]
