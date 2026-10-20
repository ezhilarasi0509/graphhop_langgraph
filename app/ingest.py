"""Phases 1+2: documents -> chunks -> (graph in Neo4j) + (vectors in pgvector), sharing chunk ids.
Usage:  python -m app.ingest --dry-run   (budget first!)    python -m app.ingest
        DATA_DIR=data/synthetic python -m app.ingest"""
import argparse
from . import config, db, llm
from .chunking import chunk_document
from .extract import extract
from .resolve import EntityResolver, normalize
from .schema import REL_TYPES

SYMMETRIC = ("PARTNERS_WITH", "COMPETES_WITH")


def load_chunks() -> list[dict]:
    out = []
    for p in sorted(config.DATA_DIR.glob("*.txt")):
        out += chunk_document(p.stem, p.read_text(encoding="utf-8"))
    return out


def retype_products(extracted: list) -> None:
    """A name typed Company anywhere is a Company everywhere (the model sometimes types suppliers as Product)."""
    companies = {normalize(n) for rels in extracted for r in rels
                 for n, t in ((r.source, r.source_type), (r.target, r.target_type)) if t == "Company"}
    for rels in extracted:
        for r in rels:
            if r.source_type == "Product" and normalize(r.source) in companies:
                r.source_type = "Company"
            if r.target_type == "Product" and normalize(r.target) in companies:
                r.target_type = "Company"


def write_graph(session, resolver, chunk, rels) -> set[str]:
    """MERGE (never CREATE): re-running ingestion is idempotent. Every edge keeps its source chunk ids + evidence."""
    touched = set()
    for r in rels:
        s = resolver.resolve(r.source, r.source_type)
        t = resolver.resolve(r.target, r.target_type)
        if not s or not t or s == t:
            continue
        if r.relation in SYMMETRIC and s > t:
            s, t = t, s  # symmetric relations: one canonical direction, so one fact = one edge
        for eid in (s, t):
            session.run("MERGE (e:Entity {id:$id}) SET e.name=$name, e.type=$type, e.aliases=$aliases",
                        **resolver.row(eid))
        assert r.relation in REL_TYPES  # whitelist check before interpolating a type into Cypher
        session.run(f"""MATCH (a:Entity {{id:$s}}), (b:Entity {{id:$t}})
            MERGE (a)-[r:{r.relation}]->(b)
            SET r.confidence = CASE WHEN coalesce(r.confidence, 0) > $conf THEN r.confidence ELSE $conf END,
                r.evidence = coalesce(r.evidence, $ev),
                r.chunk_ids = CASE WHEN $cid IN coalesce(r.chunk_ids, []) THEN r.chunk_ids
                                   ELSE coalesce(r.chunk_ids, []) + [$cid] END""",
                    s=s, t=t, conf=r.confidence, ev=r.evidence, cid=chunk["id"])
        touched |= {s, t}
    session.run("""MERGE (c:Chunk {id:$cid}) SET c.doc_id=$doc
        WITH c UNWIND $eids AS eid MATCH (e:Entity {id:eid}) MERGE (c)-[:MENTIONS]->(e)""",
                cid=chunk["id"], doc=chunk["doc_id"], eids=sorted(touched))
    return touched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="estimate cost only")
    args = ap.parse_args()
    chunks = load_chunks()
    est_tokens = sum(len(c["text"]) for c in chunks) // 4
    print(f"{len(chunks)} chunks from {len(list(config.DATA_DIR.glob('*.txt')))} docs "
          f"| ~{est_tokens:,} input tokens (+~{len(chunks) * 700:,} prompt/output overhead) "
          f"| {len(chunks)} LLM calls on first run (0 on re-runs: cached)")
    if args.dry_run:
        return
    db.init_stores()
    resolver = EntityResolver()
    resolver.load_from_graph()
    vecs = llm.embed([c["text"] for c in chunks])
    extracted = [extract(c["text"]) for c in chunks]  # cached: no API calls on re-runs
    retype_products(extracted)
    for i, (c, v, rels) in enumerate(zip(chunks, vecs, extracted), 1):
        with db.neo().session() as s:
            ents = write_graph(s, resolver, c, rels)
        db.pg().execute(
            """INSERT INTO chunks (id, doc_id, section, text, entity_ids, embedding) VALUES (%s,%s,%s,%s,%s,%s)
               ON CONFLICT (id) DO UPDATE SET text=EXCLUDED.text, entity_ids=EXCLUDED.entity_ids,
               embedding=EXCLUDED.embedding""",
            (c["id"], c["doc_id"], c["section"], c["text"], sorted(ents), v))
        print(f"[{i}/{len(chunks)}] {c['id']}: {len(rels)} relations, {len(ents)} entities")
    n = db.run_cypher("MATCH (e:Entity) RETURN count(e) AS n")[0]["n"]
    m = db.run_cypher("MATCH ()-[r]->() WHERE type(r) <> 'MENTIONS' RETURN count(r) AS n")[0]["n"]
    print(f"Done. Graph: {n} entities, {m} relationships. Inspect at http://localhost:7474")


if __name__ == "__main__":
    main()
