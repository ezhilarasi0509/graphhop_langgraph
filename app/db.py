"""Connections to both stores. They share ONE key: the chunk id."""
import psycopg
from neo4j import GraphDatabase
from pgvector.psycopg import register_vector
from . import config

_driver = None
_pg = None


def neo():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
                                       notifications_min_severity="OFF")
    return _driver


def run_cypher(query: str, **params) -> list[dict]:
    with neo().session() as s:
        return [r.data() for r in s.run(query, **params)]


def pg():
    global _pg
    if _pg is None:
        _pg = psycopg.connect(config.PG_DSN, autocommit=True)
        register_vector(_pg)
    return _pg


def init_stores():
    with psycopg.connect(config.PG_DSN, autocommit=True) as c:  # extension must exist before register_vector
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        c.execute(f"""CREATE TABLE IF NOT EXISTS chunks (
            id text PRIMARY KEY, doc_id text, section text, text text,
            entity_ids text[] DEFAULT '{{}}', embedding vector({config.EMBED_DIM}))""")
        c.execute("""CREATE INDEX IF NOT EXISTS chunks_hnsw ON chunks
            USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)""")
    run_cypher("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
    run_cypher("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
