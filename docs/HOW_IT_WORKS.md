# How it works internally

## The one idea
Two stores, **one shared key: the chunk id** (`co07:0`).
- pgvector: chunk text + embedding + the entity ids it mentions.
- Neo4j: entities as nodes, relationships as edges; every edge stores `chunk_ids` and an `evidence` quote.

From a graph path you can jump to the exact passages (citations). From a passage you know which entities it mentions.

## Ingestion (`app/ingest.py`)
```
data/*.txt -> chunking.py -> chunks {id: doc:n}
  each chunk -> extract.py (LLM, strict JSON) -> pydantic validation (+repair retry, disk-cached by request hash)
             -> retype_products (a name typed Company anywhere is a Company everywhere)
             -> resolve.py (names -> canonical entity ids; aliases stored on the node)
             -> Neo4j MERGE entities/edges (idempotent)  +  pgvector upsert (embedding, entity_ids)
```
- **Constrained ontology** (`schema.py`): 5 entity types, 11 relation types. Invalid output is rejected and retried.
- **Entity resolution** (`resolve.py`): normalize ("Inc/Corp" stripped, lowercase) -> exact alias hit -> unambiguous
  single-token prefix ("Norda" -> "Norda Semiconductor") -> embedding similarity >= `ER_THRESHOLD` -> new entity.
  Too low a threshold merges different companies silently; too high leaves duplicates.
- **Symmetric relations** (`PARTNERS_WITH`, `COMPETES_WITH`) are written in one canonical direction, so one fact is one edge.
- **Idempotency**: `MERGE` + deterministic ids + LLM cache keyed by request hash. Re-running costs nothing.

## Query: a LangGraph state machine (`app/pipeline.py`)
```
START -> route -> graph  --\
              \-> vector --+-> merge -> generate -> validate --invalid--> generate (max 3 attempts)
                                                          \--valid or refused--> END
```
(`docs/pipeline.mmd` holds the diagram exported from the compiled graph; `GET /pipeline` serves it too.)

| Node | What it does |
|---|---|
| route | small model returns `graph` / `vector` + confidence. Low confidence => `both`, and both branches run **in parallel** |
| graph | `graph.py`: the planner picks a Cypher *template* and fills validated parameters; the model never writes Cypher |
| vector | pgvector HNSW top-k. Also the fallback when the graph returns fewer than 3 facts |
| merge | labels GRAPH FACTS and PASSAGES, fetches the text of every chunk a graph fact cites |
| generate | LangChain `ChatPromptTemplate` + `ChatOpenAI` produce the cited answer |
| validate | every `[chunk:id]` must be a retrieved chunk; otherwise the error is fed back and `generate` runs again |

Why LangGraph: the retrieve/generate/validate flow has a real **cycle** (the citation retry) and a real **parallel
fan-out** (`both`). A state machine makes both explicit, testable (`tests/test_pipeline.py`) and traceable
(`trace` in every API response; optional LangSmith tracing). Why LangChain: `ChatOpenAI` as the single model client
(works with any OpenAI-compatible provider) and prompt templates; our own retry loop sits on top so rate-limit
waits are measured and excluded from latency.

## Chain retrieval
For "CEO of the company that acquired the supplier of the partner of X" the planner emits direction-aware steps:
`["any:PARTNERS_WITH", "in:SUPPLIES", "in:ACQUIRED", "in:CEO_OF"]`, which become the Cypher pattern
`(a)-[:PARTNERS_WITH]-()<-[:SUPPLIES]-()<-[:ACQUIRED]-()<-[:CEO_OF]-()`. Every step is checked against a regex
whitelist before it is interpolated. If a direction is wrong and nothing matches, the same chain is retried
ignoring direction.

## Decisions to be able to defend
1. **No LLM-written Cypher**: injection risk + not reproducible. Template library + validated parameters.
2. **Router fallback to "both"** when unsure: an extra retrieval costs less than a wrong route.
3. **Citation validation** is free and removes fabricated sources.
4. **Graph retrieval is slower and costlier to build**; the benchmark reports tokens, calls and latency for both systems.
5. `MENTIONS` edges (Chunk->Entity) are excluded from path queries, otherwise any two entities in one chunk look 1 hop apart.

## Exercises
1. Open Neo4j Browser; check for duplicate nodes; change `ER_THRESHOLD` and compare.
2. Add a node to the LangGraph (for example a reranker between `merge` and `generate`) and re-run the benchmark.
3. Add a template (`common_neighbors`) and teach the planner prompt about it.
4. Write 15 paraphrased questions and compare the score with the generated ones.
5. Run on SEC 10-K filings and look at the accuracy-by-hop curve.
