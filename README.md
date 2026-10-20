# GraphHop: Hybrid Knowledge Graph + Vector RAG for Multi-Hop Questions

Python | LangGraph | LangChain | Neo4j | pgvector | FastAPI | gpt-oss-120b (Groq free tier) | local MiniLM embeddings

Vector RAG fails when an answer is spread across several documents. GraphHop builds a knowledge graph from the
same documents, links it to the vector index through shared chunk ids, and routes each question to the right
retrieval path. The answer pipeline is a LangGraph state machine, and every answer carries citations that are
validated against what was actually retrieved.

## Results

Synthetic benchmark: 40 generated company documents, 56 questions with known answers, grouped by hop count
(the number of documents needed to answer). Same questions, same answer model and same embeddings for both systems.

<!-- RESULTS:START -->

| System | 1-hop | 2-hop | 3-hop | 4-hop | agg | oos | Overall | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| Vector-only baseline | 100% (n=10) | 25% (n=12) | 0% (n=12) | 0% (n=10) | 100% (n=6) | 100% (n=6) | 45% | 804 | 1066 |
| Hybrid graph+vector | 100% (n=10) | 100% (n=12) | 100% (n=12) | 90% (n=10) | 100% (n=6) | 100% (n=6) | 98% | 2276 | 3103 |

_Measured on the earlier plain-Python version of the pipeline. Re-run `python -m eval.run_eval` on this
LangGraph version, then `python scripts/update_readme.py`, to refresh this table (it will also gain tokens/query)._

<!-- RESULTS:END -->

- **Vector-only collapses as hops increase.** A diagnostic with no LLM involved (`eval/vector_recall.py`) shows the
  answer text is present in the top-5 retrieved passages for 100% of 1-hop questions and 0% of 2-, 3- and 4-hop
  questions. Retrieval fails before generation even starts.
- **Hybrid holds through 3 hops** and misses one of ten 4-hop questions (see Limitations).
- **The cost of the gain:** hybrid is roughly 2.8x slower at the median, and needs a one-time graph build
  (40 LLM extraction calls, 135 entities, 218 relationships for this corpus).

## Architecture

The answer pipeline, exported from the compiled LangGraph (`docs/pipeline.mmd`, also served at `GET /pipeline`):

```mermaid
graph TD;
	__start__([<p>__start__</p>]):::first
	route(route)
	graph(graph)
	vector(vector)
	merge(merge)
	generate(generate)
	validate(validate)
	__end__([<p>__end__</p>]):::last
	__start__ --> route;
	generate --> validate;
	graph -.-> merge;
	graph -.-> vector;
	merge --> generate;
	route -.-> graph;
	route -.-> vector;
	validate -.-> __end__;
	validate -.-> generate;
	vector --> merge;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

The route node fans out to the graph and vector nodes (in parallel when the router is unsure), `merge` labels the
evidence, `generate` writes a cited answer, and `validate` loops back to `generate` until every citation resolves
to a retrieved chunk. Both stores share one key, the chunk id: a graph edge points back to the exact passage that
justified it.

## The problem

A multi-hop question such as "Who is the CEO of the company that acquired the supplier of the partner of Amber
Harbor?" needs facts from four different documents, but the question only names Amber Harbor. The later documents
are not similar to the question, so similarity search never retrieves them and the answer cannot be assembled.

## How it works

1. **Ontology** (`app/schema.py`): 5 entity types and 11 relation types. A constrained ontology keeps the graph
   queryable; an "extract everything" prompt produces a graph nobody can query.
2. **Extraction** (`app/extract.py`): the LLM returns JSON, validated with pydantic and retried on failure.
   Results are cached by request hash, so re-ingestion costs nothing.
3. **Entity resolution** (`app/resolve.py`): "Silent Harbor", "SILENT HARBOR" and "Silent Harbor Corporation"
   become one node (normalization, unambiguous prefix match, then embedding similarity above a threshold).
4. **Graph retrieval** (`app/graph.py`): the model never writes Cypher. It picks a template and fills parameters
   (entities, direction-aware relationship steps); every value is whitelisted before it reaches a query.
5. **Router** (`app/router.py`): a small model chooses graph or vector. When it is unsure, both paths run in parallel.
6. **Pipeline** (`app/pipeline.py`): the LangGraph state machine above. LangChain provides the model client
   (`ChatOpenAI`, any OpenAI-compatible provider) and the answer prompt template.
7. **Evaluation** (`eval/run_eval.py`): same questions through both systems; reports accuracy by hop count,
   p50/p95 latency, tokens per query and LLM calls per query. Progress is saved so interrupted runs resume.

## Design decisions

- **No LLM-written Cypher.** It is an injection risk and it is not reproducible. A template library with validated
  parameters fixes both.
- **Graph edges keep their source chunk ids and an evidence quote**, so every graph fact is traceable to text.
- **Direction-aware chains.** "The supplier of X" and "the customer of X" are different questions, so each hop
  states its direction (for example `in:SUPPLIES`).
- **LangGraph for the control flow.** The citation-retry loop is a real cycle and `both` is a real parallel
  fan-out; as a state machine both are explicit, unit-tested and traceable.
- **Fallback to both paths** when the router is unsure: an extra retrieval costs less than a wrong route.
- **Idempotent ingestion:** `MERGE` plus deterministic chunk ids, so re-running changes nothing.

## What went wrong, and what fixed it

- **Chain traversal ignored edge direction**, so "supplier of the partner" returned the partner's customer.
  Fixed with direction-aware steps and a fallback that retries without direction.
- **The extractor typed 12 companies as `Product`**, creating duplicate nodes. Fixed by retyping any name that is
  a Company elsewhere in the corpus before writing to the graph.
- **Symmetric relations were stored twice** (A partners with B, B partners with A). Fixed with a canonical direction.
- **The answer model wrote citations with fullwidth brackets**, which the validator rejected. Fixed by normalizing
  them and instructing the model.
- **Free-tier rate limits** interrupted the benchmark. The client waits as long as the provider asks, excludes the
  wait from latency, and the benchmark saves every result so it can resume.

## Limitations

- **Synthetic corpus.** Relationships are planted, so this proves the mechanism, not performance on real documents.
- **Small benchmark.** 56 questions; one question moves a hop group by 8 to 10 points.
- **One known generation failure.** In one 4-hop question the graph returned the complete correct chain, but the
  answer model still replied "insufficient context" in 2 of 3 repeat runs. This is a prompt weakness in the answer
  step, not a retrieval failure.
- **Planner prompt overlap.** The planner prompt contains example phrasings close to the benchmark questions, so
  differently worded questions may score lower. Not yet tested.
- **Latency** was measured on a free tier and excludes time spent waiting on provider rate limits.

## Repository layout

    app/        schema, chunking, extract, resolve, ingest, vector, graph, router, llm (LangChain), pipeline (LangGraph), answer, api
    scripts/    gen_synthetic.py, update_readme.py, export_diagram.py
    eval/       run_eval.py (benchmark), vector_recall.py (retrieval diagnostic), question sets
    tests/      17 tests: fake-LLM-server client tests, helper tests, LangGraph control-flow tests
    data/       sample documents (docs/) and the generated synthetic corpus (synthetic/)
    docs/       HOW_IT_WORKS.md (internals walkthrough), pipeline.mmd (exported diagram)

## Run it yourself

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt -r requirements-dev.txt
    cp .env.example .env          # add a free Groq API key
    docker compose up -d          # Neo4j and Postgres/pgvector
    python -m pytest -q           # 17 tests, no API key or database needed
    python scripts/gen_synthetic.py
    DATA_DIR=data/synthetic python -m app.ingest --dry-run   # estimate cost first
    DATA_DIR=data/synthetic python -m app.ingest
    uvicorn app.api:app --reload  # http://localhost:8000/docs
    python -m eval.run_eval --questions eval/questions_synthetic.json --out results/benchmark_synthetic.md
    python scripts/update_readme.py

Explore the graph in the Neo4j browser at http://localhost:7474 (user `neo4j`, password `kgrag-password`).
For a trace view of every LangGraph step, set `LANGSMITH_TRACING=true` and a `LANGSMITH_API_KEY` in `.env`.

## Roadmap

- Benchmark on a real corpus (SEC 10-K filings)
- Questions written independently of the planner prompt
- Fix the answer-prompt refusal weakness and re-run the benchmark
- Incremental graph updates instead of full re-ingestion
- Community detection for corpus-level summaries and a graph view in the response UI
