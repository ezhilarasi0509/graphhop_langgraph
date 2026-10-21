| System | 1-hop | 2-hop | 3-hop | 4-hop | agg | oos | Overall | p50 ms | p95 ms | tokens/query | LLM calls/query | rescued / fallback |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Vector-only baseline | 100% (n=10) | 25% (n=12) | 0% (n=12) | 0% (n=10) | 100% (n=6) | 100% (n=6) | 45% | 840 | 1902 | 877 | 1.0 | 0 / 0 |
| Hybrid graph+vector | 100% (n=10) | 100% (n=12) | 100% (n=12) | 100% (n=10) | 100% (n=6) | 100% (n=6) | 100% | 2226 | 3860 | 2,305 | 2.7 | 0 / 0 |
