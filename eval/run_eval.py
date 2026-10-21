"""Phase 5: same questions, vector-only vs hybrid. Writes a markdown table for your README.
Every finished question is saved to results/progress.jsonl, so an interrupted run resumes where it stopped.
Delete that file to start a fresh benchmark.
Usage: python -m eval.run_eval --questions eval/questions_synthetic.json --out results/benchmark_synthetic.md
       [--per-group 5] [--sleep 2]"""
import argparse, json, time
from collections import defaultdict
import numpy as np
from app import answer, config, vector

PROGRESS_FILE = config.RESULTS_DIR / "progress.jsonl"
GROUPS = ["1", "2", "3", "4", "agg", "oos"]


def load_progress() -> dict:
    done = {}
    if PROGRESS_FILE.exists():
        for line in open(PROGRESS_FILE):
            row = json.loads(line)
            done[row.pop("key")] = row   # later lines override earlier ones
    return done


def correct(q, res) -> bool:
    if q["hops"] == "oos":
        return res["refused"]
    a = " ".join(res["answer"].lower().split())
    return (not res["refused"]) and all(k.lower() in a for k in q["keywords"])


def run(mode, questions, sleep, progress) -> list[dict]:
    rows = []
    for q in questions:
        key = f"{mode}:{q['id']}"
        if key in progress:  # finished in an earlier run
            rows.append(progress[key])
            continue
        res = answer.ask(q["question"], mode)
        row = {"hops": str(q["hops"]), "ok": correct(q, res), "ms": res["latency_ms"], "route": res["route"],
               "tokens": res["tokens"], "calls": res["llm_calls"], "rescued": res["rescued"],
               "fallback": res["chain_fallback"]}
        rows.append(row)
        progress[key] = row
        with open(PROGRESS_FILE, "a") as f:
            f.write(json.dumps({"key": key, **row}) + "\n")
        print(f"[{mode}] {q['id']} hops={q['hops']} ok={row['ok']} route={row['route']} "
              f"{row['ms']}ms {row['tokens']} tokens {row['calls']} calls"
              + (f" rescued={row['rescued']}" if row["rescued"] else "") + (" FALLBACK" if row["fallback"] else ""))
        if not row["ok"]:
            print("   FAILED ->", res["answer"][:200].replace("\n", " "), "| facts:", len(res["graph_facts"]),
                  "| expected:", q["keywords"])
        time.sleep(sleep)
    return rows


def build_table(systems: dict) -> str:
    head = ("| System | " + " | ".join(f"{g}-hop" if g.isdigit() else g for g in GROUPS) +
            " | Overall | p50 ms | p95 ms | tokens/query | LLM calls/query | rescued / fallback |")
    lines = [head, "|---" * (len(GROUPS) + 7) + "|"]
    for name, rows in systems.items():
        by = defaultdict(list)
        for r in rows:
            by[r["hops"]].append(r["ok"])
        cells = [f"{100 * np.mean(by[g]):.0f}% (n={len(by[g])})" if by[g] else "-" for g in GROUPS]
        ms = [r["ms"] for r in rows]
        have = all(r.get("tokens") is not None for r in rows)   # averages only when every row has the data
        tok = f"{np.mean([r['tokens'] for r in rows]):,.0f}" if have else "-"
        calls = f"{np.mean([r['calls'] for r in rows]):.1f}" if have else "-"
        resc = sum(1 for r in rows if r.get("rescued")) ; fb = sum(1 for r in rows if r.get("fallback"))
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {100 * np.mean([r['ok'] for r in rows]):.0f}% | "
                     f"{np.percentile(ms, 50):.0f} | {np.percentile(ms, 95):.0f} | {tok} | {calls} | {resc} / {fb} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between questions (free-tier rate limits)")
    ap.add_argument("--questions", default=str(config.ROOT / "eval" / "questions.json"))
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "benchmark.md"))
    ap.add_argument("--per-group", type=int, default=0, help="use only the first N questions per hop group")
    a = ap.parse_args()
    vector.search("warm up", 1)  # keep embedding-model load time out of the latency numbers
    qs = json.load(open(a.questions))
    if a.per_group:
        count, keep = defaultdict(int), []
        for q in qs:
            g = str(q["hops"])
            count[g] += 1
            if count[g] <= a.per_group:
                keep.append(q)
        qs = keep
    progress = load_progress()
    systems = {"Vector-only baseline": run("vector", qs, a.sleep, progress),
               "Hybrid graph+vector": run("auto", qs, a.sleep, progress)}
    table = build_table(systems)
    open(a.out, "w").write(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
