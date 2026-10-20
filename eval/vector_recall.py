"""Zero-LLM-token diagnostic: does vector retrieval even RETRIEVE the evidence?
For each question with a unique answer (a person or company name), check whether the answer text appears in the
top-5 retrieved passages. No generation step, so no API tokens are used.
Usage: python -m eval.vector_recall [eval/questions_synthetic.json]"""
import json, sys
from collections import defaultdict
from app import vector


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "eval/questions_synthetic.json"
    questions = json.load(open(path))
    vector.search("warm up", 1)
    by = defaultdict(list)
    for q in questions:
        if q["hops"] == "oos" or all(" " not in k for k in q["keywords"]):
            continue  # skip refusals and single-word answers like cities (too easy to match by chance)
        text = " ".join(p["text"] for p in vector.search(q["question"], 5)).lower()
        by[str(q["hops"])].append(all(k.lower() in text for k in q["keywords"]))
    print("Answer present in the top-5 vector passages (unique-answer questions only):")
    for g in ["1", "2", "3", "4", "agg"]:
        if by[g]:
            print(f"  {g}-hop: {100 * sum(by[g]) / len(by[g]):.0f}%  (n={len(by[g])})")


if __name__ == "__main__":
    main()
