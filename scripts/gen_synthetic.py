"""Generate a synthetic corpus with PLANTED multi-hop structure + questions with known answers.
Why: on 4 documents vector RAG can read everything. Here, a 3-4 hop answer is spread across 3-5 separate documents
and the question only names the FIRST company, so similarity search has no way to find the later documents.
Usage: python scripts/gen_synthetic.py [--n 40] [--seed 7]"""
import argparse, json, random
from pathlib import Path

A = "Copper Silent Amber Granite Willow Crimson Northern Velvet Iron Lunar Summit Cobalt".split()
B = "Harbor Ridge Meadow Forge Orchard Beacon Quarry Lantern Delta Canyon Falcon Bridge".split()
FIRST = "Mira Jonas Priya Tomas Elena Kofi Hana Lars Sofia Amir Nadia Felix Ingrid Omar Clara Dev Yuki Marta Idris Lena".split()
LAST = "Okafor Lindqvist Nakamura Petrov Haddad Moreau Castillo Brandt Mensah Kowalski Tanaka Rossi Varga Duarte Novak Iyer Bauer Silva Hansen Adeyemi".split()
CITIES = "Lisbon Osaka Denver Nairobi Gdansk Perth Quito Leeds Austin Hanoi Turin Bergen Tulsa Kyoto Lagos".split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    n, rnd = a.n - a.n % 2, random.Random(a.seed)
    names = rnd.sample([f"{x} {y}" for x in A for y in B], n)
    people = rnd.sample([f"{f} {l}" for f in FIRST for l in LAST], 2 * n)
    ceo = {i: people[i] for i in range(n)}
    founder = {i: people[n + i] for i in range(n)}
    hq = {i: rnd.choice(CITIES) for i in range(n)}
    supplier = {i: rnd.choice([j for j in range(1, n) if j != i]) for i in range(n)}   # supplier[X] supplies X
    acquirer = {j: rnd.randrange(0, j) for j in range(1, n)}                           # acquirer[S] acquired S
    order = list(range(n)); rnd.shuffle(order)
    partner = {}
    for k in range(0, n, 2):
        partner[order[k]], partner[order[k + 1]] = order[k + 1], order[k]
    customers = {i: [x for x in range(n) if supplier[x] == i] for i in range(n)}
    acquired = {i: [x for x in range(1, n) if acquirer[x] == i] for i in range(n)}

    def m(i):  # random surface form, to exercise entity resolution
        return rnd.choice([names[i], f"{names[i]} Inc.", names[i].upper(), f"{names[i]} Corporation"])

    out = Path(__file__).resolve().parent.parent / "data" / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.txt"):
        old.unlink()
    for i in range(n):
        s = [f"{m(i)} is headquartered in {hq[i]}.", f"The chief executive officer of {m(i)} is {ceo[i]}.",
             f"{m(i)} was founded by {founder[i]} in {rnd.randint(1960, 2015)}.",
             f"The company employs about {rnd.randint(200, 9000)} people."]
        s += [f"{m(i)} supplies {m(j)}." for j in customers[i]]
        s += [f"{m(i)} acquired {m(j)} in {rnd.randint(1995, 2024)}." for j in acquired[i]]
        s.append(f"{m(i)} partners with {m(partner[i])}.")
        rnd.shuffle(s)
        (out / f"co{i:02d}.txt").write_text(f"{names[i]} Inc.: Company Profile\n" + " ".join(s) + "\n")

    Q = []

    def add(h, q, kw):
        Q.append({"id": f"s{len(Q) + 1}", "question": q, "hops": h, "keywords": kw})

    P = lambda pool, k: rnd.sample(pool, min(k, len(pool)))
    for i in P(range(n), 5): add(1, f"Who is the CEO of {names[i]}?", [ceo[i]])
    for i in P(range(n), 5): add(1, f"Where is {names[i]} headquartered?", [hq[i]])
    for k, i in enumerate(P(range(n), 12)):
        if k % 2: add(2, f"Who is the CEO of the company that supplies {names[i]}?", [ceo[supplier[i]]])
        else: add(2, f"In which city is the supplier of {names[i]} headquartered?", [hq[supplier[i]]])
    for k, i in enumerate(P(range(n), 12)):
        s = supplier[i]
        if k % 2: add(3, f"In which city is the company that acquired the supplier of {names[i]} headquartered?", [hq[acquirer[s]]])
        else: add(3, f"Who is the CEO of the company that acquired the supplier of {names[i]}?", [ceo[acquirer[s]]])
    for i in P(range(n), 10):
        s = supplier[partner[i]]
        add(4, f"Who is the CEO of the company that acquired the supplier of the partner of {names[i]}?", [ceo[acquirer[s]]])
    for i in P([i for i in range(n) if len(customers[i]) >= 2], 6):
        add("agg", f"Which companies does {names[i]} supply?", [names[j] for j in customers[i]])
    for i in P(range(n), 6): add("oos", f"What was the 2023 revenue of {names[i]}?", [])

    (out / "_truth.json").write_text(json.dumps({"names": names, "supplier": supplier, "acquirer": acquirer,
                                                 "partner": partner, "hq": hq, "ceo": ceo}, indent=1))
    qpath = Path(__file__).resolve().parent.parent / "eval" / "questions_synthetic.json"
    qpath.write_text(json.dumps(Q, indent=2))
    from collections import Counter
    print(f"{n} docs -> {out}\n{len(Q)} questions -> {qpath}\n", dict(Counter(str(q['hops']) for q in Q)))
    print("example 4-hop:", next(q for q in Q if q["hops"] == 4))


if __name__ == "__main__":
    main()
