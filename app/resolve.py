"""Phase 1b: entity resolution. 'Acme Corp', 'Acme Corporation', 'ACME' must become ONE node.
Strategy (cheap -> expensive): exact match on normalized name/alias -> unambiguous single-token prefix
('Norda' vs 'Norda Semiconductor') -> embedding similarity above a tuned threshold -> new entity."""
import re
import numpy as np
from . import config, db, llm

SUFFIXES = {"incorporated", "inc", "corporation", "corp", "company", "co", "limited", "ltd", "llc", "plc", "gmbh"}


def normalize(name: str) -> str:
    s = re.sub(r"[^\w\s]", " ", name.lower())
    return " ".join(t for t in s.split() if t not in SUFFIXES)


class EntityResolver:
    def __init__(self):
        self.entities: dict[str, dict] = {}        # id -> {id, name, type, aliases:set}
        self.exact: dict[tuple, str] = {}          # (type, normalized alias) -> id
        self._vecs: dict[str, np.ndarray] = {}

    def load_from_graph(self):
        """Rebuild state from Neo4j so re-running ingestion keeps resolving to the same nodes."""
        q = "MATCH (e:Entity) RETURN e.id AS id, e.name AS name, e.type AS type, e.aliases AS aliases"
        for r in db.run_cypher(q):
            aliases = set(r["aliases"] or [r["name"]])
            self.entities[r["id"]] = {"id": r["id"], "name": r["name"], "type": r["type"], "aliases": aliases}
            for a in aliases:
                self.exact[(r["type"], normalize(a))] = r["id"]

    def _vec(self, norm: str) -> np.ndarray:
        if norm not in self._vecs:
            self._vecs[norm] = llm.embed([norm])[0]
        return self._vecs[norm]

    def _nearest(self, norm: str, etype: str | None = None):
        cands = [(k[1], eid) for k, eid in self.exact.items() if etype is None or k[0] == etype]
        if not cands or not norm:
            return None
        toks = norm.split()
        if etype != "Person":  # heuristic: 'norda' <-> 'norda semiconductor', only if unambiguous
            related = {eid for n, eid in cands
                       if (len(toks) == 1 and n.split()[:1] == toks) or (len(n.split()) == 1 and toks[:1] == n.split())}
            if len(related) == 1:
                return related.pop(), 1.0
        mat = np.stack([self._vec(n) for n, _ in cands])
        sims = mat @ self._vec(norm)
        i = int(sims.argmax())
        return cands[i][1], float(sims[i])

    def resolve(self, name: str, etype: str) -> str | None:
        """Write-time: find or create the canonical entity."""
        n = normalize(name)
        if not n:
            return None
        eid = self.exact.get((etype, n))
        if not eid:
            hit = self._nearest(n, etype)
            if hit and hit[1] >= config.ER_THRESHOLD:
                eid = hit[0]
            else:
                eid = f"{etype}:{n.replace(' ', '_')}"
                self.entities[eid] = {"id": eid, "name": name, "type": etype, "aliases": set()}
        ent = self.entities[eid]
        ent["aliases"].add(name)
        if len(name) > len(ent["name"]) and name.upper() != name:
            ent["name"] = name  # canonical display name = longest non-SHOUTING alias
        self.exact[(etype, n)] = eid
        return eid

    def find(self, name: str) -> str | None:
        """Query-time: link a name from the user's question to a node (no creation, any type)."""
        n = normalize(name)
        for (_, k), eid in self.exact.items():
            if k == n:
                return eid
        hit = self._nearest(n)
        return hit[0] if hit and hit[1] >= config.LINK_THRESHOLD else None

    def row(self, eid: str) -> dict:
        e = self.entities[eid]
        return {"id": e["id"], "name": e["name"], "type": e["type"], "aliases": sorted(e["aliases"])}
