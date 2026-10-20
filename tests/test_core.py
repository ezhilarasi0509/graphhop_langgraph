from app.chunking import chunk_document
from app.graph import _chain_pattern, _clean_steps
from app.llm import _wait_seconds
from app.pipeline import citations, clean
from app.resolve import normalize


def test_normalize_collapses_name_variants():
    assert normalize("HELIOS ENERGY CORP") == normalize("Helios Energy Corporation") == "helios energy"
    assert normalize("Norda Semiconductor Inc.") == "norda semiconductor"


def test_chunk_ids_are_deterministic():
    text = "Title\nOne sentence here. Another sentence there."
    assert [c["id"] for c in chunk_document("doc", text)] == ["doc:0"]
    assert chunk_document("doc", text) == chunk_document("doc", text)


def test_citation_parsing_and_cleaning():
    assert citations("Yes [helios:0, norda:1] and [not a chunk]") == ["helios:0", "norda:1"]
    assert citations(clean("Answer【helios:0】")) == ["helios:0"]


def test_chain_steps_are_validated_before_reaching_cypher():
    assert _clean_steps(["in:SUPPLIES", "PARTNERS_WITH", "bogus", "out:CEO_OF;DROP"]) == ["in:SUPPLIES", "any:PARTNERS_WITH"]
    pat = _chain_pattern(["any:PARTNERS_WITH", "in:SUPPLIES", "in:ACQUIRED", "in:CEO_OF"])
    assert pat == ("(a:Entity {id:$a})-[:PARTNERS_WITH]-(:Entity)<-[:SUPPLIES]-(:Entity)"
                   "<-[:ACQUIRED]-(:Entity)<-[:CEO_OF]-(:Entity)")
    assert "<-" not in _chain_pattern(["in:SUPPLIES"], ignore_direction=True)


def test_wait_time_parsing():
    class E(Exception):
        response = None
    assert _wait_seconds(E("Please try again in 1m23.4s."), 0) == 84.4
    assert _wait_seconds(E("try again in 6h2m1s"), 0) == 21722
    assert _wait_seconds(E("no hint"), 2) == 8
