import re
from app.diagram import pipeline_mermaid


def test_diagram_avoids_mermaid_reserved_words_and_html():
    m = pipeline_mermaid()
    assert m.startswith("graph TD;")
    assert "<p>" not in m and "__start__" not in m
    assert not re.search(r"(?m)^\tgraph[(\s]", m) and "graph_search(graph)" in m   # `graph` as a node id breaks Mermaid


def test_every_edge_connects_declared_nodes():
    m = pipeline_mermaid()
    nodes = set(re.findall(r"(?m)^\t(\w+)[(\[]", m))
    edges = re.findall(r"(?m)^\t(\w+) (?:-->|-\.->) (\w+);", m)
    assert {"route", "graph_search", "vector", "merge", "generate", "validate"} <= nodes
    assert edges and all(a in nodes and b in nodes for a, b in edges)
