"""Mermaid export of the LangGraph pipeline, made safe for GitHub's renderer.

LangGraph names the nodes `graph` and `__start__`/`__end__`. In Mermaid, `graph` is a reserved keyword and the
`<p>` tags are HTML, so the raw export does not render on GitHub. We rename those identifiers; labels stay readable."""
import re

from .pipeline import PIPELINE


def sanitize(raw: str) -> str:
    text = re.sub(r"^---.*?---\n", "", raw, flags=re.S)                    # drop the YAML front matter
    text = text.replace("<p>__start__</p>", "start").replace("<p>__end__</p>", "end")
    text = text.replace("__start__", "start_node").replace("__end__", "end_node")
    text = re.sub(r"(?m)^\tgraph(?=[(\s])", "\tgraph_search", text)         # node declaration and edge sources
    text = re.sub(r"(-->|-\.->)\s+graph;", r"\1 graph_search;", text)       # edge targets
    return text.strip() + "\n"


def pipeline_mermaid() -> str:
    return sanitize(PIPELINE.get_graph().draw_mermaid())
