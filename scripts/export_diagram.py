"""Write the real pipeline diagram (generated from the compiled LangGraph) to docs/pipeline.mmd.
Usage (from the project root): python scripts/export_diagram.py"""
import re, sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))   # so `app` is importable when this file is run from scripts/
from app.pipeline import PIPELINE  # noqa: E402

mermaid = re.sub(r"^---.*?---\n", "", PIPELINE.get_graph().draw_mermaid(), flags=re.S)
(root / "docs" / "pipeline.mmd").write_text(mermaid)
print(mermaid)
