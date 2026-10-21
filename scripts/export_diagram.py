"""Write the real pipeline diagram (generated from the compiled LangGraph) to docs/pipeline.mmd.
Usage (from the project root): python scripts/export_diagram.py"""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))   # so `app` is importable when this file is run from scripts/
from app.diagram import pipeline_mermaid  # noqa: E402

mermaid = pipeline_mermaid()
(root / "docs" / "pipeline.mmd").write_text(mermaid)
print(mermaid)
