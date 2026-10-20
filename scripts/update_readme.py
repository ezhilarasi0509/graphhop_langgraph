"""Copy the latest benchmark table into README.md (between the RESULTS markers).
Usage: python scripts/update_readme.py [results/benchmark_synthetic.md]"""
import re, sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "results" / "benchmark_synthetic.md"
if not src.exists():
    sys.exit(f"{src} not found: run the benchmark first (python -m eval.run_eval ...)")
table = src.read_text().strip()
readme = (root / "README.md").read_text()
new = re.sub(r"(<!-- RESULTS:START -->).*?(<!-- RESULTS:END -->)",
             lambda m: f"{m.group(1)}\n\n{table}\n\n{m.group(2)}", readme, flags=re.S)
(root / "README.md").write_text(new)
print("README results table updated" if new != readme else "no change (markers missing or table identical)")
