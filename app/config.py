"""Central config. Everything tunable lives here or in .env."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR") or ROOT / "data" / "docs")   # DATA_DIR=data/synthetic to switch corpus
CACHE_DIR = ROOT / ".cache"
LOG_DIR = ROOT / "logs"
RESULTS_DIR = ROOT / "results"
for _d in (CACHE_DIR, LOG_DIR, RESULTS_DIR):
    _d.mkdir(exist_ok=True)

env = os.getenv
LLM_BASE_URL = env("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = env("LLM_API_KEY", "")
LLM_MODEL = env("LLM_MODEL", "openai/gpt-oss-120b")
ROUTER_MODEL = env("ROUTER_MODEL", "openai/gpt-oss-20b")
EMBED_MODEL = env("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = int(env("EMBED_DIM", "384"))

NEO4J_URI = env("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = env("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = env("NEO4J_PASSWORD", "kgrag-password")
PG_DSN = env("PG_DSN", "postgresql://kgrag:kgrag@localhost:5433/kgrag")

MIN_CONFIDENCE = float(env("MIN_CONFIDENCE", "0.6"))   # drop low-confidence extracted edges
ER_THRESHOLD = float(env("ER_THRESHOLD", "0.88"))      # entity-resolution merge threshold (write time)
LINK_THRESHOLD = 0.75                                   # looser threshold for linking question entities
ROUTER_MIN_CONF = 0.6                                   # below this, run both retrieval paths
TOP_K = 5
EF_SEARCH = 40                                          # HNSW query-time recall/speed knob
