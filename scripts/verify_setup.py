"""

Run this to confirm every dependency installed correctly.

Usage: python scripts/verify_setup.py

"""

import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

results = []

def check(label, fn):

    try:

        val = fn()

        results.append((True, label, str(val)[:50]))

    except Exception as e:

        results.append((False, label, str(e)[:80]))

# ── Library checks ────────────────────────────────────────────

check("Python version",

      lambda: sys.version.split()[0])

check("langchain",

      lambda: __import__("langchain").__version__)

check("langchain-community",

      lambda: __import__("langchain_community").__version__)

check("llama-index",

      lambda: __import__("llama_index").__version__)

check("sentence-transformers",

      lambda: __import__("sentence_transformers").__version__)

check("torch",

      lambda: __import__("torch").__version__)

check("faiss-cpu",

      lambda: __import__("faiss").IndexFlatL2(4) and "OK")

check("chromadb",

      lambda: __import__("chromadb").__version__)

check("pymupdf",

      lambda: __import__("fitz").version)

check("pypdf",

      lambda: __import__("pypdf").__version__)

check("rank-bm25",

      lambda: __import__("rank_bm25").BM25Okapi([]) and "OK")

check("ragas",

      lambda: __import__("ragas").__version__)

check("ollama",

      lambda: __import__("ollama").__version__)

check("pyyaml",

      lambda: __import__("yaml").__version__)

check("python-dotenv",

      lambda: __import__("dotenv").__version__)

check("tqdm",

      lambda: __import__("tqdm").__version__)

check("pandas",

      lambda: __import__("pandas").__version__)

check("numpy",

      lambda: __import__("numpy").__version__)

check("jupyter",

      lambda: __import__("jupyter").__version__)

# ── Config check ──────────────────────────────────────────────

def check_config():

    import yaml

    with open("configs/config.yaml", encoding="utf-8") as f:

        cfg = yaml.safe_load(f)

    return (f"domain={cfg['project']['domain']} | "

            f"llm={cfg['llm']['provider']} | "

            f"embed={cfg['embedding']['model_name'].split('/')[-1]}")

check("config.yaml", check_config)

# ── .env check ────────────────────────────────────────────────

check(".env exists",

      lambda: "yes" if Path(".env").exists()

      else (_ for _ in ()).throw(FileNotFoundError("missing")))

# ── Folder structure check ────────────────────────────────────

folders = [

    "data/raw", "data/processed",

    "artifacts/faiss_index", "artifacts/embedding_cache",

    "src/ingestion", "src/chunking", "src/embedding",

    "src/vectorstore", "src/retrieval", "src/reranking",

    "src/llm", "src/evaluation",

    "frameworks/langchain_rag", "frameworks/llamaindex_rag",

    "configs", "notebooks", "scripts", "tests"

]

check("folder structure",

      lambda: f"all {len(folders)} folders present"

      if all(Path(f).exists() for f in folders)

      else (_ for _ in ()).throw(

          FileNotFoundError(

              next(f for f in folders if not Path(f).exists()))))

# ── Print results ─────────────────────────────────────────────

print("\n" + "=" * 60)

print("  RAG Research System — Setup Verification")

print("=" * 60)

for ok, label, detail in results:

    icon = "+" if ok else "!"

    print(f"  [{icon}] {label:<30} {detail if ok else ''}")

    if not ok:

        print(f"        ERR: {detail}")

passed = sum(1 for ok, *_ in results if ok)

total  = len(results)

print("=" * 60)

print(f"  {passed}/{total} checks passed")

if passed == total:

    print("  All systems go. Ready to build.")

elif passed >= total - 2:

    print("  Almost ready. Fix the items marked [!]")

else:

    print("  Run: pip install -r requirements.txt")

print("=" * 60 + "\n")
 