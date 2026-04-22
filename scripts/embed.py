"""

Embedding Script — RAG Research System

========================================

Embeds all chunks and saves to FAISS vector database.

What this script does:

  Step 1 — Load chunks from data/processed/chunks.json

  Step 2 — Load BGE-M3 embedding model into RAM

  Step 3 — Embed all chunk texts → numpy array (608, 1024)

  Step 4 — Build FAISS IndexFlatL2

  Step 5 — Save index + metadata to artifacts/faiss_index/

After this script two files exist:

  artifacts/faiss_index/faiss_index.bin

    Binary file containing 608 vectors of 1024 numbers each.

    Not human-readable. Used by FAISS for similarity search.

  artifacts/faiss_index/faiss_metadata.json

    Maps every vector position → chunk text + source info.

    Human-readable JSON. Used to retrieve actual text

    after FAISS returns matching positions.

Why two separate files?

  FAISS only stores numbers — no text, no metadata.

  When FAISS returns "position 47 is most similar",

  we need faiss_metadata.json to look up:

    What text is at position 47?

    Which paper did it come from?

    Which page? Which parent chunk?

Usage:

  python scripts/embed.py

Run from C:\\projects\\raris with venv activated.

Run ingest.py first to create chunks.json.

"""

import sys

import json

import yaml

import numpy as np

import faiss

from pathlib import Path

sys.path.insert(

    0,

    str(Path(__file__).parent.parent / "src")

)

from embedding.base_embedder import get_embedder


def load_config(

    path: str = "configs/config.yaml"

) -> dict:

    """Load config.yaml"""

    with open(path, "r", encoding="utf-8") as f:

        return yaml.safe_load(f)


def load_chunks(chunks_path: str) -> list:

    """

    Load chunks from JSON file.

    Args:

        chunks_path: path to chunks.json

    Returns:

        list of chunk dictionaries

    """

    with open(

        chunks_path, "r", encoding="utf-8"

    ) as f:

        chunks = json.load(f)

    print(f"  Loaded {len(chunks)} chunks")

    return chunks


def build_faiss_index(

    embeddings: np.ndarray,

    embedding_dim: int

) -> faiss.IndexFlatL2:

    """

    Build FAISS index from embedding vectors.

    IndexFlatL2 = exact L2 distance search.

    Compares query vector against every stored vector.

    Returns positions of closest matches.

    L2 distance = Euclidean distance = straight-line

    distance between two points in 1024-dim space.

    Smaller distance = more similar meaning.

    For 608 chunks: search takes microseconds.

    For 10 million chunks: would need IndexIVFFlat.

    Args:

        embeddings:    shape (n_chunks, embed_dim)

        embedding_dim: 1024 for BGE-M3

    Returns:

        FAISS index with all 608 vectors added

    """

    index = faiss.IndexFlatL2(embedding_dim)

    # FAISS requires float32 — convert from float64

    index.add(embeddings.astype(np.float32))

    print(

        f"  FAISS index built: "

        f"{index.ntotal} vectors, "

        f"{embedding_dim} dimensions"

    )

    return index


def save_faiss_index(

    index: faiss.IndexFlatL2,

    chunks: list,

    output_dir: str

) -> None:

    """

    Save FAISS index and metadata to disk.

    faiss_index.bin:

      Binary format — FAISS native format.

      Contains all vectors as float32 arrays.

      Loaded instantly by faiss.read_index().

    faiss_metadata.json:

      One entry per vector position.

      Maps position → text, source_file, page_number,

      parent_id, doc_type, metadata.

      This is what the retriever returns to the user.

    Args:

        index:      FAISS index with all vectors

        chunks:     list of chunk dicts (same order)

        output_dir: folder to save files

    """

    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save binary FAISS index

    index_path = output_dir / "faiss_index.bin"

    faiss.write_index(index, str(index_path))

    size_mb = index_path.stat().st_size / 1024 / 1024

    print(

        f"  Saved: {index_path} "

        f"({size_mb:.1f} MB)"

    )

    # Save metadata — parallel to vector positions

    # metadata[i] corresponds to vectors[i] in FAISS

    # This parallel structure is critical —

    # never reorder one without reordering the other

    metadata = []

    for i, chunk in enumerate(chunks):

        metadata.append({

            "vector_id":   i,

            "chunk_index": chunk.get("chunk_index", i),

            "text":        chunk["text"],

            "source_file": chunk["source_file"],

            "page_number": chunk["page_number"],

            "doc_type":    chunk["doc_type"],

            "strategy":    chunk["strategy"],

            "chunk_size":  chunk["chunk_size"],

            "parent_id":   chunk.get("parent_id"),

            "metadata":    chunk.get("metadata", {})

        })

    metadata_path = output_dir / "faiss_metadata.json"

    with open(

        metadata_path, "w", encoding="utf-8"

    ) as f:

        json.dump(

            metadata, f,

            indent=2,

            ensure_ascii=False

        )

    size_kb = metadata_path.stat().st_size / 1024

    print(

        f"  Saved: {metadata_path} "

        f"({size_kb:.0f} KB)"

    )


def print_header() -> None:

    print("\n" + "=" * 55)

    print("  RAG Research System — Embedding")

    print("=" * 55)


def main():

    print_header()

    # ── Load config ───────────────────────────────────────

    print("\nLoading configuration...")

    config      = load_config()

    chunks_path = (

        Path(config['paths']['processed_data'])

        / "chunks.json"

    )

    faiss_dir   = config['paths']['faiss_index']

    model_name  = config['embedding']['model_name']

    strategy    = config['chunking']['strategy']

    print(f"  Chunks      : {chunks_path}")

    print(f"  FAISS dir   : {faiss_dir}")

    print(f"  Model       : {model_name}")

    print(f"  Strategy    : {strategy}")

    # ── Check chunks exist ────────────────────────────────

    if not chunks_path.exists():

        print(

            f"\nERROR: {chunks_path} not found."

            f"\nRun: python scripts/ingest.py first."

        )

        sys.exit(1)

    # ── Step 1 — Load chunks ──────────────────────────────

    print("\n[Step 1] Loading chunks...")

    chunks = load_chunks(str(chunks_path))

    if not chunks:

        print("ERROR: No chunks found in file.")

        sys.exit(1)

    texts = [c["text"] for c in chunks]

    print(f"  {len(texts)} texts ready for embedding")

    # ── Step 2 — Load embedder ────────────────────────────

    print("\n[Step 2] Loading embedding model...")

    embedder  = get_embedder(config)

    embed_dim = embedder.get_embedding_dim()

    print(f"  Embedding dimension: {embed_dim}")

    # ── Step 3 — Embed ────────────────────────────────────

    print(

        f"\n[Step 3] Embedding {len(texts)} chunks..."

    )

    print(

        f"  BGE-M3 on CPU: ~10-15 chunks/second"

    )

    print(

        f"  Expected time: ~50-90 seconds for 608 chunks"

    )

    embeddings = embedder.embed(texts)

    print(f"\n  Shape: {embeddings.shape}")

    print(

        f"  Each chunk → {embed_dim}-dim vector"

    )

    # ── Step 4 — Build FAISS index ────────────────────────

    print("\n[Step 4] Building FAISS index...")

    index = build_faiss_index(embeddings, embed_dim)

    # ── Step 5 — Save ─────────────────────────────────────

    print("\n[Step 5] Saving to disk...")

    save_faiss_index(index, chunks, faiss_dir)

    # ── Summary ───────────────────────────────────────────

    print("\n" + "=" * 55)

    print("  Embedding Complete — Summary")

    print("=" * 55)

    print(f"  Chunks embedded : {len(chunks)}")

    print(f"  Embedding dim   : {embed_dim}")

    print(f"  FAISS vectors   : {index.ntotal}")

    print(f"  Model           : {model_name}")

    print(f"  Strategy        : {strategy}")

    print(f"\n  Files saved:")

    print(

        f"    {faiss_dir}/faiss_index.bin"

    )

    print(

        f"    {faiss_dir}/faiss_metadata.json"

    )

    print(f"\n  Next step:")

    print(f"    python scripts/retrieve.py")

    print("=" * 55 + "\n")


if __name__ == "__main__":

    main()
 