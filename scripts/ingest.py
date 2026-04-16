import sys
import json
import yaml
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).parent.parent / "src")
)

from ingestion.pdf_loader   import PDFLoader
from ingestion.preprocessor import Preprocessor
from chunking.base_chunker  import get_chunker

def load_config(

    config_path: str = "configs/config.yaml"

) -> dict:

    config_path = Path(config_path)
    if not config_path.exists():

        print(
            f"ERROR: Config file not found: {config_path}"
        )

        print(
            "Make sure you run this script from "
            "C:\\projects\\raris"
        )

        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_raw_data(raw_dir: str) -> list:

    raw_path  = Path(raw_dir)
    pdf_files = list(raw_path.glob("*.pdf"))
    if not pdf_files:

        print(f"\nERROR: No PDF files found in {raw_dir}")
        print(

            "Copy your papers and thesis into "

            "data/raw/ before running this script."

        )

        sys.exit(1)
    return pdf_files


def save_chunks(
    chunks: list,
    output_dir: str

) -> str:
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "chunks.json"
    chunks_data = [

        {

            "text":        c.text,

            "chunk_index": c.chunk_index,

            "source_file": c.source_file,

            "page_number": c.page_number,

            "doc_type":    c.doc_type,

            "strategy":    c.strategy,

            "chunk_size":  c.chunk_size,

            "parent_id":   c.parent_id,

            "metadata":    c.metadata

        }

        for c in chunks
    ]

    with open(
        output_path, "w", encoding="utf-8"

    ) as f:
        json.dump(
            chunks_data, f,
            indent=2,
            ensure_ascii=False
        )

    print(f"\n  Saved {len(chunks)} chunks to {output_path}")
    return str(output_path)


def print_header() -> None:
    """Print a clear header so terminal output is readable."""
    print("\n" + "=" * 55)
    print("  RAG Research System — Document Ingestion")
    print("=" * 55)


def print_step(
    number: str,
    description: str

) -> None:

    print(f"\n[Step {number}] {description}")
    print("-" * 45)


def print_summary(

    documents:     list,

    clean_pages:   list,

    chunks:        list,

    pending_pages: list

) -> None:

    print("\n" + "=" * 55)
    print("  Ingestion Complete — Summary")
    print("=" * 55)
    print(f"\n  Documents processed : {len(documents)}")
    print(f"  Clean pages         : {len(clean_pages)}")
    print(f"  Chunks created      : {len(chunks)}")
    print(f"  Pending pages       : {len(pending_pages)}")

    # Per-document breakdown

    print("\n  Per-document breakdown:")
    for doc in documents:
        # Count clean pages from this document
        doc_clean = [
            p for p in clean_pages
            if p["source_file"] == doc.filename
        ]

        # Count chunks from this document
        doc_chunks = [
            c for c in chunks
            if c.source_file == doc.filename
        ]

        # Count pending pages from this document

        doc_pending = [
            p for p in pending_pages
            if p["source_file"] == doc.filename
        ]

        pending_str = (
            f"  ({len(doc_pending)} pending)"
            if doc_pending else ""
        )

        print(

            f"    {doc.filename:<40} "
            f"{len(doc_clean):>3} pages  "
            f"{len(doc_chunks):>4} chunks"
            f"{pending_str}"

        )

    # Chunk statistics

    if chunks:

        sizes   = [c.chunk_size for c in chunks]
        avg     = round(sum(sizes) / len(sizes))
        min_s   = min(sizes)
        max_s   = max(sizes)
        strategy = chunks[0].strategy
        print(f"\n  Chunk statistics:")
        print(f"    Strategy  : {strategy}")
        print(f"    Avg size  : {avg} chars")
        print(f"    Min size  : {min_s} chars")
        print(f"    Max size  : {max_s} chars")

    # Show pending breakdown if any

    if pending_pages:

        image_pending = [
            p for p in pending_pages
            if p["pending_reason"] == "image"
        ]

        table_pending = [
            p for p in pending_pages
            if p["pending_reason"] == "table"
        ]

        print(f"\n  Pending breakdown:")

        if image_pending:
            print(
                f"    Image pages needing OCR       : "
                f"{len(image_pending)}"
            )

        if table_pending:
            print(
                f"    Table pages needing pdfplumber: "
                f"{len(table_pending)}"
            )

    # Output files created

    print(f"\n  Output files created:")
    print(
        f"    data/processed/extracted_documents.json"
    )

    print(
        f"    data/processed/cleaned_pages.json"
    )

    print(
        f"    data/processed/chunks.json"
    )

    if pending_pages:
        print(
            f"    data/processed/pending_pages.json"
        )

    # Next step instructions

    print(f"\n  Next steps:")

    if pending_pages:
        print("    1. Review pending_pages.json")
        print("    2. Run ocr_handler.py for image pages")
        print("    3. Run table_extractor.py for tables")
        print("    4. Run page_assembler.py to merge all")
        print("    5. Then proceed to embedding")

    else:

        print(
            "    Proceed to embedding — "
            "python scripts/embed.py"
        )

    print("=" * 55 + "\n")


def main():

    print_header()

    # ── Load configuration ────────────────────────────────────

    print("\nLoading configuration...")

    config        = load_config()

    raw_dir       = config['paths']['raw_data']

    processed_dir = config['paths']['processed_data']

    strategy      = config['chunking']['strategy']

    print(f"  Raw data       : {raw_dir}")

    print(f"  Output         : {processed_dir}")

    print(f"  Chunking       : {strategy}")

    print(f"  LLM provider   : {config['llm']['provider']}")

    print(

        f"  Embedding      : "

        f"{config['embedding']['model_name']}"

    )

    # ── Check PDFs exist ──────────────────────────────────────

    pdf_files = check_raw_data(raw_dir)

    print(

        f"\nFound {len(pdf_files)} PDF(s) "

        f"in {raw_dir}:"

    )

    for f in pdf_files:

        size_mb = f.stat().st_size / 1024 / 1024

        print(f"  {f.name:<45} {size_mb:.1f} MB")

    # ── Step 1 — Load PDFs ────────────────────────────────────

    print_step(1, "Loading PDFs")

    loader    = PDFLoader(config)

    documents = loader.load_directory(raw_dir)

    if not documents:

        print("ERROR: No documents loaded.")

        sys.exit(1)

    loader.save_processed(documents, processed_dir)

    # ── Step 2 — Clean and preprocess ─────────────────────────

    print_step(

        2,

        "Cleaning pages — layout detection + content cleaning"

    )

    preprocessor = Preprocessor(config)

    clean_pages  = preprocessor.clean_documents(documents)

    preprocessor.save_cleaned(clean_pages, processed_dir)

    preprocessor.save_alerts(processed_dir)

    if preprocessor.pending_pages:

        preprocessor.save_pending(processed_dir)

    # ── Step 2b — Chunk the clean pages ───────────────────────

    print_step(

        "2b",

        f"Chunking pages — strategy: {strategy}"

    )

    chunker = get_chunker(strategy, config)

    chunks  = chunker.chunk(clean_pages)

    save_chunks(chunks, processed_dir)

    # ── Step 3 — Print alert summary ──────────────────────────

    print_step(3, "Alert summary")

    preprocessor.print_alert_summary()

    # ── Final summary ─────────────────────────────────────────

    print_summary(

        documents,

        clean_pages,

        chunks,

        preprocessor.pending_pages

    )


# ── Entry point ───────────────────────────────────────────────

# main() only runs when script is executed directly.

# Not when imported by another module.

if __name__ == "__main__":

    main()
 