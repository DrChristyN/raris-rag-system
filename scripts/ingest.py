import sys

import yaml

from pathlib import Path


sys.path.insert(

    0,

    str(Path(__file__).parent.parent / "src")

)

from ingestion.pdf_loader import PDFLoader
from ingestion.preprocessor import Preprocessor


def load_config(

    config_path: str = "configs/config.yaml"

) -> dict:


    config_path = Path(config_path)

    if not config_path.exists():

        print(f"ERROR: Config file not found: {config_path}")

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


def print_header() -> None:

    print("\n" + "=" * 55)
    print("  RAG Research System — Document Ingestion")
    print("=" * 55)


def print_step(number: int, description: str) -> None:

    print(f"\n[Step {number}] {description}")
    print("-" * 45)

def print_summary(

    documents: list,
    clean_pages: list,
    pending_pages: list

) -> None:

    print("\n" + "=" * 55)
    print("  Ingestion Complete — Summary")
    print("=" * 55)
    print(f"\n  Documents processed : {len(documents)}")
    print(f"  Clean pages         : {len(clean_pages)}")
    print(f"  Pending pages       : {len(pending_pages)}")

    # Per-document breakdown

    print("\n  Per-document breakdown:")

    for doc in documents:

        # Count clean pages from this document

        doc_clean = [

            p for p in clean_pages

            if p["source_file"] == doc.filename

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

            f"{len(doc_clean):>3} pages "

            f"{pending_str}"

        )

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

                f"    Image pages needing OCR    : "

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

    if pending_pages:

        print(

            f"    data/processed/pending_pages.json"

        )

    # Next step instructions

    print(f"\n  Next steps:")

    if pending_pages:

        print(

            "    1. Review pending_pages.json"

        )

        print(

            "    2. Run ocr_handler.py for image pages"

        )

        print(

            "    3. Run table_extractor.py for tables"

        )

        print(

            "    4. Run page_assembler.py to merge all"

        )

        print(

            "    5. Then proceed to chunking"

        )

    else:

        print(

            "    Proceed to chunking — "

            "python scripts/chunk.py"

        )

    print("=" * 55 + "\n")


def main():

    print_header()

    # ── Load configuration ────────────────────────────────────

    print("\nLoading configuration...")

    config      = load_config()

    raw_dir     = config['paths']['raw_data']

    processed_dir = config['paths']['processed_data']

    print(f"  Raw data    : {raw_dir}")

    print(f"  Output      : {processed_dir}")

    print(

        f"  LLM provider: {config['llm']['provider']}"

    )

    print(

        f"  Embedding   : "

        f"{config['embedding']['model_name']}"

    )

    # ── Check PDFs exist ──────────────────────────────────────

    pdf_files = check_raw_data(raw_dir)

    print(f"\nFound {len(pdf_files)} PDF(s) in {raw_dir}:")

    for f in pdf_files:

        size_mb = f.stat().st_size / 1024 / 1024

        print(f"  {f.name:<45} {size_mb:.1f} MB")

    # ── Step 1 — Load PDFs ────────────────────────────────────

    print_step(1, "Loading PDFs")

    loader    = PDFLoader(config)

    documents = loader.load_directory(raw_dir)

    if not documents:

        print("ERROR: No documents loaded. Check your PDFs.")

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

    # Save pending pages if any exist

    if preprocessor.pending_pages:

        preprocessor.save_pending(processed_dir)

    # ── Step 3 — Print alert summary ──────────────────────────

    print_step(3, "Alert summary")

    preprocessor.print_alert_summary()

    # ── Final summary ─────────────────────────────────────────

    print_summary(

        documents,

        clean_pages,

        preprocessor.pending_pages

    )

if __name__ == "__main__":

    main()
 