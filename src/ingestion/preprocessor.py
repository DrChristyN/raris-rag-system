import re
import json
import fitz
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

class PageAlert:

    WARNING = "WARNING"
    INFO    = "INFO"
    PENDING = "PENDING"

    def __init__(

        self,
        level: str,
        source_file: str,
        page_number: int,
        reason: str,
        pending_tool: Optional[str] = None

    ):

        self.level        = level
        self.source_file  = source_file
        self.page_number  = page_number
        self.reason       = reason
        self.pending_tool = pending_tool

    def __str__(self) -> str:

        tool_str = (
            f" → queued for {self.pending_tool}"
            if self.pending_tool else ""

        )

        return (

            f"  [{self.level}] "
            f"{self.source_file} "
            f"page {self.page_number}: "
            f"{self.reason}{tool_str}"

        )


class Preprocessor:


    def __init__(self, config: Dict[str, Any]):
        self.config       = config
        self.prep_cfg     = config.get('preprocessing', {})
        self.min_length   = self.prep_cfg.get(
            'min_chunk_length', 100
        )

        self._patterns    = self._compile_patterns()
        self.alerts: List[PageAlert] = []
        self.pending_pages: List[Dict[str, Any]] = []

    def clean_documents(
        self,
        documents: List[Any]
    ) -> List[Dict[str, Any]]:

        self.alerts       = []
        self.pending_pages = []
        all_clean_pages = []

        for doc in documents:

            print(f"\n  Cleaning: {doc.filename}")
            ref_start = self._find_references_start(
                doc.pages
            )

            if ref_start:

                print(
                    f"    References section found at "
                    f"page {ref_start} — will exclude"
                )

            doc_clean_pages = self._clean_document(
                doc, ref_start

            )

            all_clean_pages.extend(doc_clean_pages)

            print(
                f"    Kept {len(doc_clean_pages)} "
                f"normal pages"
            )

        pending_count = len(self.pending_pages)

        if pending_count > 0:
            print(
                f"\n  {pending_count} pages queued for "
                f"specialist processing"
            )

        print(
            f"\n  Total clean pages: {len(all_clean_pages)}"
        )

        return all_clean_pages

    def save_pending(
        self,
        output_dir: str
    ) -> Optional[str]:


        if not self.pending_pages:
            print("  No pending pages to save.")
            return None

        output_dir  = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "pending_pages.json"

        with open(
            output_path, "w", encoding="utf-8"
        ) as f:

            json.dump(
                self.pending_pages, f,
                indent=2,
                ensure_ascii=False

            )

        print(
            f"\n  Saved {len(self.pending_pages)} "
            f"pending pages to {output_path}"
        )

        print(
            f"  Run ocr_handler.py and "
            f"table_extractor.py to process these."
        )

        return str(output_path)

    def save_cleaned(
        self,
        clean_pages: List[Dict],
        output_dir: str

    ) -> str:


        output_dir  = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "cleaned_pages.json"

        with open(
            output_path, "w", encoding="utf-8"

        ) as f:
            json.dump(
                clean_pages, f,
                indent=2,
                ensure_ascii=False
            )

        print(
            f"\n  Saved {len(clean_pages)} clean pages "
            f"to {output_path}"
        )

        return str(output_path)

    def load_cleaned(
        self,
        input_path: str

    ) -> List[Dict[str, Any]]:

        with open(
            input_path, "r", encoding="utf-8"

        ) as f:
            pages = json.load(f)

        print(
            f"  Loaded {len(pages)} pre-cleaned pages "
            f"from cache"
        )

        return pages

    def print_alert_summary(self) -> None:


        if not self.alerts:
            print(
                "\n  No alerts — all pages "
                "processed cleanly."
            )

            return
        
        pending  = [
            a for a in self.alerts
            if a.level == PageAlert.PENDING
        ]

        warnings = [
            a for a in self.alerts
            if a.level == PageAlert.WARNING

        ]

        infos    = [

            a for a in self.alerts
            if a.level == PageAlert.INFO

        ]

        print("\n" + "=" * 50)
        print("  Page Processing Alert Summary")
        print("=" * 50)

        if pending:

            print(
                f"\n  PENDING — queued for "
                f"specialist processing ({len(pending)}):"
            )

            for alert in pending:
                print(str(alert))

        if warnings:
            print(f"\n  WARNINGS ({len(warnings)}):")
            for alert in warnings:
                print(str(alert))

        if infos:
            print(f"\n  INFO ({len(infos)}):")
            for alert in infos:
                print(str(alert))

        print("\n" + "=" * 50)

        print(
            f"  {len(pending)} pending, "
            f"{len(warnings)} warnings, "
            f"{len(infos)} info"

        )

        if pending:

            print(
                "  Run ocr_handler.py then "
                "table_extractor.py then "
                "page_assembler.py to restore "
                "all pending pages."
            )

        print("=" * 50 + "\n")

    def save_alerts(self, output_dir: str) -> str:

        output_dir  = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "alerts.json"
        alert_data = [

            {

                "level":        a.level,
                "source_file":  a.source_file,
                "page_number":  a.page_number,
                "reason":       a.reason,
                "pending_tool": a.pending_tool

            }

            for a in self.alerts
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(alert_data, f, indent=2)
        print(f"  Alerts saved to {output_path}")
        return str(output_path)


    def _clean_document(
        self,
        doc: Any,
        ref_start_page: Optional[int]

    ) -> List[Dict[str, Any]]:

        clean_pages = []

        for page in doc.pages:

            if (

                ref_start_page is not None
                and page.page_number >= ref_start_page
                and self.prep_cfg.get(

                    'remove_references_section', True
                )

            ):
                continue

            result = self._extract_with_detection(
                page.source_path,
                page.page_number - 1,
                page.source_file,
                page.page_number

            )

            page_type  = result["page_type"]
            clean_text = result["text"]

            # ── Handle image-only pages ───────────────────

            if page_type == "image":

                self._queue_pending(
                    page=page,
                    reason="image",
                    tool="pytesseract"
                )

                continue

            # ── Handle table pages ────────────────────────

            if page_type == "table":
                self._queue_pending(
                    page=page,
                    reason="table",
                    tool="pdfplumber"
                )

                if clean_text:

                    clean_text = (
                        "[TABLE — awaiting pdfplumber "
                        "extraction]\n\n"
                        + self._clean_text_content(
                            clean_text
                        )
                    )


            if not clean_text:
                clean_text = self._clean_text_content(
                    page.text
                )

            elif page_type not in ("table",):
                clean_text = self._clean_text_content(
                    clean_text
                )

            if len(clean_text.strip()) < self.min_length:
                if page_type not in ("image", "table"):
                    self.alerts.append(PageAlert(
                        level=PageAlert.INFO,
                        source_file=page.source_file,
                        page_number=page.page_number,

                        reason=(

                            f"Too short after cleaning "
                            f"({len(clean_text.strip())} "
                            f"chars) — skipped"

                        )

                    ))

                continue

            page_dict = {
                "text":        clean_text,
                "source_file": page.source_file,
                "source_path": page.source_path,
                "page_number": page.page_number,
                "doc_type":    page.doc_type,
                "total_pages": page.total_pages,
                "metadata":    page.metadata,
                "page_type":   page_type
            }

            clean_pages.append(page_dict)
        return clean_pages

    def _queue_pending(

        self,
        page: Any,
        reason: str,
        tool: str

    ) -> None:


        self.pending_pages.append({

            "source_file":  page.source_file,
            "source_path":  page.source_path,
            "page_number":  page.page_number,
            "page_index":   page.page_number - 1,
            "doc_type":     page.doc_type,
            "total_pages":  page.total_pages,
            "metadata":     page.metadata,
            "pending_reason": reason,
            "pending_tool": tool,
            "extracted_text": None,
            "processed": False
        })

        reason_text = (

            "Image-only page — no text could be extracted. "
            "Queued for pytesseract OCR."
            if reason == "image"
            else
            "Table-detected page — structure needs pdfplumber. "
            "Flat text saved as placeholder."

        )

        self.alerts.append(PageAlert(

            level=PageAlert.PENDING,
            source_file=page.source_file,
            page_number=page.page_number,
            reason=reason_text,
            pending_tool=tool

        ))

    def _extract_with_detection(

        self,
        filepath: str,
        page_index: int,
        source_file: str,
        page_number: int

    ) -> Dict[str, Any]:

        try:

            pdf_doc    = fitz.open(filepath)
            page       = pdf_doc[page_index]
            blocks     = page.get_text("blocks")
            page_width = page.rect.width
            text_blocks  = [
                b for b in blocks if b[6] == 0
            ]

            image_blocks = [
                b for b in blocks if b[6] == 1
            ]

            if (
                len(text_blocks) == 0
                and len(image_blocks) >= 1
            ):

                pdf_doc.close()

                return {
                    "text":      "",
                    "page_type": "image"
                }

            # ── Empty page ────────────────────────────────

            if len(text_blocks) == 0:
                pdf_doc.close()

                return {
                    "text":      "",
                    "page_type": "empty"
                }

            # ── Table page ────────────────────────────────

            if self._is_table_page(text_blocks):
                text = self._extract_table_as_text(
                    text_blocks
                )

                pdf_doc.close()

                return {
                    "text":      text,
                    "page_type": "table"
                }

            # ── Normal layout detection ───────────────────

            layout = self._detect_layout(
                text_blocks, page_width

            )

            if layout == "double":
                text = self._extract_double_column(
                    text_blocks, page_width
                )

            elif layout == "mixed":
                text = self._extract_mixed_layout(
                    text_blocks, page_width
                )

            else:
                text = self._extract_single_column(
                    text_blocks
                )

            pdf_doc.close()

            return {
                "text":      text,
                "page_type": layout
            }

        except Exception:

            return {
                "text":      "",
                "page_type": "error"
            }

    def _is_table_page(

        self,
        text_blocks: List

    ) -> bool:

        if len(text_blocks) < 10:
            return False
        total_chars = sum(
            len(b[4].strip()) for b in text_blocks
        )

        avg_chars = total_chars / len(text_blocks)

        return (

            len(text_blocks) > 15
            and avg_chars < 20

        )

    def _extract_table_as_text(

        self,
        text_blocks: List

    ) -> str:

        rows: Dict[int, List] = {}
        for block in text_blocks:
            y0   = block[1]
            x0   = block[0]
            text = block[4].strip()

            if not text:
                continue
            matched_row = None

            for row_y in rows:
                if abs(row_y - y0) <= 5:
                    matched_row = row_y
                    break

            if matched_row is None:
                rows[int(y0)] = []
                matched_row   = int(y0)
            rows[matched_row].append((x0, text))
        sorted_row_ys  = sorted(rows.keys())
        result_lines   = []

        for row_y in sorted_row_ys:

            cells = sorted(
                rows[row_y], key=lambda c: c[0]
            )

            row_text = "\t".join(
                text for _, text in cells
            )

            result_lines.append(row_text)

        return "\n".join(result_lines)

    def _detect_layout(

        self,
        text_blocks: List,
        page_width: float

    ) -> str:

        midpoint             = page_width / 2
        tolerance            = 20
        full_width_threshold = page_width * 0.60
        full_width_count = 0
        left_count       = 0
        right_count      = 0

        for block in text_blocks:
            x0   = block[0]
            x1   = block[2]
            text = block[4].strip()

            if not text:
                continue

            block_width  = x1 - x0
            block_center = (x0 + x1) / 2

            if block_width > full_width_threshold:
                full_width_count += 1

            elif block_center < midpoint - tolerance:
                left_count += 1

            elif block_center > midpoint + tolerance:
                right_count += 1

        has_columns    = (
            left_count >= 2 and right_count >= 2
        )

        has_full_width = (full_width_count >= 1)

        if has_columns and has_full_width:
            return "mixed"

        elif has_columns:
            return "double"

        else:
            return "single"

    def _extract_mixed_layout(

        self,
        text_blocks: List,
        page_width: float

    ) -> str:

 
        midpoint             = page_width / 2
        full_width_threshold = page_width * 0.60
        full_width_blocks = []
        left_blocks       = []
        right_blocks      = []

        for block in text_blocks:

            x0   = block[0]
            y0   = block[1]
            x1   = block[2]
            text = block[4].strip()

            if not text:
                continue

            block_width  = x1 - x0
            block_center = (x0 + x1) / 2

            if block_width > full_width_threshold:
                full_width_blocks.append((y0, text))

            elif block_center <= midpoint:
                left_blocks.append((y0, text))

            else:
                right_blocks.append((y0, text))

        full_width_blocks.sort(key=lambda b: b[0])
        left_blocks.sort(key=lambda b: b[0])
        right_blocks.sort(key=lambda b: b[0])

        col_ys = [
            b[0] for b in left_blocks + right_blocks
        ]

        if not col_ys:

            return "\n\n".join(
                t for _, t in full_width_blocks
            )

        col_top_y = min(col_ys)

        fw_above = [
            (y, t) for y, t in full_width_blocks
            if y < col_top_y
        ]

        fw_below = [
            (y, t) for y, t in full_width_blocks
            if y >= col_top_y
        ]

        result_parts = []

        for _, t in fw_above:
            result_parts.append(t)

        for _, t in left_blocks:
            result_parts.append(t)

        for _, t in right_blocks:
            result_parts.append(t)

        for _, t in fw_below:
            result_parts.append(t)

        return "\n\n".join(result_parts)

    def _extract_double_column(

        self,
        text_blocks: List,
        page_width: float

    ) -> str:


        midpoint     = page_width / 2
        left_blocks  = []
        right_blocks = []

        for block in text_blocks:

            x0   = block[0]
            y0   = block[1]
            x1   = block[2]
            text = block[4].strip()

            if not text:
                continue

            if (x0 + x1) / 2 <= midpoint:
                left_blocks.append((y0, text))

            else:
                right_blocks.append((y0, text))

        left_blocks.sort(key=lambda b: b[0])
        right_blocks.sort(key=lambda b: b[0])

        parts = (
            [t for _, t in left_blocks]
            + [t for _, t in right_blocks]

        )

        return "\n\n".join(parts)

    def _extract_single_column(

        self,
        text_blocks: List
    ) -> str:


        sorted_blocks = sorted(
            text_blocks, key=lambda b: b[1]
        )

        parts = [

            b[4].strip()
            for b in sorted_blocks
            if b[4].strip()

        ]

        return "\n\n".join(parts)

    def _find_references_start(

        self,
        pages: List[Any]
    ) -> Optional[int]:

        reference_headings = {

            "references",
            "bibliography",
            "reference list",
            "works cited",
            "citations"
        }

        for page in pages:

            lines = page.text.strip().split('\n')
            for line in lines[:8]:

                clean_line = (
                    line.strip().lower().rstrip('.')
                )

                if clean_line in reference_headings:
                    return page.page_number

        return None

    def _clean_text_content(self, text: str) -> str:


        if self.prep_cfg.get('remove_page_numbers', True):
            text = self._patterns[

                "page_numbers"

            ].sub("", text)

        if self.prep_cfg.get(
            'remove_headers_footers', True

        ):

            text = self._patterns[

                "running_headers"

            ].sub("", text)

        if self.prep_cfg.get('detect_equations', True):
            text = self._patterns[

                "equation_lines"

            ].sub("[EQUATION]", text)

        text = self._patterns["multi_space"].sub(

            " ", text

        )

        text = self._patterns["multi_newline"].sub(

            "\n\n", text
        )

        return text.strip()

    def _compile_patterns(
        self

    ) -> Dict[str, re.Pattern]:

        return {

            "page_numbers": re.compile(
                r'^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$'
                r'|^\s*[Pp]age\s+\d+\s*$',
                re.MULTILINE
            ),

            "running_headers": re.compile(
                r'^[A-Z][A-Z\s,.:;0-9\-–]{10,79}$',
                re.MULTILINE

            ),

            "equation_lines": re.compile(
                r'^.*[∂∑∫√≤≥≠≈∈∉∀∃×÷±]{2,}.*$',
                re.MULTILINE

            ),

            "multi_space":   re.compile(r'  +'),
            "multi_newline": re.compile(r'\n{3,}'),

        }
 