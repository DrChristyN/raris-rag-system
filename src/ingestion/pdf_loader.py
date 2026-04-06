import fitz
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class DocumentPage:
   text: str
   page_number: int
   source_file: str
   source_path: str
   doc_type: str
   total_pages: int
   metadata: Dict[str, Any]

@dataclass
class LoadedDocument:
   filename: str
   filepath: str
   doc_type: str
   pages: List[DocumentPage]
   total_pages: int
   pdf_metadata: Dict[str, Any]
   def to_dict(self) -> Dict:
       return asdict(self)
   def get_full_text(self) -> str:
       return "\n\n".join(p.text for p in self.pages)
   def get_page_count(self) -> int:
       return len(self.pages)

class PDFLoader:
   def __init__(self, config: Dict[str, Any]):
       self.config = config
       self.preprocessing_cfg = config.get('preprocessing', {})
       self.min_page_chars = 50
   def load_pdf(
       self,
       filepath: str,
       doc_type: str = "paper"
   ) -> LoadedDocument:
       
       filepath = Path(filepath)
       if not filepath.exists():
           raise FileNotFoundError(
               f"PDF not found: {filepath}"
           )
       print(f"  Loading: {filepath.name}")
       try:
           pdf_doc = fitz.open(str(filepath))
       except Exception as e:
           raise ValueError(
               f"Cannot open {filepath.name}: {e}"
           )
       if pdf_doc.is_encrypted:
           raise ValueError(
               f"{filepath.name} is password protected. "
               f"Please remove password first."
           )
       pdf_metadata = self._extract_pdf_metadata(
           pdf_doc, filepath
       )
       pages = []
       total_pages = len(pdf_doc)
       print(
           f"    Extracting {total_pages} pages...",
           end="",
           flush=True
       )
       for page_num in range(total_pages):
           page = pdf_doc[page_num]
           text = page.get_text("text")
           if len(text.strip()) < self.min_page_chars:
               continue
           text = self._clean_page_text(text)
           page_obj = DocumentPage(
               text=text,
               page_number=page_num + 1,
               source_file=filepath.name,
               source_path=str(filepath),
               doc_type=doc_type,
               total_pages=total_pages,
               metadata=pdf_metadata
           )
           pages.append(page_obj)
       pdf_doc.close()
       print(f" done. ({len(pages)} pages extracted)")
       return LoadedDocument(
           filename=filepath.name,
           filepath=str(filepath),
           doc_type=doc_type,
           pages=pages,
           total_pages=total_pages,
           pdf_metadata=pdf_metadata
       )
   def load_directory(
       self,
       directory: str,
       doc_types: Optional[Dict[str, str]] = None
   ) -> List[LoadedDocument]:
       directory = Path(directory)
       if not directory.exists():
           raise FileNotFoundError(
               f"Directory not found: {directory}"
           )
       pdf_files = sorted(directory.glob("*.pdf"))
       if not pdf_files:
           print(f"  No PDFs found in {directory}")
           return []
       print(f"\nFound {len(pdf_files)} PDF(s) in {directory}")
       doc_types = doc_types or {}
       documents = []
       for pdf_path in pdf_files:
           doc_type = doc_types.get(
               pdf_path.name, "publication"
           )
           try:
               doc = self.load_pdf(
                   str(pdf_path),
                   doc_type=doc_type
               )
               documents.append(doc)
           except (FileNotFoundError, ValueError) as e:
               print(f"  SKIPPED {pdf_path.name}: {e}")
               continue
       total_pages = sum(d.total_pages for d in documents)
       print(
           f"\nLoaded {len(documents)} documents, "
           f"{total_pages} total pages"
       )
       return documents
   def save_processed(
       self,
       documents: List[LoadedDocument],
       output_dir: str
   ) -> str:
       output_dir = Path(output_dir)
       output_dir.mkdir(parents=True, exist_ok=True)
       output_path = output_dir / "extracted_documents.json"
       data = {
           "total_documents": len(documents),
           "total_pages": sum(
               d.total_pages for d in documents
           ),
           "documents": [
               doc.to_dict() for doc in documents
           ]
       }
       with open(output_path, "w", encoding="utf-8") as f:
           json.dump(data, f, indent=2, ensure_ascii=False)
       size_kb = output_path.stat().st_size / 1024
       print(f"\nSaved to: {output_path}")
       print(f"File size: {size_kb:.1f} KB")
       return str(output_path)
   def load_processed(self, input_path: str) -> List[Dict]:
       with open(input_path, "r", encoding="utf-8") as f:
           data = json.load(f)
       print(
           f"Loaded {data['total_documents']} documents "
           f"from cache ({data['total_pages']} pages)"
       )
       return data['documents']
   def _extract_pdf_metadata(
       self,
       pdf_doc: fitz.Document,
       filepath: Path
   ) -> Dict[str, Any]:
       raw_meta = pdf_doc.metadata
       return {
           "title": raw_meta.get(
               "title", filepath.stem
           ),
           "author": raw_meta.get("author", "Unknown"),
           "subject": raw_meta.get("subject", ""),
           "keywords": raw_meta.get("keywords", ""),
           "creation_date": raw_meta.get(
               "creationDate", ""
           ),
           "filename": filepath.name,
           "file_size_kb": round(
               filepath.stat().st_size / 1024, 1
           )
       }
   def _clean_page_text(self, text: str) -> str:
       text = text.replace("\f", "\n")
       text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
       text = re.sub(r'\n{3,}', '\n\n', text)
       text = text.strip()
       return text