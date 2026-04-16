from typing import List, Dict, Any, Tuple
from chunking.base_chunker import BaseChunker, Chunk

class HybridChunker(BaseChunker):
   """
   Creates parent and child chunks from pages.
   Returns child chunks for FAISS indexing.
   Stores parent chunks separately for LLM context.
   After chunking:
     self.parent_chunks → dict mapping parent_id → text
     Use this when building the vector store to
     retrieve parent text at query time.
   Usage:
     chunker = HybridChunker(config)
     child_chunks = chunker.chunk(clean_pages)
     parents = chunker.parent_store
   """
   def __init__(self, config: Dict[str, Any]):
       """
       Calls parent __init__ then adds parent_store.
       parent_store is a dictionary:
         key   → parent_id string
         value → parent chunk text
       This is how child chunks find their parent
       at retrieval time.
       """
       # Call BaseChunker.__init__ first
       # This sets self.config and self.chunking_cfg
       super().__init__(config)
       # Additional storage specific to HybridChunker
       # Maps parent_id → parent text
       # Populated during chunk() call
       self.parent_store: Dict[str, str] = {}
   def chunk(
       self,
       pages: List[Dict[str, Any]]
   ) -> List[Chunk]:
       """
       Create child chunks for retrieval.
       Simultaneously build parent_store for LLM context.
       Process per page:
         1. Split page into parent chunks (large)
         2. For each parent, split into child chunks (small)
         3. Each child stores its parent_id
         4. Return all child chunks for FAISS indexing
       Args:
           pages: list of clean page dicts from preprocessor
       Returns:
           list of child Chunk objects for FAISS indexing
       """
       hybrid_cfg = self.chunking_cfg.get('hybrid', {})
       parent_chunk_size = hybrid_cfg.get(
           'parent_chunk_size', 1800
       )
       parent_overlap = hybrid_cfg.get(
           'parent_chunk_overlap', 200
       )
       child_chunk_size = hybrid_cfg.get(
           'child_chunk_size', 400
       )
       child_overlap = hybrid_cfg.get(
           'child_chunk_overlap', 50
       )
       print(
           f"\n  HybridChunker: "
           f"parent={parent_chunk_size}, "
           f"child={child_chunk_size}, "
           f"parent_overlap={parent_overlap}, "
           f"child_overlap={child_overlap}"
       )
       # Reset parent store for fresh run
       self.parent_store = {}
       all_child_chunks = []
       parent_index     = 0
       child_index      = 0
       for page in pages:
           text = page['text']
           if not text.strip():
               continue
           # Step 1 — Split page into parent chunks
           parent_texts = self._split_into_parents(
               text,
               parent_chunk_size,
               parent_overlap
           )
           for parent_text in parent_texts:
               parent_text = parent_text.strip()
               if len(parent_text) < 100:
                   continue
               # Step 2 — Create parent ID
               # Format: parent_{index}_{filename}_{page}
               # Filename shortened to first word for readability
               short_name = (
                   page['source_file']
                   .replace('.pdf', '')
                   .replace(' ', '_')[:10]
               )
               parent_id = (
                   f"parent_{parent_index}"
                   f"_{short_name}"
                   f"_p{page['page_number']}"
               )
               # Step 3 — Store parent text
               # Child chunks reference this by parent_id
               self.parent_store[parent_id] = {
                   "text":        parent_text,
                   "source_file": page['source_file'],
                   "page_number": page['page_number'],
                   "doc_type":    page['doc_type'],
                   "metadata":    page['metadata']
               }
               # Step 4 — Split parent into children
               child_texts = self._split_into_children(
                   parent_text,
                   child_chunk_size,
                   child_overlap
               )
               for child_text in child_texts:
                   child_text = child_text.strip()
                   if len(child_text) < 80:
                       continue
                   # Child chunk carries parent_id
                   # This is how retrieval finds the parent
                   child_chunk = self._make_chunk(
                       text=child_text,
                       chunk_index=child_index,
                       page=page,
                       strategy="hybrid",
                       parent_id=parent_id
                   )
                   all_child_chunks.append(child_chunk)
                   child_index += 1
               parent_index += 1
       stats = self.get_stats(all_child_chunks)
       print(
           f"  Parents created : {len(self.parent_store)}"
       )
       print(
           f"  Children created: {stats['count']} | "
           f"avg: {stats['avg_size']} chars | "
           f"min: {stats['min_size']} | "
           f"max: {stats['max_size']}"
       )
       print(
           f"  Avg children per parent: "
           f"{round(stats['count'] / max(len(self.parent_store), 1), 1)}"
       )
       return all_child_chunks
   def save_parent_store(
       self,
       output_path: str
   ) -> None:
       """
       Save parent chunks to JSON.
       Called after chunk() to persist parent texts.
       Loaded at query time to retrieve parent context.
       Args:
           output_path: where to save parent_store.json
       """
       import json
       from pathlib import Path
       Path(output_path).parent.mkdir(
           parents=True, exist_ok=True
       )
       with open(
           output_path, 'w', encoding='utf-8'
       ) as f:
           json.dump(
               self.parent_store, f,
               indent=2,
               ensure_ascii=False
           )
       print(
           f"  Saved {len(self.parent_store)} "
           f"parent chunks to {output_path}"
       )
   def load_parent_store(
       self,
       input_path: str
   ) -> None:
       """
       Load parent chunks from JSON.
       Called at query time to restore parent texts
       after loading child chunks from FAISS.
       Args:
           input_path: path to parent_store.json
       """
       import json
       with open(
           input_path, 'r', encoding='utf-8'
       ) as f:
           self.parent_store = json.load(f)
       print(
           f"  Loaded {len(self.parent_store)} "
           f"parent chunks from cache"
       )
   def get_parent_text(
       self,
       parent_id: str
   ) -> str:
       """
       Retrieve parent text by parent_id.
       Called at query time after retrieval finds
       a child chunk. Returns the full parent text
       to send to the LLM as context.
       Args:
           parent_id: the parent_id from a child Chunk
       Returns:
           parent text string, or empty string if not found
       """
       parent = self.parent_store.get(parent_id)
       if parent:
           return parent['text']
       return ""
   # ── Private methods ───────────────────────────────────────
   def _split_into_parents(
       self,
       text: str,
       parent_size: int,
       overlap: int
   ) -> List[str]:
       """
       Split page text into large parent chunks.
       Uses double newlines as primary separator to
       preserve paragraph boundaries. Falls back to
       single newlines then sentence boundaries.
       Parent chunks are large enough to contain a
       complete idea — typically one full section or
       two substantial paragraphs.
       Args:
           text:        page text to split
           parent_size: max chars per parent chunk
           overlap:     overlap between parent chunks
       Returns:
           list of parent text strings
       """
       parents = self._split_text(
           text,
           parent_size,
           separators=["\n\n", "\n", ". ", " "]
       )
       # Add overlap between parents
       if overlap > 0 and len(parents) > 1:
           parents = self._apply_overlap(
               parents, overlap
           )
       return parents
   def _split_into_children(
       self,
       parent_text: str,
       child_size: int,
       overlap: int
   ) -> List[str]:
       """
       Split one parent chunk into small child chunks.
       Children are precise — each contains one
       specific claim, finding, or description.
       FAISS searches children for exact matches.
       Args:
           parent_text: the parent chunk text
           child_size:  max chars per child chunk
           overlap:     overlap between child chunks
       Returns:
           list of child text strings
       """
       children = self._split_text(
           parent_text,
           child_size,
           separators=["\n\n", "\n", ". ", " "]
       )
       if overlap > 0 and len(children) > 1:
           children = self._apply_overlap(
               children, overlap
           )
       return children
   def _split_text(
       self,
       text: str,
       max_size: int,
       separators: List[str]
   ) -> List[str]:
       """
       Split text into pieces within max_size.
       Uses separator hierarchy — same logic as
       RecursiveChunker._recursive_split but
       simplified for the hybrid use case.
       Args:
           text:       text to split
           max_size:   maximum chars per piece
           separators: separators to try in order
       Returns:
           list of text pieces within max_size
       """
       if len(text) <= max_size:
           return [text]
       if not separators:
           # Force split by character count
           return [
               text[i:i+max_size]
               for i in range(0, len(text), max_size)
           ]
       separator = separators[0]
       remaining = separators[1:]
       pieces      = text.split(separator)
       final       = []
       current     = ""
       for piece in pieces:
           if not piece.strip():
               continue
           test = (
               current + separator + piece
               if current else piece
           )
           if len(test) <= max_size:
               current = test
           else:
               if current:
                   final.append(current)
               if len(piece) > max_size:
                   # Piece itself too large — recurse
                   sub = self._split_text(
                       piece, max_size, remaining
                   )
                   final.extend(sub)
                   current = ""
               else:
                   current = piece
       if current:
           final.append(current)
       return final if final else [text]
   def _apply_overlap(
       self,
       pieces: List[str],
       overlap: int
   ) -> List[str]:
       """
       Add overlap between consecutive text pieces.
       Takes last N chars of piece K and prepends
       to piece K+1 for context continuity.
       Args:
           pieces:  list of text strings
           overlap: number of chars to repeat
       Returns:
           list of overlapping text strings
       """
       if len(pieces) <= 1:
           return pieces
       result = [pieces[0]]
       for i in range(1, len(pieces)):
           prev    = pieces[i - 1]
           current = pieces[i]
           tail = (
               prev[-overlap:]
               if len(prev) > overlap
               else prev
           )
           result.append(tail + " " + current)
       return result