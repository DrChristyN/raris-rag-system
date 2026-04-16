from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass

# ── Data structure ────────────────────────────────────────────
@dataclass
class Chunk:
   # The actual text content of this chunk
   # This is what gets embedded by BGE-M3
   text: str
   # Position of this chunk within the document
   # Chunk 0, chunk 1, chunk 2...
   chunk_index: int
   # Which PDF this chunk came from
   # e.g. "Published paper_BSPC.pdf"
   source_file: str
   # Which page in that PDF
   # Appears in the final answer citation
   page_number: int
   # "publication" | "thesis" | "paper"
   # Used for filtered retrieval later
   doc_type: str
   # Title, author, keywords from the PDF header
   # Travels with every chunk through the pipeline
   metadata: Dict[str, Any]
   # Which chunking strategy created this chunk
   # "recursive" | "semantic" | "hybrid"
   # Used in experiment comparisons
   strategy: str
   # Actual character count of this chunk
   # Used in get_stats() to compare strategies
   chunk_size: int
   # For hybrid chunking only — which parent owns this chunk
   # None for recursive and semantic chunks
   # "parent_0_BSPC_p4" for hybrid child chunks
   parent_id: str = None

# ── Base class ────────────────────────────────────────────────
class BaseChunker(ABC):

   def __init__(self, config: Dict[str, Any]):
       self.config       = config
       self.chunking_cfg = config.get('chunking', {})
   @abstractmethod
   def chunk(
       self,
       pages: List[Dict[str, Any]]
   ) -> List[Chunk]:
       pass
   
   def _make_chunk(
       self,
       text: str,
       chunk_index: int,
       page: Dict[str, Any],
       strategy: str,
       parent_id: str = None
   ) -> Chunk:
       return Chunk(
           text=text,
           chunk_index=chunk_index,
           source_file=page['source_file'],
           page_number=page['page_number'],
           doc_type=page['doc_type'],
           metadata=page['metadata'],
           strategy=strategy,
           chunk_size=len(text),
           parent_id=parent_id
       )
   def get_stats(
       self,
       chunks: List[Chunk]
   ) -> Dict[str, Any]:
       if not chunks:
           return {"count": 0}
       sizes = [c.chunk_size for c in chunks]
       return {
           "count":    len(chunks),
           "avg_size": round(sum(sizes) / len(sizes)),
           "min_size": min(sizes),
           "max_size": max(sizes),
           "strategy": chunks[0].strategy
       }

# ── Factory function ──────────────────────────────────────────
def get_chunker(
   strategy: str,
   config: Dict[str, Any]
) -> BaseChunker:
   
   # Imports are inside the function to avoid
   # circular imports — these files import from
   # base_chunker.py themselves
   from chunking.recursive import RecursiveChunker
   from chunking.semantic  import SemanticChunker
   from chunking.hybrid    import HybridChunker
   # Maps strategy name → class
   # Adding new strategy = adding one line here
   strategies = {
       "recursive": RecursiveChunker,
       "semantic":  SemanticChunker,
       "hybrid":    HybridChunker
   }
   # Validate the strategy name
   # Gives clear error if typo in config.yaml
   if strategy not in strategies:
       raise ValueError(
           f"Unknown chunking strategy: '{strategy}'\n"
           f"Available strategies: "
           f"{list(strategies.keys())}"
       )
   # Two steps in one line:
   # strategies["hybrid"]  → HybridChunker (the class)
   # HybridChunker(config) → creates and returns object
   return strategies[strategy](config)