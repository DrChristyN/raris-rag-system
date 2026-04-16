from typing import List, Dict, Any
from chunking.base_chunker import BaseChunker, Chunk

class SemanticChunker(BaseChunker):
   def chunk(
       self,
       pages: List[Dict[str, Any]]
   ) -> List[Chunk]:
       raise NotImplementedError(
           "SemanticChunker coming soon. "
           "Use strategy: recursive for now."
       )