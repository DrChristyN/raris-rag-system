from typing import List, Dict, Any
from chunking.base_chunker import BaseChunker, Chunk

class HybridChunker(BaseChunker):
   def chunk(
       self,
       pages: List[Dict[str, Any]]
   ) -> List[Chunk]:
       raise NotImplementedError(
           "HybridChunker coming soon. "
           "Use strategy: recursive for now."
       )