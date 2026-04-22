"""

Base Embedder — Abstract Interface

=====================================

All embedding models inherit from this class.

Same pattern as BaseChunker:

  - Shared setup in __init__

  - Abstract methods embed() and get_embedding_dim()

  - Factory function get_embedder()

Why abstract interface?

  Experiment compares BGE-M3 vs BGE-Large.

  Phase B adds Azure OpenAI embeddings.

  All implement embed() identically from outside.

  Swap models by changing config.yaml only.

Hospital analogy:

  BaseEmbedder     = DepartmentProcessor(ABC)

  embed(texts)     = examine(patient)

  get_embedder()   = get_department()

  BGEEmbedder      = OphthalmologyDepartment

"""

from abc import ABC, abstractmethod

from typing import List, Dict, Any

import numpy as np


class BaseEmbedder(ABC):

    """

    Abstract base class for all embedding models.

    Every embedder must implement:

      embed()              → convert texts to vectors

      get_embedding_dim()  → return vector dimension

    Every embedder inherits:

      __init__()           → shared config setup

      embed_single()       → convenience wrapper

    """

    def __init__(self, config: Dict[str, Any]):

        """

        Shared setup for all embedders.

        Reads embedding settings from config.yaml:

          embedding:

            model_name: "BAAI/bge-m3"

            device: "cpu"

            batch_size: 8

        Args:

            config: dictionary from config.yaml

        """

        self.config        = config

        self.embedding_cfg = config.get('embedding', {})

        self.model_name = self.embedding_cfg.get(

            'model_name', 'BAAI/bge-m3'

        )

        self.device = self.embedding_cfg.get(

            'device', 'cpu'

        )

        self.batch_size = self.embedding_cfg.get(

            'batch_size', 8

        )

    @abstractmethod

    def embed(

        self,

        texts: List[str]

    ) -> np.ndarray:

        """

        Convert list of texts into embedding vectors.

        Must be implemented by every subclass.

        Args:

            texts: list of text strings to embed

        Returns:

            numpy array shape (len(texts), embed_dim)

            Each row is one text's embedding vector.

        """

        pass

    @abstractmethod

    def get_embedding_dim(self) -> int:

        """

        Return the dimension of this model's vectors.

        BGE-M3     → 1024

        BGE-Large  → 1024

        BGE-Base   → 768

        MiniLM-L6  → 384

        FAISS uses this to initialise the index:

          faiss.IndexFlatL2(embedding_dim)

        Must be implemented by every subclass.

        Returns:

            integer — number of dimensions per vector

        """

        pass

    def embed_single(self, text: str) -> np.ndarray:

        """

        Embed a single text string.

        Convenience wrapper around embed().

        Used at query time to embed the user's question.

        Args:

            text: single text string

        Returns:

            1D numpy array of shape (embed_dim,)

        """

        vectors = self.embed([text])

        return vectors[0]


def get_embedder(

    config: Dict[str, Any]

) -> BaseEmbedder:

    """

    Factory function — returns the right embedder

    based on config.yaml embedding.model_name.

    Current supported models:

      "BAAI/bge-m3"           → BGEEmbedder

      "BAAI/bge-large-en-v1.5"→ BGEEmbedder

      "BAAI/bge-base-en-v1.5" → BGEEmbedder

    Adding a new embedder in future:

      1. Write the new class inheriting BaseEmbedder

      2. Add one condition here

      Zero other changes needed.

    Args:

        config: dictionary from config.yaml

    Returns:

        appropriate BaseEmbedder subclass instance

    """

    from embedding.bge_embedder import BGEEmbedder

    model_name = (

        config

        .get('embedding', {})

        .get('model_name', 'BAAI/bge-m3')

        .lower()

    )

    if 'bge' in model_name:

        return BGEEmbedder(config)

    # Default fallback

    print(

        f"  Warning: unknown model '{model_name}', "

        f"defaulting to BGEEmbedder"

    )

    return BGEEmbedder(config)
 