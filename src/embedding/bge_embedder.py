"""

BGE Embedder

=============

Embeds text using BAAI BGE models via sentence-transformers.

Why BGE-M3:

  - 8192 token context window

    Your thesis sections never get truncated

  - 1024 embedding dimensions

    Rich, precise vector representations

  - Trained on academic and scientific text

    Understands biomedical terminology

  - Free, runs locally, 2.3GB RAM

    Fits your 16GB machine comfortably

Alternative BGE models for comparison experiments:

  bge-large-en-v1.5  → 1024 dims, 512 tokens, English only

  bge-base-en-v1.5   → 768 dims,  512 tokens, faster

  bge-small-en-v1.5  → 384 dims,  512 tokens, fastest

How BGE-M3 produces a vector:

  1. Tokenise text into subword pieces

  2. Pass through 24 transformer layers

  3. Each layer refines contextual understanding

  4. Final layer output pooled into one 1024-dim vector

  5. L2 normalised so all vectors have length 1

Normalisation (normalize_embeddings=True) is required

for BGE models. Without it cosine similarity scores

are meaningless.

"""

import os

import numpy as np

from typing import List, Dict, Any

from tqdm import tqdm

from embedding.base_embedder import BaseEmbedder


class BGEEmbedder(BaseEmbedder):

    """

    Embeds text using BAAI BGE models.

    Inherits from BaseEmbedder:

      __init__(config)      ← sets model_name, device, batch_size

      embed_single(text)    ← wraps embed() for single texts

    Implements required abstract methods:

      embed(texts)          ← written here

      get_embedding_dim()   ← written here

    Lazy loads model on first embed() call.

    Model stays in RAM for entire session.

    All 608 chunks embedded in one session.

    Usage:

      embedder = BGEEmbedder(config)

      vectors  = embedder.embed(["text1", "text2"])

      dim      = embedder.get_embedding_dim()  # 1024

    """

    def __init__(self, config: Dict[str, Any]):

        """

        Calls BaseEmbedder.__init__ then adds

        BGE-specific attributes.

        super().__init__(config) must be called first

        to set self.model_name, self.device, self.batch_size.

        Then we add BGE-specific settings on top.

        """

        super().__init__(config)

        # Model loaded lazily on first embed() call

        # Avoids loading 2.3GB into RAM if never used

        self._model = None

        self.normalize = self.embedding_cfg.get(

            'normalize_embeddings', True

        )

        self.max_seq_length = self.embedding_cfg.get(

            'max_seq_length', 512

        )

    def embed(

        self,

        texts: List[str]

    ) -> np.ndarray:

        """

        Embed a list of texts into vectors.

        Processes in batches to manage RAM usage.

        Shows progress bar so you can monitor progress.

        Batch processing explained:

          batch_size=8 means 8 texts fed to BGE-M3 at once.

          GPU can handle 32-64 simultaneously.

          CPU safely handles 8 without running out of RAM.

        For your 608 chunks on CPU:

          608 ÷ 8 = 76 batches

          ~12 chunks/second on modern CPU

          608 ÷ 12 ≈ 50 seconds total

        Args:

            texts: list of text strings to embed

        Returns:

            numpy array shape (len(texts), 1024)

            dtype: float32

        """

        self._load_model()

        if not texts:

            return np.array([])

        all_embeddings = []

        for i in tqdm(

            range(0, len(texts), self.batch_size),

            desc="  Embedding",

            unit="batch"

        ):

            batch = texts[i:i + self.batch_size]

            batch_embeddings = self._model.encode(

                batch,

                batch_size=self.batch_size,

                normalize_embeddings=self.normalize,

                show_progress_bar=False

            )

            all_embeddings.append(batch_embeddings)

        # Stack all batch results into one array

        # numpy.vstack combines list of 2D arrays:

        # [ (8,1024), (8,1024), ... ] → (608, 1024)

        return np.vstack(all_embeddings)

    def get_embedding_dim(self) -> int:

        """

        Return embedding dimension for this model.

        Called by embed.py to initialise FAISS:

          faiss.IndexFlatL2(embedder.get_embedding_dim())

        Loads model if not already loaded.

        Uses sentence-transformers built-in method.

        Returns:

            1024 for BGE-M3 and BGE-Large

            768  for BGE-Base

            384  for BGE-Small

        """

        self._load_model()

        return (

            self._model

            .get_sentence_embedding_dimension()

        )

    def _load_model(self) -> None:

        """

        Load BGE model into RAM.

        Called automatically on first embed() call.

        Subsequent calls return immediately —

        self._model is already set.

        First run:

          Downloads model from HuggingFace (~2.3GB for BGE-M3)

          Saves to artifacts/hf_cache/

          Takes 5-10 minutes depending on internet speed

        Subsequent runs:

          Loads from local cache

          Takes 10-20 seconds to load into RAM

        Why set HF_HOME?

          HuggingFace by default saves models to:

            C://Users//2000141904//.cache//huggingface

          That is buried in your user folder.

          We redirect to artifacts/hf_cache/ so

          everything stays inside the project folder.

          Easier to manage, easier to delete if needed.

        """

        if self._model is not None:

            return

        from sentence_transformers import (

            SentenceTransformer

        )

        # Redirect HuggingFace cache to project folder

        hf_cache = (

            self.config

            .get('paths', {})

            .get('hf_cache', './artifacts/hf_cache')

        )

        os.environ['HF_HOME'] = hf_cache

        os.environ['TRANSFORMERS_CACHE'] = hf_cache

        os.environ['SENTENCE_TRANSFORMERS_HOME'] = (

            hf_cache

        )

        print(f"\n  Loading {self.model_name}...")

        print(f"  Cache  : {hf_cache}")

        print(f"  Device : {self.device}")

        print(

            f"  First run downloads ~2.3GB. "

            f"Subsequent runs load from cache."

        )

        self._model = SentenceTransformer(

            self.model_name,

            device=self.device,

            cache_folder=hf_cache

        )

        # Set maximum sequence length

        # 512 tokens ≈ 380 words ≈ 2500 chars

        # Your hybrid child chunks avg 333 chars — fits

        self._model.max_seq_length = self.max_seq_length

        dim = (

            self._model

            .get_sentence_embedding_dimension()

        )

        print(

            f"  {self.model_name} loaded. "

            f"Dim: {dim}, "

            f"Max tokens: {self.max_seq_length}"

        )
 