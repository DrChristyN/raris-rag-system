import re

import numpy as np

from typing import List, Dict, Any, Tuple

from chunking.base_chunker import BaseChunker, Chunk


class SemanticChunker(BaseChunker):


    def __init__(self, config: Dict[str, Any]):


        super().__init__(config)

        # Model loaded lazily on first chunk() call

        self._embedder = None

        # Read semantic config

        semantic_cfg = self.chunking_cfg.get(

            'semantic', {}

        )

        self.threshold_type = semantic_cfg.get(

            'breakpoint_threshold_type', 'percentile'

        )

        self.threshold_amount = semantic_cfg.get(

            'breakpoint_threshold_amount', 95

        )

        self.min_chunk_size = self.chunking_cfg.get(

            'recursive', {}

        ).get('min_chunk_size', 150)

    def chunk(

        self,

        pages: List[Dict[str, Any]]

    ) -> List[Chunk]:


        print(

            f"\n  SemanticChunker: "

            f"threshold_type={self.threshold_type}, "

            f"threshold_amount={self.threshold_amount}"

        )

        # Load embedding model on first use

        self._load_embedder()

        all_chunks  = []

        chunk_index = 0

        for page in pages:

            text = page['text']

            if not text.strip():

                continue

            # Step 1 — split into sentences

            sentences = self._split_sentences(text)

            if len(sentences) <= 1:

                # Only one sentence — keep whole page

                if len(text.strip()) >= self.min_chunk_size:

                    chunk = self._make_chunk(

                        text=text.strip(),

                        chunk_index=chunk_index,

                        page=page,

                        strategy="semantic"

                    )

                    all_chunks.append(chunk)

                    chunk_index += 1

                continue

            # Step 2 — embed each sentence

            embeddings = self._embed_sentences(sentences)

            if embeddings is None:

                # Embedding failed — fall back to whole page

                if len(text.strip()) >= self.min_chunk_size:

                    chunk = self._make_chunk(

                        text=text.strip(),

                        chunk_index=chunk_index,

                        page=page,

                        strategy="semantic"

                    )

                    all_chunks.append(chunk)

                    chunk_index += 1

                continue

            # Step 3 — find breakpoints

            breakpoints = self._find_breakpoints(

                embeddings

            )

            # Step 4 — group sentences at breakpoints

            text_pieces = self._group_sentences(

                sentences, breakpoints

            )

            # Step 5 — merge small pieces

            text_pieces = self._merge_small_pieces(

                text_pieces

            )

            # Step 6 — create Chunk objects

            for piece in text_pieces:

                piece = piece.strip()

                if len(piece) < self.min_chunk_size:

                    continue

                chunk = self._make_chunk(

                    text=piece,

                    chunk_index=chunk_index,

                    page=page,

                    strategy="semantic"

                )

                all_chunks.append(chunk)

                chunk_index += 1

        stats = self.get_stats(all_chunks)

        print(

            f"  Created {stats['count']} chunks | "

            f"avg: {stats['avg_size']} chars | "

            f"min: {stats['min_size']} | "

            f"max: {stats['max_size']}"

        )

        return all_chunks

    # ── Private methods ───────────────────────────────────────

    def _load_embedder(self) -> None:

        if self._embedder is not None:

            return

        print(

            "  Loading sentence embedder for "

            "boundary detection..."

        )

        try:

            from sentence_transformers import (

                SentenceTransformer

            )

            # Small fast model for boundary detection

            # Not for final FAISS embeddings

            self._embedder = SentenceTransformer(

                'sentence-transformers/all-MiniLM-L6-v2'

            )

            print("  Sentence embedder loaded.")

        except Exception as e:

            print(f"  WARNING: Could not load embedder: {e}")

            print(

                "  SemanticChunker will fall back to "

                "whole-page chunks."

            )

            self._embedder = None

    def _split_sentences(

        self,

        text: str

    ) -> List[str]:

        # Replace newlines with spaces for consistent splitting

        text = re.sub(r'\n+', ' ', text)

        text = re.sub(r'\s+', ' ', text).strip()


        abbreviations = {

            'fig', 'et', 'al', 'vs', 'dr', 'prof',

            'mr', 'mrs', 'ms', 'no', 'vol', 'pp',

            'eq', 'sec', 'tab', 'approx', 'dept'

        }

        # Split on period + space + capital

        raw_sentences = re.split(

            r'(?<=[.!?])\s+(?=[A-Z0-9\[])', text

        )

        # Filter out abbreviation splits

        sentences = []

        current   = ""

        for sent in raw_sentences:

            if not sent.strip():

                continue

            # Check if this split was on an abbreviation

            last_word = (

                current.strip().split()[-1].lower().rstrip('.')

                if current.strip()

                else ""

            )

            if last_word in abbreviations and current:

                # Rejoin — was an abbreviation not sentence end

                current = current + " " + sent

            else:

                if current:

                    sentences.append(current.strip())

                current = sent

        if current:

            sentences.append(current.strip())

        # Filter empty and very short

        sentences = [

            s for s in sentences

            if len(s.strip()) > 10

        ]

        return sentences

    def _embed_sentences(

        self,

        sentences: List[str]

    ) -> np.ndarray:


        if self._embedder is None:

            return None

        try:

            embeddings = self._embedder.encode(

                sentences,

                batch_size=32,

                show_progress_bar=False,

                normalize_embeddings=True

            )

            return embeddings

        except Exception as e:

            print(f"  WARNING: Embedding failed: {e}")

            return None

    def _find_breakpoints(

        self,

        embeddings: np.ndarray

    ) -> List[int]:

        if len(embeddings) < 2:

            return []


        similarities = []

        for i in range(len(embeddings) - 1):

            sim = float(

                np.dot(embeddings[i], embeddings[i + 1])

            )

            similarities.append(sim)

        if not similarities:

            return []


        if self.threshold_type == "percentile":

            threshold = float(

                np.percentile(

                    similarities,

                    100 - self.threshold_amount

                )

            )

        else:

            # Fixed threshold

            threshold = float(self.threshold_amount)

        # Collect breakpoint indices

        # A breakpoint at index i means:

        # cut between sentence i and sentence i+1

        breakpoints = [

            i for i, sim in enumerate(similarities)

            if sim < threshold

        ]

        return breakpoints

    def _group_sentences(

        self,

        sentences: List[str],

        breakpoints: List[int]

    ) -> List[str]:


        if not breakpoints:

            return [" ".join(sentences)]

        groups = []

        start  = 0

        for bp in sorted(breakpoints):

            group = " ".join(sentences[start:bp + 1])

            if group.strip():

                groups.append(group.strip())

            start = bp + 1

        # Last group

        last_group = " ".join(sentences[start:])

        if last_group.strip():

            groups.append(last_group.strip())

        return groups

    def _merge_small_pieces(

        self,

        pieces: List[str]

    ) -> List[str]:


        if not pieces:

            return pieces

        target = (

            self.chunking_cfg

            .get('recursive', {})

            .get('target_chunk_size', 400)

        )

        merged  = []

        current = ""

        for piece in pieces:

            piece = piece.strip()

            if not piece:

                continue

            if not current:

                current = piece

            elif len(current) + len(piece) + 1 <= target:

                current = current + " " + piece

            else:

                if len(current) >= self.min_chunk_size:

                    merged.append(current)

                elif merged:

                    merged[-1] = merged[-1] + " " + current

                current = piece

        if current:

            if len(current) >= self.min_chunk_size:

                merged.append(current)

            elif merged:

                merged[-1] = merged[-1] + " " + current

        return merged
 