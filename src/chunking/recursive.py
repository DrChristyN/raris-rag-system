"""

Recursive Character Text Splitter

===================================

Splits text using a hierarchy of separators.

Strategy:

  Try to split on double newlines first (paragraph breaks).

  If chunk is still too large, split on single newlines.

  If still too large, split on sentences.

  If still too large, split on spaces (words).

  Last resort — split on individual characters.

Then merges small fragments into target-sized chunks.

Then adds overlap between consecutive chunks.

Why "recursive"?

  The splitter recursively tries each separator in order

  until the chunk fits within chunk_size.

Best for:

  General text, thesis body chapters, methodology sections.

Hospital analogy:

  RecursiveChunker = GeneralWardDepartment

  chunk(pages)     = examine(patient)

  Inherits __init__, get_stats, _make_chunk from BaseChunker

"""

from typing import List, Dict, Any

from chunking.base_chunker import BaseChunker, Chunk


class RecursiveChunker(BaseChunker):

    """

    Splits pages into chunks using separator hierarchy.

    Inherits from BaseChunker:

      __init__(config)      ← never written here

      get_stats(chunks)     ← never written here

      _make_chunk(...)      ← never written here

    Implements required abstract method:

      chunk(pages)          ← written here

    """

    def chunk(

        self,

        pages: List[Dict[str, Any]]

    ) -> List[Chunk]:

        """

        Split all pages into chunks.

        Three stages:

          1. Split text using separator hierarchy

          2. Merge small fragments into target-sized pieces

          3. Add overlap between consecutive chunks

        Args:

            pages: list of clean page dicts from preprocessor

        Returns:

            list of Chunk objects

        """

        recursive_cfg = self.chunking_cfg.get(

            'recursive', {}

        )

        chunk_size    = recursive_cfg.get(

            'chunk_size', 1000

        )

        chunk_overlap = recursive_cfg.get(

            'chunk_overlap', 200

        )

        min_chunk_size = recursive_cfg.get(

            'min_chunk_size', 150

        )

        target_chunk_size = recursive_cfg.get(

            'target_chunk_size', 400

        )

        separators = recursive_cfg.get(

            'separators',

            ["\n\n", "\n", ". ", " ", ""]

        )

        all_chunks  = []

        chunk_index = 0

        print(

            f"\n  RecursiveChunker: "

            f"chunk_size={chunk_size}, "

            f"overlap={chunk_overlap}, "

            f"target={target_chunk_size}, "

            f"min={min_chunk_size}"

        )

        for page in pages:

            text = page['text']

            if not text.strip():

                continue

            # Stage 1 — split using separator hierarchy

            text_pieces = self._recursive_split(

                text,

                separators,

                chunk_size

            )

            # Stage 2 — merge small fragments together

            text_pieces = self._merge_small_pieces(

                text_pieces,

                target_chunk_size,

                min_chunk_size

            )

            # Stage 3 — add overlap between pieces

            text_pieces = self._add_overlap(

                text_pieces,

                chunk_overlap

            )

            # Create Chunk objects

            for piece in text_pieces:

                piece = piece.strip()

                if len(piece) < min_chunk_size:

                    continue

                chunk = self._make_chunk(

                    text=piece,

                    chunk_index=chunk_index,

                    page=page,

                    strategy="recursive"

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

    def _recursive_split(

        self,

        text: str,

        separators: List[str],

        chunk_size: int

    ) -> List[str]:

        """

        Core recursive splitting algorithm.

        Tries each separator in order.

        If a piece is still too large after splitting,

        recursively splits it with the next separator.

        Args:

            text:       the text to split

            separators: list of separators to try in order

            chunk_size: maximum characters per chunk

        Returns:

            list of text pieces all within chunk_size

        """

        # Base case — already fits

        if len(text) <= chunk_size:

            return [text]

        # No more separators — force split

        if not separators:

            return self._force_split(text, chunk_size)

        current_separator    = separators[0]

        remaining_separators = separators[1:]

        if current_separator == "":

            return self._force_split(text, chunk_size)

        # Split on current separator

        raw_pieces = text.split(current_separator)

        # Rejoin separator to preserve text structure

        pieces = []

        for i, piece in enumerate(raw_pieces):

            if i < len(raw_pieces) - 1:

                pieces.append(piece + current_separator)

            else:

                pieces.append(piece)

        # Check each piece — recurse if too large

        final_pieces = []

        for piece in pieces:

            if not piece.strip():

                continue

            if len(piece) <= chunk_size:

                final_pieces.append(piece)

            else:

                sub_pieces = self._recursive_split(

                    piece,

                    remaining_separators,

                    chunk_size

                )

                final_pieces.extend(sub_pieces)

        return final_pieces

    def _merge_small_pieces(

        self,

        pieces: List[str],

        target_size: int,

        min_size: int

    ) -> List[str]:

        """

        Merge consecutive small pieces into larger chunks.

        Combines pieces until the combined text reaches

        target_size characters. This prevents the pipeline

        from creating hundreds of tiny meaningless fragments

        from short lines in academic papers.

        Example with target_size=400:

          Input pieces (from double-column paper):

            "The proposed method"           (19 chars)

            "uses attention gates"           (20 chars)

            "to focus on vessels."           (20 chars)

            "The gate suppresses background" (30 chars)

            "and improves segmentation of"   (28 chars)

            "thin vessels significantly."    (27 chars)

          After merging:

            "The proposed method uses attention gates

             to focus on vessels. The gate suppresses

             background and improves segmentation of

             thin vessels significantly."

             (combined: 144 chars — still under target,

              but next piece starts a new topic)

        Args:

            pieces:      list of text pieces from splitter

            target_size: aim to combine up to this size

            min_size:    minimum size to keep a chunk

        Returns:

            list of merged text pieces

        """

        if not pieces:

            return pieces

        merged  = []

        current = ""

        for piece in pieces:

            piece = piece.strip()

            if not piece:

                continue

            if not current:

                # Start accumulating

                current = piece

            elif len(current) + len(piece) + 1 <= target_size:

                # Combine — fits within target

                current = current + " " + piece

            else:

                # Current reached target size

                # Save it and start fresh

                if len(current) >= min_size:

                    merged.append(current)

                elif merged:

                    # Too small — append to previous chunk

                    merged[-1] = merged[-1] + " " + current

                current = piece

        # Handle the final accumulated piece

        if current:

            if len(current) >= min_size:

                merged.append(current)

            elif merged:

                merged[-1] = merged[-1] + " " + current

        return merged

    def _add_overlap(

        self,

        pieces: List[str],

        overlap: int

    ) -> List[str]:

        """

        Add overlap between consecutive chunks.

        Repeats the last N characters of chunk K

        at the start of chunk K+1.

        Why overlap?

        Prevents context loss at chunk boundaries.

        A sentence split across two chunks remains

        understandable in both chunks.

        Args:

            pieces:  list of text pieces

            overlap: number of characters to repeat

        Returns:

            list of overlapping text pieces

        """

        if overlap == 0 or len(pieces) <= 1:

            return pieces

        overlapped = [pieces[0]]

        for i in range(1, len(pieces)):

            previous = pieces[i - 1]

            current  = pieces[i]

            if len(previous) > overlap:

                overlap_text = previous[-overlap:]

            else:

                overlap_text = previous

            overlapped.append(overlap_text + " " + current)

        return overlapped

    def _force_split(

        self,

        text: str,

        chunk_size: int

    ) -> List[str]:

        """

        Last resort — split by character count.

        Only triggers when no separator works and

        text is still too large. Should rarely happen

        with well-formatted academic text.

        Args:

            text:       text to split

            chunk_size: maximum characters per piece

        Returns:

            list of fixed-size text pieces

        """

        pieces = []

        start  = 0

        while start < len(text):

            end = start + chunk_size

            pieces.append(text[start:end])

            start = end

        return pieces
 