"""
FAISS Vector Store
==================
Manages the embedding and retrieval of text chunks using Sentence-Transformers
and FAISS for fast approximate nearest neighbour search.

This module powers the RAG (Retrieval-Augmented Generation) component of the
pipeline: when generating questions, we retrieve the most relevant context
chunks to feed to the LLM.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS-backed vector store for text chunks.

    Handles:
    - Embedding text chunks with a Sentence-Transformer model
    - Building and saving a FAISS index
    - Querying the index for top-k similar chunks
    """

    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        index_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        from config import EMBEDDING_MODEL_NAME, FAISS_INDEX_DIR, DEVICE

        self.embedding_model_name = embedding_model_name or EMBEDDING_MODEL_NAME
        self.index_dir = Path(index_dir or str(FAISS_INDEX_DIR))
        self.device = device or DEVICE

        self._model = None
        self._index = None
        self._chunks_metadata: list[dict] = []  # parallel to index vectors
        self._dimension: int = 0

    # ─── Lazy model loading ──────────────────────────────────────────────────

    @property
    def model(self):
        """Lazy-load the SentenceTransformer embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", self.embedding_model_name)
            self._model = SentenceTransformer(
                self.embedding_model_name, device=self.device
            )
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info("Embedding dimension: %d", self._dimension)
        return self._model

    # ─── Embedding ──────────────────────────────────────────────────────────

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Embed a list of texts into dense vectors.

        Parameters
        ----------
        texts : List[str]
            Texts to embed.
        batch_size : int
            Batch size for encoding.

        Returns
        -------
        np.ndarray
            Matrix of shape (len(texts), dimension).
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True,  # for cosine similarity via inner product
        )
        return embeddings.astype(np.float32)

    # ─── Index building ──────────────────────────────────────────────────────

    def build_index(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
    ) -> None:
        """
        Build a FAISS index from a list of texts.

        Parameters
        ----------
        texts : List[str]
            The text chunks to index.
        metadatas : List[dict], optional
            Parallel metadata for each chunk (heading, page, chunk_id, etc.).
        """
        import faiss

        logger.info("Building FAISS index from %d chunks...", len(texts))
        embeddings = self.embed_texts(texts)
        dimension = embeddings.shape[1]

        # Use IndexFlatIP (inner product ≈ cosine similarity for normalised vectors)
        self._index = faiss.IndexFlatIP(dimension)
        self._index.add(embeddings)
        self._dimension = dimension
        self._chunks_metadata = metadatas or [
            {"text": t, "index": i} for i, t in enumerate(texts)
        ]

        logger.info(
            "FAISS index built: %d vectors, dimension %d.",
            self._index.ntotal,
            dimension,
        )

    # ─── Querying ────────────────────────────────────────────────────────────

    def query(
        self, query_text: str, top_k: int = 5
    ) -> List[Tuple[dict, float]]:
        """
        Retrieve the top-k most similar chunks to the query.

        Parameters
        ----------
        query_text : str
            The query string.
        top_k : int
            Number of results to return.

        Returns
        -------
        List[Tuple[dict, float]]
            List of (metadata_dict, similarity_score) tuples sorted by relevance.
        """
        if self._index is None or self._index.ntotal == 0:
            logger.warning("FAISS index is empty. Returning empty results.")
            return []

        query_embedding = self.embed_texts([query_text])
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue  # FAISS returns -1 for padding
            meta = self._chunks_metadata[idx].copy()
            results.append((meta, float(score)))

        return results

    # ─── Persistence ─────────────────────────────────────────────────────────

    def save(self, name: str = "default") -> None:
        """Save the FAISS index and metadata to disk."""
        import faiss

        if self._index is None:
            raise ValueError("No index to save. Call build_index() first.")

        self.index_dir.mkdir(parents=True, exist_ok=True)
        index_path = self.index_dir / f"{name}.faiss"
        meta_path = self.index_dir / f"{name}_meta.json"

        faiss.write_index(self._index, str(index_path))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._chunks_metadata, f, ensure_ascii=False, indent=2)

        logger.info("Saved FAISS index to %s", index_path)

    def load(self, name: str = "default") -> None:
        """Load a FAISS index and metadata from disk."""
        import faiss

        index_path = self.index_dir / f"{name}.faiss"
        meta_path = self.index_dir / f"{name}_meta.json"

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        self._index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            self._chunks_metadata = json.load(f)

        logger.info(
            "Loaded FAISS index from %s (%d vectors).",
            index_path,
            self._index.ntotal,
        )

    # ─── Utilities ───────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self._index.ntotal if self._index else 0

    def clear(self) -> None:
        """Reset the index."""
        self._index = None
        self._chunks_metadata = []
