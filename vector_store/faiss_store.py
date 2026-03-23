"""FAISS vector index management.

Provides an interface for building, querying, saving, and loading
FAISS indexes with associated document metadata.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import yaml
from loguru import logger


@dataclass
class SearchResult:
    """A single FAISS search result.

    Attributes:
        doc_id: Document identifier from metadata.
        chunk_text: The original text chunk.
        score: Similarity score (higher is better for inner-product).
        metadata: Additional metadata for this chunk.
    """

    doc_id: str
    chunk_text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class FAISSStore:
    """FAISS index wrapper with metadata tracking.

    Manages a flat inner-product index and an aligned metadata list.
    Supports incremental addition, search, and persistence.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize FAISS store.

        Args:
            config: FAISS config dict. If None, loads from configs/config.yaml.
        """
        if config is None:
            config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                full_config = yaml.safe_load(f)
            config = full_config["faiss"]

        self.index_path = Path(config["index_path"])
        self.metadata_path = Path(config["metadata_path"])
        self.top_k: int = config.get("top_k", 5)

        self._index: faiss.Index | None = None
        self._metadata: list[dict[str, Any]] = []
        self._dim: int | None = None

        logger.info(
            "FAISSStore initialized | index_path={} | top_k={}",
            self.index_path,
            self.top_k,
        )

    def _ensure_index(self, dim: int) -> None:
        """Create the FAISS index if it doesn't exist.

        Args:
            dim: Embedding dimension.
        """
        if self._index is None:
            self._dim = dim
            self._index = faiss.IndexFlatIP(dim)
            logger.info("Created new FAISS IndexFlatIP | dim={}", dim)

    @property
    def size(self) -> int:
        """Return the number of vectors in the index."""
        if self._index is None:
            return 0
        return self._index.ntotal

    def add(
        self,
        embeddings: np.ndarray,
        metadata_list: list[dict[str, Any]],
    ) -> None:
        """Add embeddings with metadata to the index.

        Args:
            embeddings: Array of shape (n, dim).
            metadata_list: List of metadata dicts, one per embedding.

        Raises:
            ValueError: If embeddings and metadata counts don't match.
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError(
                f"Embeddings ({len(embeddings)}) and metadata ({len(metadata_list)}) count mismatch"
            )

        self._ensure_index(embeddings.shape[1])
        assert self._index is not None

        # Normalize for inner-product search
        faiss.normalize_L2(embeddings)
        self._index.add(embeddings.astype(np.float32))
        self._metadata.extend(metadata_list)

        logger.debug(
            "Added {} vectors to FAISS | total={}",
            len(embeddings),
            self._index.ntotal,
        )

    def search(self, query_embedding: np.ndarray, top_k: int | None = None) -> list[SearchResult]:
        """Search for nearest neighbors.

        Args:
            query_embedding: 1-D or 2-D query vector.
            top_k: Number of results. Defaults to config top_k.

        Returns:
            List of SearchResult, sorted by score descending.
        """
        if self._index is None or self._index.ntotal == 0:
            logger.warning("FAISS index is empty, returning no results")
            return []

        k = top_k or self.top_k
        k = min(k, self._index.ntotal)

        # Reshape to 2-D if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)

        scores, indices = self._index.search(query_embedding, k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = self._metadata[idx]
            results.append(SearchResult(
                doc_id=meta.get("doc_id", ""),
                chunk_text=meta.get("chunk_text", ""),
                score=float(score),
                metadata=meta,
            ))

        return results

    def save(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        if self._index is None:
            logger.warning("No index to save")
            return

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(self.index_path))
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self._metadata, f)

        logger.info(
            "FAISS index saved | vectors={} | path={}",
            self._index.ntotal,
            self.index_path,
        )

    def load(self) -> bool:
        """Load a previously saved FAISS index and metadata.

        Returns:
            True if loading succeeded, False otherwise.
        """
        if not self.index_path.exists() or not self.metadata_path.exists():
            logger.warning("FAISS index files not found at {}", self.index_path)
            return False

        try:
            self._index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, "rb") as f:
                self._metadata = pickle.load(f)
            self._dim = self._index.d
            logger.info(
                "FAISS index loaded | vectors={} | dim={}",
                self._index.ntotal,
                self._dim,
            )
            return True
        except Exception as exc:
            logger.error("Failed to load FAISS index: {}", exc)
            return False
