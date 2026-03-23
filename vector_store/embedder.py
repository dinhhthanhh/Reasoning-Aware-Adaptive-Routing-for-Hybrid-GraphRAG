"""Sentence-transformer embedding wrapper.

Provides Vietnamese-capable text embeddings using keepitreal/vietnamese-sbert
from HuggingFace sentence-transformers. Supports batched encoding with GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from loguru import logger
from sentence_transformers import SentenceTransformer


class Embedder:
    """Text embedder using Vietnamese sentence-transformers.

    Lazy-loads the model on first use to conserve VRAM until needed.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize embedder.

        Args:
            config: Embedding config dict. If None, loads from configs/config.yaml.
        """
        if config is None:
            config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                full_config = yaml.safe_load(f)
            config = full_config["embedding"]

        self.model_name: str = config["model_name"]
        self.batch_size: int = config.get("batch_size", 32)
        self.max_length: int = config.get("max_length", 512)
        self.device: str = config.get("device", "auto")
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model: SentenceTransformer | None = None
        self._embedding_dim: int | None = None
        logger.info("Embedder initialized | model={} | device={}", self.model_name, self.device)

    def _load_model(self) -> None:
        """Lazy-load the sentence-transformer model."""
        if self._model is not None:
            return
        logger.info("Loading embedding model: {}", self.model_name)
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._model.max_seq_length = self.max_length
        self._embedding_dim = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Embedding model loaded | dim={} | max_seq_length={}",
            self._embedding_dim,
            self.max_length,
        )

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension of the loaded model."""
        self._load_model()
        assert self._embedding_dim is not None
        return self._embedding_dim

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Encode texts into embedding vectors.

        Args:
            texts: List of text strings to embed.
            show_progress: Whether to show a progress bar.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        self._load_model()
        assert self._model is not None

        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        logger.debug("Encoded {} texts → shape {}", len(texts), embeddings.shape)
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text into an embedding vector.

        Args:
            text: Single text string.

        Returns:
            1-D numpy array of shape (embedding_dim,).
        """
        result = self.encode([text])
        return result[0]
