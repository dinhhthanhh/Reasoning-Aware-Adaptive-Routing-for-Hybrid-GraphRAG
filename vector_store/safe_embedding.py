"""Safe Embedding Function for ChromaDB to ensure consistent tokenization."""

from __future__ import annotations
from typing import Any
from loguru import logger
from chromadb.utils import embedding_functions


def load_sentence_transformer(model_name: str, device: str):
    """Load a sentence-transformer from local cache first.

    Evaluation should be reproducible offline; if the model is not cached,
    the fallback keeps the original online-loading behavior.
    """
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(model_name, device=device, local_files_only=True)
    except TypeError:
        return SentenceTransformer(model_name, device=device)
    except Exception as exc:
        logger.warning("Local embedding model load failed, trying online load: {}", exc)
        return SentenceTransformer(model_name, device=device)


class SafeEmbeddingFunction(embedding_functions.EmbeddingFunction):
    _model_cache: dict[str, Any] = {}

    def __init__(self, model_name: str, device: str, max_seq_length: int = 512):
        """Initialize safe embedding function with model caching.
        
        Args:
            model_name: HuggingFace model name.
            device: 'cuda' or 'cpu'.
            max_seq_length: Force token-level truncation at this length.
        """
        cache_key = f"{model_name}_{device}"
        
        if cache_key in SafeEmbeddingFunction._model_cache:
            self._model = SafeEmbeddingFunction._model_cache[cache_key]
        else:
            logger.info("Loading embedding model into cache: {} (device: {})", model_name, device)
            self._model = load_sentence_transformer(model_name, device)
            SafeEmbeddingFunction._model_cache[cache_key] = self._model
            
        self._model.max_seq_length = max_seq_length  # Force token-level truncation

    def __call__(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts."""
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        return self._model.encode(list(texts), show_progress_bar=False).tolist()
