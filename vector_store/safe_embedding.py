"""Safe Embedding Function for ChromaDB to ensure consistent tokenization."""

from __future__ import annotations
from typing import Any
import threading
from loguru import logger
from chromadb.utils import embedding_functions


def load_sentence_transformer(model_name: str, device: str):
    """Load a sentence-transformer from local cache first.

    Evaluation should be reproducible offline; if the model is not cached,
    the fallback keeps the original online-loading behavior.
    """
    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer(model_name, device="cpu", local_files_only=True)
    except TypeError:
        model = SentenceTransformer(model_name, device="cpu")
    except Exception as exc:
        logger.warning("Local embedding model load failed, trying online load: {}", exc)
        model = SentenceTransformer(model_name, device="cpu")
        
    if device != "cpu":
        model = model.to(device)
    return model


class SafeEmbeddingFunction(embedding_functions.EmbeddingFunction):
    _model_cache: dict[str, Any] = {}
    _cache_lock = threading.Lock()

    def __init__(self, model_name: str, device: str = "cpu", max_seq_length: int = 512):
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
            with SafeEmbeddingFunction._cache_lock:
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
