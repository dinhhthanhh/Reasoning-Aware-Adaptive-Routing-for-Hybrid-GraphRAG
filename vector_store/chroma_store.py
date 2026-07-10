"""ChromaDB vector store management."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from loguru import logger
from .safe_embedding import SafeEmbeddingFunction

@dataclass
class SearchResult:
    """A single vector search result."""
    doc_id: str
    chunk_text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

class ChromaStore:
    """ChromaDB wrapper for persistent vector storage and retrieval."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize ChromaDB store.
        
        Args:
            config: Configuration dict with chroma and embedding settings.
        """
        self.path = Path(config.get("path", "data/vector_store/chroma"))
        self.collection_name = config.get("collection_name", "legal_docs")
        
        # Use SafeEmbeddingFunction to ensure consistency with build_vectordb.py
        model_name = config.get("model_name", "microsoft/Harrier-OSS-v1-0.6B")
        device = config.get("device", "cuda")
        max_length = config.get("max_length", 512)
        
        self.ef = SafeEmbeddingFunction(
            model_name=model_name,
            device=device,
            max_seq_length=max_length
        )
        
        try:
            import chromadb.config
            settings = chromadb.config.Settings(
                anonymized_telemetry=False,
                is_persistent=True,
            )
            self.client = chromadb.PersistentClient(path=str(self.path), settings=settings)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.ef,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.error("Failed to connect to ChromaDB or load collection: {}", e)
            if "hnsw" in str(e).lower():
                logger.warning("HNSW index error detected. This often happens due to memory pressure or a corrupted index.")
            raise
        
        logger.info("ChromaStore initialized | path={} | collection={}", self.path, self.collection_name)

    def search(self, query: Any, top_k: int = 5, where: dict[str, Any] | None = None) -> list[SearchResult]:
        """Search ChromaDB. Accepts string query or embedding vector."""
        
        query_kwargs = {"n_results": top_k}
        if where:
            query_kwargs["where"] = where
            
        if isinstance(query, str):
            # Chroma automatically uses the collection's embedding_function for string queries
            results = self.collection.query(
                query_texts=[query],
                **query_kwargs
            )
        else:
            # Handle vector input
            if hasattr(query, "tolist"):
                query = query.tolist()
            
            # Ensure query is a list of lists for Chroma
            query_embeddings = [query] if not isinstance(query[0], list) else query
                
            results = self.collection.query(
                query_embeddings=query_embeddings,
                **query_kwargs
            )
        
        search_results = []
        if not results or not results['documents']:
            return []
            
        for i in range(len(results['documents'][0])):
            doc = results['documents'][0][i]
            meta = results['metadatas'][0][i]
            dist = results['distances'][0][i]
            score = 1.0 - dist # approximate cosine similarity from distance
            
            search_results.append(SearchResult(
                doc_id=results['ids'][0][i],
                chunk_text=doc,
                score=float(score),
                metadata=meta
            ))
            
        return search_results

    def get_by_canonical_id(self, canonical_id: str) -> SearchResult | None:
        """Fetch one chunk by Pháp Điển canonical id (law_number::article_num)."""
        if not canonical_id or "::" not in canonical_id:
            return None
        try:
            got = self.collection.get(ids=[canonical_id])
        except Exception:
            return None
        if not got or not got.get("ids"):
            # Metadata filter fallback when id was deduplicated (e.g. cid_2)
            try:
                got = self.collection.get(where={"canonical_id": canonical_id}, limit=1)
            except Exception:
                return None
        if not got or not got.get("ids"):
            return None
        doc = got["documents"][0]
        meta = got["metadatas"][0] if got.get("metadatas") else {}
        return SearchResult(
            doc_id=got["ids"][0],
            chunk_text=doc,
            score=1.0,
            metadata=meta or {},
        )

    def load(self) -> bool:
        """Dummy load for compatibility with VectorRetriever.
        ChromaStore loads automatically on initialization.
        """
        return True

    def count(self) -> int:
        """Return total number of documents in collection."""
        return self.collection.count()
