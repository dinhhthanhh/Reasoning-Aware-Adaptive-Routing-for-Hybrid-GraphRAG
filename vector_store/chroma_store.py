"""ChromaDB vector store management."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from loguru import logger

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
        
        # We need embedding function for retrieval in Chroma
        model_name = config.get("model_name", "keepitreal/vietnamese-sbert")
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info("ChromaStore initialized | path={} | collection={}", self.path, self.collection_name)

    def search(self, query_embedding: Any, top_k: int = 5) -> list[SearchResult]:
        """Search ChromaDB using query embedding."""
        # Convert numpy array to list for Chroma
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()
            
        results = self.collection.query(
            query_embeddings=query_embedding if isinstance(query_embedding[0], list) else [query_embedding],
            n_results=top_k
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
                doc_id=meta.get("doc_id", meta.get("article_id", "")),
                chunk_text=doc,
                score=float(score),
                metadata=meta
            ))
            
        return search_results

    def load(self) -> bool:
        """Dummy load for compatibility with VectorRetriever.
        ChromaStore loads automatically on initialization.
        """
        return True

    def count(self) -> int:
        """Return total number of documents in collection."""
        return self.collection.count()
