"""Vector retriever combining embedder and FAISS store.

Handles document chunking, indexing, and top-k retrieval
for the Vector RAG pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from vector_store.embedder import Embedder
from vector_store.faiss_store import FAISSStore, SearchResult


class VectorRetriever:
    """Top-k vector retrieval with chunking support.

    Orchestrates the embedder and FAISS store to index documents
    and retrieve the most relevant chunks for a given query.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize retriever with embedder and FAISS store.

        Args:
            config: Full config dict. If None, loads from configs/config.yaml.
        """
        if config is None:
            config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

        self.embed_config = config["embedding"]
        self.vector_store_type = config.get("vector_store", "faiss")
        
        # Consistent parameters from faiss/chroma section
        vstore_config = config.get(self.vector_store_type, config.get("faiss", {}))
        self.chunk_size: int = vstore_config.get("chunk_size", 512)
        self.chunk_overlap: int = vstore_config.get("chunk_overlap", 64)
        self.top_k: int = vstore_config.get("top_k", 5)

        if self.vector_store_type == "chroma":
            from vector_store.chroma_store import ChromaStore
            self.store = ChromaStore({**config.get("chroma", {}), **self.embed_config, "top_k": self.top_k})
        else:
            self.store = FAISSStore(config["faiss"])
        
        self.embedder = Embedder(self.embed_config)

        logger.info(
            "VectorRetriever initialized | chunk_size={} | overlap={} | top_k={}",
            self.chunk_size,
            self.chunk_overlap,
            self.top_k,
        )

    def chunk_text(self, text: str, doc_id: str = "") -> list[dict[str, Any]]:
        """Split text into overlapping chunks.

        Args:
            text: Full document text.
            doc_id: Document identifier for metadata.

        Returns:
            List of chunk metadata dicts with keys: doc_id, chunk_text, chunk_idx.
        """
        words = text.split()
        chunks: list[dict[str, Any]] = []
        start = 0
        chunk_idx = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append({
                "doc_id": doc_id,
                "chunk_text": chunk_text,
                "chunk_idx": chunk_idx,
            })
            chunk_idx += 1
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def index_documents(self, docs_dir: str | Path) -> int:
        """Index all JSON documents from a directory.

        Reads processed documents, chunks them, embeds, and adds to FAISS.

        Args:
            docs_dir: Path to directory containing JSON document files.

        Returns:
            Total number of chunks indexed.
        """
        docs_path = Path(docs_dir)
        if not docs_path.exists():
            logger.error("Documents directory not found: {}", docs_path)
            return 0

        all_chunks: list[dict[str, Any]] = []
        all_texts: list[str] = []

        json_files = sorted(docs_path.glob("*.json"))
        logger.info("Indexing {} document files from {}", len(json_files), docs_path)

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    doc = json.load(f)

                doc_id = doc.get("doc_id", json_file.stem)
                content = doc.get("content", "")
                if not content:
                    continue

                # Add title to content for better retrieval
                title = doc.get("title", "")
                if title:
                    content = f"{title}\n{content}"

                chunks = self.chunk_text(content, doc_id=doc_id)
                for chunk in chunks:
                    chunk["title"] = title
                    chunk["source_url"] = doc.get("source_url", "")
                    chunk["law_name"] = doc.get("law_name", "")

                all_chunks.extend(chunks)
                all_texts.extend(c["chunk_text"] for c in chunks)
            except (json.JSONDecodeError, IOError) as exc:
                logger.warning("Failed to process {}: {}", json_file, exc)

        if not all_texts:
            logger.warning("No texts to index")
            return 0

        # Batch embed
        logger.info("Embedding {} chunks...", len(all_texts))
        embeddings = self.embedder.encode(all_texts, show_progress=True)

        # Add to FAISS
        self.store.add(embeddings, all_chunks)
        self.store.save()

        logger.info("Indexed {} chunks from {} documents", len(all_chunks), len(json_files))
        return len(all_chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Retrieve top-k most relevant chunks for a query.

        Args:
            query: Search query text.
            top_k: Number of results. Defaults to config top_k.

        Returns:
            List of SearchResult with chunks and scores.
        """
        query_emb = self.embedder.encode_single(query)
        results = self.store.search(query_emb, top_k=top_k or self.top_k)
        logger.debug(
            "Retrieved {} chunks for query: {}",
            len(results),
            query[:80],
        )
        return results

    def load_index(self) -> bool:
        """Load a previously saved FAISS index.

        Returns:
            True if loading succeeded.
        """
        return self.store.load()
