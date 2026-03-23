"""Vector RAG pipeline for single-hop legal question answering.

Embeds queries, retrieves relevant document chunks via FAISS,
and generates answers using the Ollama LLM.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from llm.ollama_client import OllamaClient
from vector_store.vector_retriever import VectorRetriever


@dataclass
class RAGResponse:
    """Response from the Vector RAG pipeline.

    Attributes:
        answer: Generated answer text.
        sources: Document IDs used as context.
        retrieval_scores: Similarity scores for retrieved chunks.
        latency_ms: Total pipeline latency in milliseconds.
    """

    answer: str
    sources: list[str] = field(default_factory=list)
    retrieval_scores: list[float] = field(default_factory=list)
    latency_ms: float = 0.0


class VectorRAG:
    """Pipeline RAG for simple single-hop legal questions.

    Combines vector retrieval with LLM generation for straightforward
    legal queries that can be answered from a single document context.
    """

    SYSTEM_PROMPT = (
        "Bạn là trợ lý pháp luật Việt Nam. BẮT BUỘC trả lời hoàn toàn bằng tiếng Việt. "
        "Trả lời câu hỏi dựa trên ngữ cảnh được cung cấp. "
        "Nếu ngữ cảnh không đủ thông tin, hãy nói rõ bằng tiếng Việt. "
        "Tuyệt đối không sử dụng tiếng Anh trong câu trả lời."
    )

    ANSWER_TEMPLATE = """## Ngữ cảnh:
{context}

## Lịch sử hội thoại:
{history}

## Câu hỏi:
{query}

Hãy trả lời câu hỏi trên dựa vào ngữ cảnh được cung cấp. Trích dẫn điều luật cụ thể nếu có."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize Vector RAG pipeline.

        Args:
            config: Full config dict. If None, loads from configs/config.yaml.
        """
        if config is None:
            config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

        self.retriever = VectorRetriever(config)
        self.llm = OllamaClient(config["ollama"])
        self.top_k: int = config["faiss"].get("top_k", 5)

        logger.info("VectorRAG initialized | top_k={}", self.top_k)

    def answer(self, query: str, history: str | None = None) -> RAGResponse:
        """Answer a query using vector retrieval + LLM generation.

        Pipeline:
        1. Embed query
        2. FAISS search top-k
        3. Build context from retrieved chunks
        4. Generate answer with Ollama
        5. Return answer with sources

        Args:
            query: User question.
            history: Optional conversation history string.

        Returns:
            RAGResponse with answer, sources, scores, and latency.
        """
        start = time.perf_counter()

        # Retrieve relevant chunks
        results = self.retriever.retrieve(query, top_k=self.top_k)

        if not results:
            return RAGResponse(
                answer="Không tìm thấy tài liệu liên quan trong cơ sở dữ liệu.",
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        # Build context from retrieved chunks
        context_parts: list[str] = []
        sources: list[str] = []
        scores: list[float] = []

        for i, result in enumerate(results, 1):
            context_parts.append(f"[{i}] {result.chunk_text}")
            if result.doc_id and result.doc_id not in sources:
                sources.append(result.doc_id)
            scores.append(result.score)

        context = "\n\n".join(context_parts)
        history_str = history or "Không có"

        # Generate answer
        prompt = self.ANSWER_TEMPLATE.format(
            context=context,
            history=history_str,
            query=query,
        )

        try:
            answer = self.llm.generate(prompt, system_prompt=self.SYSTEM_PROMPT)
        except RuntimeError as exc:
            logger.error("LLM generation failed: {}", exc)
            answer = f"Lỗi khi tạo câu trả lời: {exc}"

        latency = (time.perf_counter() - start) * 1000

        logger.info(
            "VectorRAG answered | sources={} | latency={:.0f}ms",
            len(sources),
            latency,
        )

        return RAGResponse(
            answer=answer,
            sources=sources,
            retrieval_scores=scores,
            latency_ms=latency,
        )

    def load_index(self) -> bool:
        """Load existing FAISS index from disk.

        Returns:
            True if the index was loaded successfully.
        """
        return self.retriever.load_index()

    def index_documents(self, docs_dir: str | Path) -> int:
        """Index documents for retrieval.

        Args:
            docs_dir: Path to processed document directory.

        Returns:
            Number of chunks indexed.
        """
        return self.retriever.index_documents(docs_dir)
