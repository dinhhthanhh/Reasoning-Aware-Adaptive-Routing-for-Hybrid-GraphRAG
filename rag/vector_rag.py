"""Vector RAG pipeline for single-hop legal question answering.

Embeds queries, retrieves relevant document chunks via FAISS,
and generates answers using the Ollama LLM.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from llm.openai_client import OpenAIClient
from pipeline.i18n import get_template
from vector_store.vector_retriever import VectorRetriever


def _hyde_enabled(config: dict[str, Any]) -> bool:
    """Resolve HyDE flag from rag.use_hyde or top-level hyde.enabled."""
    rag_cfg = config.get("rag", {})
    if "use_hyde" in rag_cfg:
        return bool(rag_cfg["use_hyde"])
    return bool(config.get("hyde", {}).get("enabled", False))


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
    context: str = ""  # Raw text passed to LLM


class VectorRAG:
    """Pipeline RAG for simple single-hop legal questions.

    Combines vector retrieval with LLM generation for straightforward
    legal queries that can be answered from a single document context.
    """

    BASE_SYSTEM_PROMPT_EN = (
        "You are a precise English AI assistant. "
        "Answer the question based strictly on the provided context. "
        "You MUST answer in English."
    )

    BASE_SYSTEM_PROMPT_VI = (
        "Bạn là một chuyên gia pháp lý AI. "
        "Hãy dựa vào ngữ cảnh văn bản được cung cấp để trả lời câu hỏi. "
        "Luôn luôn trả lời bằng tiếng Việt trừ khi được yêu cầu khác."
    )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize Vector RAG pipeline.

        Args:
            config: Full config dict. If None, loads from configs/config.yaml.
        """
        if config is None:
            config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

        self._config = config
        self.retriever = VectorRetriever(config)
        self.llm = OpenAIClient(config.get("openai", config.get("ollama", {})))
        
        try:
            from rag.hyde_generator import HyDEGenerator
            self.hyde = HyDEGenerator(self.llm, enabled=_hyde_enabled(config))
        except Exception as e:
            logger.warning("Failed to initialize HyDEGenerator: {}", e)
            self.hyde = None
            
        try:
            from vector_store.reranker import VNLegalReranker
            self.reranker = VNLegalReranker(config)
        except Exception as e:
            logger.warning("Failed to initialize VNLegalReranker: {}", e)
            self.reranker = None
            
        self.top_k: int = config["faiss"].get("top_k", 5)
        self.language: str = config.get("language", "vi")
        self.task_type: str = config.get("task_type", "short_factoid")
        self.concise_answer: bool = bool(config.get("concise_answer", False))
        rag_cfg = config.get("rag", {})
        self.max_context_chars: int = rag_cfg.get("max_context_chars", 12000)
        self.max_chunk_chars: int = rag_cfg.get("max_chunk_chars", 1500)
        self.retrieval_candidate_multiplier: int = int(rag_cfg.get("retrieval_candidate_multiplier", 3))
        
        from pipeline.i18n import get_answer_format
        answer_task = "legal_eval" if self.concise_answer and self.task_type == "legal_citation" else self.task_type
        self._answer_format = get_answer_format(answer_task)

        logger.info(
            "VectorRAG initialized | top_k={} | language={} | task={}", 
            self.top_k, self.language, self.task_type
        )

    @property
    def SYSTEM_PROMPT(self) -> str:
        """Dynamic system prompt based on language and task."""
        base = self.BASE_SYSTEM_PROMPT_EN if self.language == "en" else self.BASE_SYSTEM_PROMPT_VI
        return f"{base} {self._answer_format['system_suffix']}"

    @property
    def ANSWER_TEMPLATE(self) -> str:
        """Prompt suffix for formatting the answer."""
        return self._answer_format["prompt_suffix"]

    def _build_system_prompt(self) -> str:
        """Construct the system prompt dynamically."""
        return self.SYSTEM_PROMPT

    def _build_prompt(self, context: str, history: str, query: str) -> str:
        """Construct the user prompt dynamically."""
        return f"""Context:
{context}

History:
{history}

Question:
{query}

{self.ANSWER_TEMPLATE}"""

    @staticmethod
    def _is_low_value_chunk(text: str) -> bool:
        """Filter crawled placeholder chunks that add noise but no answer evidence."""
        normalized = " ".join(text.lower().split())
        placeholder_patterns = (
            "đang cập nhật nội dung",
            "dang cap nhat noi dung",
            "văn bản đang được cập nhật",
            "noi dung dang duoc cap nhat",
        )
        return any(normalized.count(pattern) >= 1 for pattern in placeholder_patterns)

    def answer(self, query: str, history: str | None = None) -> RAGResponse:
        """Answer a query using vector retrieval + LLM generation.

        Args:
            query: User question.
            history: Optional conversation history string.

        Returns:
            RAGResponse with answer, sources, scores, and latency.
        """
        start = time.perf_counter()

        search_query = query
        if getattr(self, "hyde", None) and self.hyde.enabled:
            search_query = self.hyde.generate(query)
            
        # Optional metadata filtering
        where_clause = None
        # year_match = re.search(r'\b(19\d{2}|20\d{2})\b', query)
        # if year_match:
        #     # Just a simple example, assuming 'year' is indexed in metadata
        #     where_clause = {"year": int(year_match.group(1))}
            
        # Retrieve relevant chunks
        candidate_k = max(self.top_k, self.top_k * self.retrieval_candidate_multiplier)
        candidates = self.retriever.retrieve(search_query, top_k=candidate_k, where=where_clause)
        results = [result for result in candidates if not self._is_low_value_chunk(result.chunk_text)]
        if not results:
            results = candidates
            
        # Optional Reranking
        if getattr(self, "reranker", None) and self.reranker.enabled and results:
            documents = [res.chunk_text for res in results]
            reranked = self.reranker.rerank(query, documents, top_n=self.top_k)
            # Reorder results based on reranker indices
            results = [results[r["index"]] for r in reranked]
        else:
            results = results[: self.top_k]

        if not results:
            msg = get_template(self.language, "no_context_found")
            return RAGResponse(
                answer=msg,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        context_parts: list[str] = []
        sources: list[str] = []
        scores: list[float] = []

        for i, result in enumerate(results, 1):
            if hasattr(result, "metadata"):
                meta = result.metadata or {}
                raw_doc_id = (
                    meta.get("canonical_id")
                    or meta.get("doc_id")
                    or meta.get("article_id")
                    or meta.get("title")
                    or result.doc_id
                )
            else:
                raw_doc_id = result.doc_id
                meta = {}

            # Canonical chunk IDs (law_number::article_num) — use as-is
            if isinstance(raw_doc_id, str) and "::" in raw_doc_id and re.match(
                r"^[^:]+\::\d", raw_doc_id
            ):
                doc_id = raw_doc_id
            else:
                law_id = str(raw_doc_id)
                for prefix in ("hf_processed_", "phapdien_processed_"):
                    if law_id.startswith(prefix):
                        law_id = law_id[len(prefix):]

                art_match = re.search(
                    r"^(?:\*\*|###\s*|#+\s*)?(\u0110i\u1ec1u\s+\d+[a-zA-Z]*)",
                    result.chunk_text,
                    re.IGNORECASE,
                )
                if art_match:
                    article_id = art_match.group(1).strip()
                    doc_id = f"{law_id}::{article_id}"
                else:
                    doc_id = law_id
                
            title = result.metadata.get("title", f"Document {doc_id}" if self.language == "en" else f"Văn bản {doc_id}") if hasattr(result, "metadata") else f"Văn bản {doc_id}"

            chunk_text = " ".join(result.chunk_text.split())
            if len(chunk_text) > self.max_chunk_chars:
                chunk_text = chunk_text[: self.max_chunk_chars] + "..."

            part = f"[{doc_id}] {title}:\n{chunk_text}"
            current_len = sum(len(p) for p in context_parts)
            if context_parts and current_len + len(part) > self.max_context_chars:
                logger.debug(
                    "Vector context budget reached | budget={} | used={} | skipped_doc={}",
                    self.max_context_chars,
                    current_len,
                    doc_id,
                )
                break

            context_parts.append(part)
            
            if doc_id and doc_id not in sources:
                sources.append(doc_id)
            scores.append(result.score)

        context = "\n\n".join(context_parts)
        history_str = history or ("None" if self.language == "en" else "Không có")

        # Generate answer using dynamic prompts
        system_prompt = self._build_system_prompt()
        prompt = self._build_prompt(context, history_str, query)

        try:
            answer = self.llm.generate(prompt, system_prompt=system_prompt)
            # Use safe duck typing for thinking strip
            strip_fn = getattr(self.llm, "_strip_thinking", lambda x: x)
            answer = answer.strip()
        except Exception as exc:
            logger.error("LLM generation failed: {}", exc)
            raise


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
            context=context,
        )

    def load_index(self) -> bool:
        """Load existing FAISS index from disk."""
        return self.retriever.load_index()

    def index_documents(self, docs_dir: str | Path) -> int:
        """Index documents for retrieval."""
        return self.retriever.index_documents(docs_dir)
