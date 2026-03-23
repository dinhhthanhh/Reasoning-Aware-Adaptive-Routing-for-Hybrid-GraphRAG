"""Graph RAG adapter combining Neo4j and GraphRAG.

Provides multi-hop reasoning by combining local Neo4j graph queries
with Microsoft GraphRAG. Falls back to Neo4j-only when GraphRAG
is unavailable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from graph.neo4j_client import Neo4jClient
from graphrag_wrapper.graphrag_client import GraphRAGClient
from llm.ollama_client import OllamaClient
from ner.vi_ner import ViNER


@dataclass
class GraphRAGResponse:
    """Response from the Graph RAG adapter.

    Attributes:
        answer: Generated answer text.
        sources: Document/entity identifiers used.
        graph_context: Relevant triples from Neo4j.
        latency_ms: Total pipeline latency.
    """

    answer: str
    sources: list[str] = field(default_factory=list)
    graph_context: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0


class GraphRAGAdapter:
    """Adapter for multi-hop graph-based legal question answering.

    Combines Neo4j knowledge graph traversal with Microsoft GraphRAG
    for complex legal reasoning across multiple documents and articles.
    """

    SYSTEM_PROMPT = (
        "Bạn là trợ lý pháp luật Việt Nam chuyên về phân tích đa bước. "
        "BẮT BUỘC trả lời hoàn toàn bằng tiếng Việt. "
        "Sử dụng thông tin từ đồ thị tri thức để trả lời câu hỏi phức tạp. "
        "Nếu thông tin ngữ cảnh là tiếng Anh, hãy DỊCH sang tiếng Việt trong câu trả lời."
    )

    GRAPH_ANSWER_TEMPLATE = """## Thông tin từ đồ thị tri thức:
{graph_context}

## Kết quả GraphRAG (nếu có):
{graphrag_context}

## Lịch sử hội thoại:
{history}

## Câu hỏi:
{query}

HÃY TRẢ LỜI HOÀN TOÀN BẰNG TIẾNG VIỆT. Phân tích mối quan hệ giữa các điều luật và trả lời câu hỏi. 
Nếu các kết quả GraphRAG ở trên là tiếng Anh, bạn CẦN dịch và tổng hợp lại bằng tiếng Việt chuyên nghiệp."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize Graph RAG adapter.

        Args:
            config: Full config dict. If None, loads from configs/config.yaml.
        """
        if config is None:
            config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

        self.neo4j = Neo4jClient(config["neo4j"])
        self.llm = OllamaClient(config["ollama"])
        self.ner = ViNER(config["ner"])

        # GraphRAG is optional — falls back to Neo4j-only
        try:
            self.graphrag = GraphRAGClient(config)
        except Exception as exc:
            logger.warning("GraphRAG not available, using Neo4j-only: {}", exc)
            self.graphrag = None

        logger.info(
            "GraphRAGAdapter initialized | graphrag_available={}",
            self.graphrag is not None and self.graphrag.is_available(),
        )

    def answer(self, query: str, history: str | None = None) -> GraphRAGResponse:
        """Answer a complex query using graph-based multi-hop reasoning.

        Pipeline:
        1. Extract entities from query using NER
        2. Query Neo4j for relevant subgraph (neighbors)
        3. Optionally query GraphRAG for additional context
        4. Combine contexts and generate answer with LLM

        Args:
            query: Complex legal question.
            history: Optional conversation history string.

        Returns:
            GraphRAGResponse with answer and graph context.
        """
        start = time.perf_counter()

        # Step 1: Extract entities from query
        entities_list = self.ner.extract([query])
        entities = entities_list[0] if entities_list else []
        entity_names = [e.text for e in entities if len(e.text) > 1]

        logger.debug("Graph query entities: {}", entity_names)

        # Step 2: Query Neo4j for related triples
        graph_triples: list[dict[str, Any]] = []
        sources: list[str] = []

        for entity_name in entity_names[:5]:  # Limit to avoid over-querying
            try:
                neighbors = self.neo4j.query_neighbors(entity_name, depth=2)
                graph_triples.extend(neighbors)
                for n in neighbors:
                    for key in ("source", "target"):
                        if n[key] not in sources:
                            sources.append(n[key])
            except Exception as exc:
                logger.warning("Neo4j query failed for '{}': {}", entity_name, exc)

        # Format graph context
        if graph_triples:
            graph_context = "\n".join(
                f"- {t['source']} --[{t['relation']}]--> {t['target']}"
                for t in graph_triples[:20]  # Limit context size
            )
        else:
            graph_context = "Không tìm thấy thông tin trong đồ thị tri thức."

        # Step 3: Query GraphRAG if available
        graphrag_context = "Không khả dụng"
        if self.graphrag and self.graphrag.is_available():
            try:
                graphrag_answer = self.graphrag.answer(query, history=history)
                graphrag_context = graphrag_answer
            except RuntimeError as exc:
                logger.warning("GraphRAG query failed: {}", exc)
                graphrag_context = f"Lỗi: {exc}"

        # Step 4: Generate answer using LLM
        prompt = self.GRAPH_ANSWER_TEMPLATE.format(
            graph_context=graph_context,
            graphrag_context=graphrag_context,
            history=history or "Không có",
            query=query,
        )

        try:
            answer = self.llm.generate(prompt, system_prompt=self.SYSTEM_PROMPT)
        except RuntimeError as exc:
            logger.error("LLM generation failed: {}", exc)
            answer = f"Lỗi khi tạo câu trả lời: {exc}"

        latency = (time.perf_counter() - start) * 1000

        logger.info(
            "GraphRAGAdapter answered | entities={} | triples={} | latency={:.0f}ms",
            len(entity_names),
            len(graph_triples),
            latency,
        )

        return GraphRAGResponse(
            answer=answer,
            sources=sources[:10],
            graph_context=graph_triples[:20],
            latency_ms=latency,
        )
