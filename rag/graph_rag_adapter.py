"""Adapter for Graph RAG operations.

Integrates Neo4j/SQLite graph retrieval with LLM generation,
supporting multi-hop reasoning by traversal.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from loguru import logger

from graph.neo4j_client import NO_GRAPH_CONTEXT, Neo4jClient
from graph.sqlite_kg import SQLiteKG
from llm.openai_client import OpenAIClient
from pipeline.i18n import get_template
from ner.factory import get_ner_model


class GraphRAGAdapter:
    """Adapter for graph-based RAG operations.
    
    Orchestrates entity extraction, graph traversal, and response synthesis.
    """

    BASE_SYSTEM_PROMPT_EN = (
        "You are a precise English AI assistant. "
        "Answer based on the provided knowledge graph context. "
        "You MUST answer in English."
    )

    BASE_SYSTEM_PROMPT_VI = (
        "Bạn là trợ lý AI chuyên nghiệp. "
        "Trả lời dựa trên ngữ cảnh đồ thị tri thức được cung cấp. "
        "BẮT BUỘC trả lời bằng tiếng Việt."
    )

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize GraphRAG adapter.
        
        Args:
            config: Full config dict.
        """
        self.language = config.get("language", "vi")
        self.task_type = config.get("task_type", "short_factoid")
        self.concise_answer = bool(config.get("concise_answer", False))
        rag_cfg = config.get("rag", {})
        self.graph_top_k = int(rag_cfg.get("graph_top_k", 4))
        self.graph_context_chars = int(rag_cfg.get("graph_context_chars", 12000))
        self.llm = OpenAIClient(config.get("openai", config.get("ollama", {})))
        
        from pipeline.i18n import get_answer_format
        answer_task = "legal_eval" if self.concise_answer and self.task_type == "legal_citation" else self.task_type
        self._answer_format = get_answer_format(answer_task)

        # Initialize knowledge graphs
        try:
            self._neo4j_client = Neo4jClient(config.get("neo4j", {}))
        except Exception as e:
            logger.warning("Failed to initialize Neo4j: {}", e)
            self._neo4j_client = None

        kg_dir = config.get("data", {}).get("kg_dir", "data/kg")
        kg_filename = config.get("data", {}).get("kg_filename", "legal_kg.db")
        kg_path = Path(kg_dir) / kg_filename
        self._sqlite_kg = SQLiteKG(kg_path, language=self.language)
        
        # NER for entity extraction
        self.ner_model = get_ner_model(config.get("ner", {}))
        
        logger.info(
            "GraphRAGAdapter initialized | language={} | task={}", 
            self.language, self.task_type
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
        return f"""Knowledge Graph Context:
{context}

Query:
{query}

History:
{history or 'None'}

{self.ANSWER_TEMPLATE}"""

    @staticmethod
    def _extract_sources_from_context(context: str, limit: int = 12) -> list[str]:
        """Extract compact legal source identifiers from formatted graph context."""
        patterns = (
            r"\b\d{1,4}/(?:\d{4}/)?[A-ZĐ]{1,12}(?:-[A-ZĐ]{1,12}){0,4}\b",
            r"\bpd_\d{3}_\d{3}_\d{4}\b",
        )
        sources: list[str] = []
        for pattern in patterns:
            for match in re.findall(pattern, context):
                if match not in sources:
                    sources.append(match)
                    if len(sources) >= limit:
                        return sources
        return sources

    def answer(self, query: str, history: str | None = None) -> dict[str, Any]:
        """Synthesize an answer using graph traversal.
        
        Args:
            query: User query.
            history: Optional history.
            
        Returns:
            Dict containing answer, sources, and metadata.
        """
        start = time.perf_counter()
        
        # Step 1: Extract entities and search graph
        entities_list = self.ner_model.extract([query])
        entities = entities_list[0] if entities_list else []
        graph_context_str = ""
        sources = []
        kg_source = "sqlite"

        if self._neo4j_client and self._neo4j_client.verify_connection():
            # Try Neo4j first
            try:
                graph_context_str = self._neo4j_client.get_multi_hop_context(query, top_k=self.graph_top_k)
                if graph_context_str and NO_GRAPH_CONTEXT not in graph_context_str:
                    kg_source = "neo4j"
                    sources = self._extract_sources_from_context(graph_context_str)
                else:
                    graph_context_str = ""
            except Exception as e:
                logger.warning("Neo4j traversal failed: {}", e)

        if not graph_context_str:
            # Fallback to SQLite
            graph_context_str = self._sqlite_kg.multi_hop_context(query, top_k=3)
            nodes = self._sqlite_kg.search_nodes(query, top_k=4)
            sources = [n.get("article_id", "") for n in nodes if n.get("article_id")]

        no_graph_msg = get_template(self.language, "no_graph_found")
        if not graph_context_str or no_graph_msg in graph_context_str or NO_GRAPH_CONTEXT in graph_context_str:
            graph_context_str = no_graph_msg
            kg_source = "none"
        elif len(graph_context_str) > self.graph_context_chars:
            graph_context_str = graph_context_str[: self.graph_context_chars] + "\n[Context truncated]"

        # Step 2: Generate answer using dynamic prompts
        system_prompt = self._build_system_prompt()
        prompt = self._build_prompt(graph_context_str, history, query)

        try:
            answer = self.llm.generate(prompt, system_prompt=system_prompt)
            # Use safe duck typing for thinking strip
            strip_fn = getattr(self.llm, "_strip_thinking", lambda x: x)
            answer = strip_fn(answer)
            # HotpotQA optimization: Normalize answer for higher EM
            answer = answer.strip().replace(".", "").replace('"', "").replace("'", "")
        except RuntimeError as exc:
            logger.error("LLM generation failed: {}", exc)
            # Return raw context as fallback answer
            answer = get_template(self.language, "fallback_llm_fail", context=graph_context_str[:1000])

        latency = (time.perf_counter() - start) * 1000

        return {
            "answer": answer,
            "sources": sources,
            "kg_source": kg_source,
            "latency_ms": latency,
            "context": graph_context_str
        }
