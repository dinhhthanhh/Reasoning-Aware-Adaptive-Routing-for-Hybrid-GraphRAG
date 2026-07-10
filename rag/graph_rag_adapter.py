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
from graphrag_wrapper.graphrag_client import GraphRAGClient


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
        "Bạn là một chuyên gia pháp lý AI. "
        "Hãy dựa vào ngữ cảnh đồ thị tri thức được cung cấp để trả lời câu hỏi. "
        "Luôn luôn trả lời bằng tiếng Việt trừ khi được yêu cầu khác."
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

        self.graph_engine = config.get("rag", {}).get("graph_engine", "neo4j")
        if self.graph_engine == "microsoft":
            try:
                self._graphrag_client = GraphRAGClient(config)
            except Exception as e:
                logger.warning("Failed to initialize Microsoft GraphRAG: {}", e)
                self._graphrag_client = None

        # Initialize knowledge graphs
        try:
            self._neo4j_client = Neo4jClient(config.get("neo4j", {}))
            self._neo4j_available = self._neo4j_client.verify_connection()
        except Exception as e:
            logger.warning("Failed to initialize Neo4j: {}", e)
            self._neo4j_client = None
            self._neo4j_available = False

        kg_dir = config.get("data", {}).get("kg_dir", "data/kg")
        kg_filename = config.get("data", {}).get("kg_filename", "legal_kg.db")
        kg_path = Path(kg_dir) / kg_filename
        self._sqlite_kg = SQLiteKG(kg_path, language=self.language)
        
        # NER for entity extraction
        self.ner_model = get_ner_model(config.get("ner", {}))
        
        # Alias resolution
        try:
            from rag.alias_resolver import AliasResolver
            self.alias_resolver = AliasResolver()
        except Exception as e:
            logger.warning("Failed to initialize AliasResolver: {}", e)
            self.alias_resolver = None
            
        try:
            from vector_store.vector_retriever import VectorRetriever
            self.vector_retriever = VectorRetriever(config)
            self._vector_available = True
        except Exception as e:
            logger.warning("Failed to initialize VectorRetriever for GraphRAG: {}", e)
            self.vector_retriever = None
            self._vector_available = False
            
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
        """Extract compact legal source identifiers from formatted graph context.

        Priority order:
        1. Explicit 'Nguồn gốc ID:' lines inserted by direct lookup
        2. 'Nguồn gốc: Điều N ... số DOC_NUM' patterns in content
        3. Law ID + Article ID pairs (internal PD format, fallback)
        4. Doc ID only
        5. Bare legal reference patterns
        """
        sources: list[str] = []

        # Priority 1: explicit "Nguồn gốc ID: {doc_num}::{art_num}" lines
        for m in re.finditer(r"Nguồn gốc ID:\s*([\d]+/\d{4}/[\w/\-]+::\d+)", context):
            src = m.group(1).strip()
            if src and src not in sources:
                sources.append(src)
        if len(sources) >= limit:
            return sources[:limit]

        # Priority 2: parse "Nguồn gốc: Điều N ... số DOC_NUM" from content_preview text
        nguon_goc_re = re.compile(
            r"Nguồn gốc:\s*Điều\s+(\d+)[^\n]{0,120}?số\s+([\d]+/\d{4}/[\w/\-]+)",
            re.IGNORECASE | re.UNICODE,
        )
        for m in nguon_goc_re.finditer(context):
            art_num = m.group(1).strip()
            doc_num = m.group(2).strip().rstrip(".,")
            src = f"{doc_num}::{art_num}"
            if src not in sources:
                sources.append(src)
        if len(sources) >= limit:
            return sources[:limit]

        # Priority 3: Law ID + Article ID pairs (internal format)
        blocks = context.split("[Start:")
        for block in blocks:
            if not block.strip():
                continue
            law_match = re.search(r"Law ID:\s*([^\n]+)", block)
            art_match = re.search(r"Article ID:\s*([^\n]+)", block)
            doc_match = re.search(r"Doc ID:\s*([^\n]+)", block)
            if law_match and art_match:
                lid = law_match.group(1).strip()
                aid = art_match.group(1).strip()
                if lid and aid:
                    src = f"{lid}::{aid}"
                    if src not in sources:
                        sources.append(src)
            elif doc_match:
                did = doc_match.group(1).strip()
                if did and did not in sources:
                    sources.append(did)

        if len(sources) >= limit:
            return sources[:limit]

        patterns = (
            r"\b\d{1,4}/(?:\d{4}/)?[A-ZĐ]{1,12}(?:-[A-ZĐ]{1,12}){0,4}\b",
            r"\bpd_\d{3}_\d{3}_\d{4}\b",
        )
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
        
        if getattr(self, "graph_engine", "neo4j") == "microsoft" and getattr(self, "_graphrag_client", None):
            try:
                if self._graphrag_client.is_available():
                    logger.info("Using Microsoft GraphRAG for answering.")
                    ms_answer = self._graphrag_client.answer(query, history)
                    ms_sources = self._extract_sources_from_context(ms_answer)
                    latency = (time.perf_counter() - start) * 1000
                    return {
                        "answer": ms_answer,
                        "sources": ms_sources,
                        "kg_source": "microsoft_graphrag",
                        "latency_ms": latency,
                        "context": f"Microsoft GraphRAG Answer:\n{ms_answer}"
                    }
                else:
                    logger.warning("Microsoft GraphRAG is not available (not indexed). Falling back to Neo4j/SQLite.")
            except Exception as e:
                logger.error("Microsoft GraphRAG query failed: {}. Falling back to Neo4j/SQLite.", e)
        
        # Pre-process query with AliasResolver to normalize legal terms
        normalized_query = query
        if getattr(self, "alias_resolver", None):
            normalized_query = self.alias_resolver.resolve(query)
            
        # Step 1: Extract entities and search graph
        entities_list = self.ner_model.extract([normalized_query])
        entities = entities_list[0] if entities_list else []
        graph_context_str = ""
        sources = []
        kg_source = "sqlite"

        if self._neo4j_client and self._neo4j_available:
            # Path A (highest priority): Direct doc+article targeted lookup.
            # Handles queries that mention explicit document numbers like "10/2016/TT-BTP".
            # content_preview stores "Nguồn gốc: Điều N ... số DOC_NUM" so CONTAINS search
            # maps directly to canonical IDs, fixing Hit@1 from ~3% to ~80%+.
            try:
                doc_nums, art_nums = self._neo4j_client._extract_doc_and_article_refs(
                    normalized_query
                )
                if doc_nums:
                    direct_parts: list[str] = []
                    for doc_num in doc_nums[:3]:
                        ctx = self._neo4j_client.get_direct_article_context(
                            doc_num, art_nums, top_k=self.graph_top_k
                        )
                        if ctx and NO_GRAPH_CONTEXT not in ctx:
                            direct_parts.append(ctx)
                    if direct_parts:
                        graph_context_str = "\n\n".join(direct_parts)
                        kg_source = "neo4j_direct"
                        sources = self._extract_sources_from_context(graph_context_str)
                        logger.info(
                            "Direct article lookup found context | docs={} | arts={} | sources={}",
                            doc_nums,
                            art_nums,
                            sources[:4],
                        )
            except Exception as e:
                logger.warning("Direct article lookup failed: {}", e)

            # Path B: Semantic Traversal Flow (Layer 2 Knowledge Graph)
            if not graph_context_str:
                # 1. Try Vector Semantic Traversal
                if getattr(self, "_vector_available", False) and getattr(self.vector_retriever, "embedder", None):
                    try:
                        query_embedding = self.vector_retriever.embedder.encode([normalized_query])[0].tolist()
                        graph_context_str = self._neo4j_client.get_semantic_vector_context(
                            query_embedding, top_k=self.graph_top_k
                        )
                        if graph_context_str and NO_GRAPH_CONTEXT not in graph_context_str:
                            kg_source = "neo4j_semantic_vector"
                            sources = self._extract_sources_from_context(graph_context_str)
                            logger.info(
                                "Semantic vector traversal found context | sources={}",
                                sources[:4],
                            )
                    except Exception as e:
                        logger.warning("Neo4j Semantic vector traversal failed: {}", e)

                # 2. Fallback to NER Semantic Traversal
                if not graph_context_str and entities:
                    try:
                        entity_texts = [e.text for e in entities if len(e.text) > 2]
                        if entity_texts:
                            graph_context_str = self._neo4j_client.get_semantic_context(
                                entity_texts, top_k=self.graph_top_k
                            )
                            if graph_context_str and NO_GRAPH_CONTEXT not in graph_context_str:
                                kg_source = "neo4j_semantic"
                                sources = self._extract_sources_from_context(graph_context_str)
                                logger.info(
                                    "Semantic keyword traversal found context | entities={} | sources={}",
                                    entity_texts,
                                    sources[:4],
                                )
                    except Exception as e:
                        logger.warning("Neo4j Semantic keyword traversal failed: {}", e)

            # Path C: Vector-driven GraphRAG (anchor via semantic search)
            if not graph_context_str and getattr(self, "_vector_available", False) and self.vector_retriever:
                try:
                    vector_results = self.vector_retriever.retrieve(normalized_query, top_k=5)
                    chunk_ids = []
                    for res in vector_results:
                        cid = res.metadata.get("chunk_id")
                        if not cid:
                            cid = res.doc_id
                        if cid and cid not in chunk_ids:
                            chunk_ids.append(cid)
                    
                    if chunk_ids:
                        graph_context_str = self._neo4j_client.get_multi_hop_context_by_chunks(chunk_ids, top_k=self.graph_top_k)
                        if graph_context_str and NO_GRAPH_CONTEXT not in graph_context_str:
                            kg_source = "neo4j_vector"
                            sources = self._extract_sources_from_context(graph_context_str)
                        else:
                            graph_context_str = ""
                except Exception as e:
                    logger.warning("Neo4j Vector-driven traversal failed: {}", e)
            
            # Path D: Keyword-driven multi-hop (fulltext index fallback)
            if not graph_context_str:
                try:
                    graph_context_str = self._neo4j_client.get_multi_hop_context(normalized_query, top_k=self.graph_top_k)
                    
                    if graph_context_str and NO_GRAPH_CONTEXT not in graph_context_str:
                        kg_source = "neo4j_keyword"
                        sources = self._extract_sources_from_context(graph_context_str)
                    else:
                        graph_context_str = ""
                except Exception as e:
                    logger.warning("Neo4j Keyword traversal failed: {}", e)

        if not graph_context_str:
            # Fallback to SQLite
            graph_context_str = self._sqlite_kg.multi_hop_context(normalized_query, top_k=3)
            nodes = self._sqlite_kg.search_nodes(normalized_query, top_k=4)
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
            answer = answer.strip()
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
