"""Hybrid pipeline orchestrator — main entry point for all queries.

Routes queries through the two-stage router to the appropriate
RAG pipeline (Vector, Graph, or Clarify), manages conversation
history, and produces structured responses.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from llm.openai_client import OpenAIClient
from pipeline.conversation_manager import ConversationManager
from graph.neo4j_client import NO_GRAPH_CONTEXT
from rag.graph_rag_adapter import GraphRAGAdapter
from rag.vector_rag import VectorRAG
from router.ambiguity_detector import AmbiguityDetector
from router.two_stage_router import TwoStageRouter
from pipeline.i18n import get_template


@dataclass
class PipelineResponse:
    """Complete response from the hybrid pipeline.

    Attributes:
        answer: Generated answer text.
        route_used: Which pipeline was used.
        confidence: Router confidence in the route.
        router_reasoning: Explanation of routing decision.
        stage1_route: Stage 1 route before any post-processing.
        stage2_invoked: Whether LLM verifier was used.
        stage2_override: Whether LLM verifier changed the route.
        sources: Document IDs referenced in the answer.
        latency_ms: Total pipeline latency.
        is_ambiguous: Whether the query was deemed ambiguous.
        actual_pipeline_used: The actual RAG pipeline that served the answer (handles fallbacks).
    """

    answer: str = ""
    route_used: str = ""
    confidence: float = 0.0
    router_reasoning: str = ""
    stage1_route: str = ""
    stage2_invoked: bool = False
    stage2_override: bool = False
    sources: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    is_ambiguous: bool = False
    context: str = ""
    actual_pipeline_used: str = ""
    kg_source: str = ""
    resolved_query: str = ""


class HybridPipeline:
    """Main orchestrator for the Vietnamese Legal QA system.

    Integrates:
    - TwoStageRouter for adaptive query routing
    - VectorRAG for single-hop retrieval
    - GraphRAGAdapter for multi-hop reasoning
    - AmbiguityDetector for clarification generation
    - ConversationManager for multi-turn context
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        """Initialize hybrid pipeline from config file.

        Args:
            config_path: Path to config.yaml. Defaults to configs/config.yaml.
        """
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"

        config_path = Path(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

        # Initialize components
        logger.info("Initializing HybridPipeline...")

        self.router = TwoStageRouter(self._config)
        self.conversation_manager = ConversationManager(self._config.get("conversation"))
        self.ambiguity_detector = AmbiguityDetector(self._config.get("ambiguity"))

        # RAG pipelines (lazy init to save memory)
        self._vector_rag: VectorRAG | None = None
        self._graph_rag: GraphRAGAdapter | None = None

        # Logging
        self.log_path = Path(self._config["logging"]["routing_log_path"])
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("HybridPipeline initialized")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "HybridPipeline":
        """Create a pipeline directly from an in-memory config dict.

        This is used by benchmark runners that need to compare variants such as
        single-stage routing without writing temporary config files.
        """
        instance = cls.__new__(cls)
        instance._config = config

        logger.info("Initializing HybridPipeline from in-memory config...")
        instance.router = TwoStageRouter(instance._config)
        instance.conversation_manager = ConversationManager(instance._config.get("conversation"))
        instance.ambiguity_detector = AmbiguityDetector(instance._config.get("ambiguity"))
        instance._vector_rag = None
        instance._graph_rag = None
        instance.log_path = Path(instance._config["logging"]["routing_log_path"])
        instance.log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("HybridPipeline initialized from in-memory config")
        return instance

    @property
    def vector_rag(self) -> VectorRAG:
        """Lazy-initialize VectorRAG pipeline."""
        if self._vector_rag is None:
            self._vector_rag = VectorRAG(self._config)
            self._vector_rag.load_index()
        return self._vector_rag

    @property
    def graph_rag(self) -> GraphRAGAdapter:
        """Lazy-initialize GraphRAGAdapter pipeline."""
        if self._graph_rag is None:
            self._graph_rag = GraphRAGAdapter(self._config)
        return self._graph_rag

    def query(
        self,
        query: str,
        session_id: str = "default",
        verbose: bool = False,
    ) -> PipelineResponse:
        """Process a query through the full hybrid pipeline.

        Flow:
        1. Resolve coreference with conversation history
        2. Run TwoStageRouter to determine route
        3. Branch to appropriate pipeline:
           - 'vector' → VectorRAG.answer()
           - 'graph'  → GraphRAGAdapter.answer()
           - 'clarify' → Generate clarification question
        4. Update ConversationManager
        5. Log routing decision
        6. Return PipelineResponse

        Args:
            query: Vietnamese legal question.
            session_id: Session identifier for conversation tracking.
            verbose: Whether to print detailed debug info.

        Returns:
            PipelineResponse with answer and metadata.
        """
        start = time.perf_counter()

        # Step 1: Resolve coreference
        history_turns = self.conversation_manager.get_history(session_id)
        resolved_query = self.conversation_manager.resolve_coreference(query, history_turns)

        history_str = self.conversation_manager.get_history_string(session_id)

        if verbose and resolved_query != query:
            logger.info("Coreference: '{}' → '{}'", query[:60], resolved_query[:60])

        # Step 2: Route the query
        router_output = self.router.route(
            query=resolved_query,
            history=history_str,
            session_id=session_id,
        )

        if verbose:
            logger.info(
                "Router decision: route={} | confidence={:.3f} | "
                "stage2={} | reasoning={}",
                router_output.route,
                router_output.confidence,
                router_output.stage2_invoked,
                router_output.reasoning,
            )

        # Step 3: Execute appropriate pipeline
        answer = ""
        sources: list[str] = []
        context: str = ""
        kg_source = ""

        if router_output.route == "dense_retrieval":
            answer, sources, context = self._handle_vector(resolved_query, history_str)
            actual_pipeline = "dense_retrieval"
            if (
                self._config.get("rag", {}).get("vector_fallback_to_graph", True)
                and self._is_uninformative_answer(answer)
            ):
                logger.info("Vector answer lacks evidence; falling back to GraphRAG answer")
                graph_answer, graph_sources, graph_context, graph_pipeline, graph_kg_source = self._handle_graph(
                    resolved_query, history_str
                )
                if not self._is_uninformative_answer(graph_answer):
                    answer = graph_answer
                    sources = graph_sources
                    context = graph_context
                    kg_source = graph_kg_source
                    actual_pipeline = f"dense_retrieval->{graph_pipeline}"
        elif router_output.route == "graph_traversal":
            answer, sources, context, actual_pipeline, kg_source = self._handle_graph(
                resolved_query, history_str
            )
        elif router_output.route == "hybrid_reasoning":
            answer, sources, context, kg_source = self._handle_hybrid(resolved_query, history_str)
            actual_pipeline = f"hybrid_reasoning:{kg_source}" if kg_source else "hybrid_reasoning"
        elif router_output.route == "clarify":
            answer = self._handle_clarify(resolved_query, history_str)
            context = ""
            actual_pipeline = "clarify"
        else:
            logger.warning("Unknown route '{}', defaulting to dense_retrieval", router_output.route)
            answer, sources, context = self._handle_vector(resolved_query, history_str)
            actual_pipeline = "dense_retrieval"

        latency_ms = (time.perf_counter() - start) * 1000

        # Step 4: Update conversation
        self.conversation_manager.add_turn(
            session_id=session_id,
            query=query,
            response=answer,
            route=router_output.route,
        )

        # Step 5: Log
        self._log_pipeline(
            query=query,
            resolved_query=resolved_query,
            router_output=router_output,
            answer=answer,
            latency_ms=latency_ms,
            session_id=session_id,
        )

        # Step 6: Build response
        response = PipelineResponse(
            answer=answer,
            route_used=router_output.route,
            confidence=router_output.confidence,
            router_reasoning=router_output.reasoning,
            stage1_route=router_output.stage1_route,
            stage2_invoked=router_output.stage2_invoked,
            stage2_override=router_output.stage2_override,
            sources=sources,
            latency_ms=latency_ms,
            is_ambiguous=router_output.is_ambiguous,
            context=context,
            actual_pipeline_used=actual_pipeline,
            kg_source=kg_source,
            resolved_query=resolved_query,
        )

        logger.info(
            "Pipeline complete | route={} | latency={:.0f}ms | answer_len={}",
            response.route_used,
            response.latency_ms,
            len(response.answer),
        )

        return response

    def _handle_vector(self, query: str, history: str) -> tuple[str, list[str], str]:
        """Handle query via Vector RAG pipeline.

        Args:
            query: Resolved query.
            history: Formatted conversation history.

        Returns:
            Tuple of (answer, sources, context).
        """
        try:
            result = self.vector_rag.answer(query, history=history)
            ctx = getattr(result, "context", "")
            return result.answer, result.sources, ctx
        except Exception as exc:
            logger.error("VectorRAG failed: {}", exc)
            lang = self._config.get("language", "vi")
            err_msg = f"Error during lookup: {exc}" if lang == "en" else f"Lỗi khi tra cứu: {exc}"
            return err_msg, [], ""

    @staticmethod
    def _is_uninformative_answer(answer: str) -> bool:
        """Detect answers that say retrieval context did not contain evidence."""
        text = " ".join(str(answer).lower().split())
        patterns = (
            "không có thông tin",
            "không tìm thấy",
            "không có căn cứ",
            "không đủ thông tin",
            "không có trong ngữ cảnh",
            "không được đề cập trong ngữ cảnh",
            "không được nêu cụ thể",
            "không nêu cụ thể",
            "không xác định",
            "no relevant information",
            "not enough information",
        )
        return any(pattern in text for pattern in patterns)

    def _handle_graph(self, query: str, history: str) -> tuple[str, list[str], str, str, str]:
        """Handle query via Graph RAG pipeline.

        Args:
            query: Resolved query.
            history: Formatted conversation history.

        Returns:
            Tuple of (answer, sources, context, actual_pipeline_used, kg_source).
        """
        try:
            result = self.graph_rag.answer(query, history=history)
            
            # Extract from dict
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            ctx = result.get("context", "")
            kg_source = result.get("kg_source", "")

            if (
                self._config.get("rag", {}).get("graph_fallback_to_vector", True)
                and self._is_uninformative_answer(answer)
            ):
                logger.info("Graph answer lacks evidence; falling back to VectorRAG answer")
                vec_answer, vec_sources, vec_ctx = self._handle_vector(query, history)
                return (
                    vec_answer,
                    vec_sources,
                    vec_ctx,
                    f"graph_traversal:{kg_source}->dense_retrieval" if kg_source else "dense_retrieval",
                    f"{kg_source}->vector" if kg_source else "vector",
                )
                
            actual_pipeline = f"graph_traversal:{kg_source}" if kg_source else "graph_traversal"
            return answer, sources, ctx, actual_pipeline, kg_source
        except Exception as exc:
            logger.error("GraphRAG failed, falling back to VectorRAG: {}", exc)
            ans, src, ctx = self._handle_vector(query, history)
            return ans, src, ctx, "dense_retrieval", ""

    def _handle_hybrid(
        self, query: str, history: str
    ) -> tuple[str, list[str], str, str]:
        """Handle hybrid_reasoning queries using both vector + graph context.

        This is the most powerful route — used for cross-document queries
        that require synthesizing information from multiple legal documents.
        Combines vector retrieval (for dense coverage) with graph traversal
        (for relational context) before generating the final answer in a single LLM call.

        Args:
            query: Resolved query.
            history: Formatted conversation history.

        Returns:
            Tuple of (answer, sources, context, kg_source). This route is strictly 'hybrid_reasoning'
            and does not fallback to other routes.
        """
        logger.info("Executing Hybrid RAG: merging vector and graph contexts")
        sources: list[str] = []
        kg_source = ""
        
        rag_cfg = self._config.get("rag", {})

        # 1. Get Vector Context
        vector_context = ""
        try:
            hybrid_vector_top_k = int(rag_cfg.get("hybrid_vector_top_k", 5))
            hybrid_vector_chunk_chars = int(rag_cfg.get("hybrid_vector_chunk_chars", 1000))
            candidate_k = int(rag_cfg.get("hybrid_vector_candidate_k", hybrid_vector_top_k * 3))
            vector_candidates = self.vector_rag.retriever.retrieve(query, top_k=candidate_k)
            vector_results = [
                result for result in vector_candidates
                if not self.vector_rag._is_low_value_chunk(result.chunk_text)
            ][:hybrid_vector_top_k]
            if not vector_results:
                vector_results = vector_candidates[:hybrid_vector_top_k]
            
            vector_parts = []
            for result in vector_results:
                metadata = getattr(result, "metadata", {})
                doc_id = (
                    metadata.get("doc_id")
                    or metadata.get("article_id")
                    or metadata.get("title")
                    or getattr(result, "doc_id", "")
                )
                lang = self._config.get("language", "vi")
                default_title = f"Document {doc_id}" if lang == "en" else f"Văn bản {doc_id}"
                title = metadata.get("title", default_title)
                
                chunk = " ".join(result.chunk_text.split())
                if len(chunk) > hybrid_vector_chunk_chars:
                    chunk = chunk[:hybrid_vector_chunk_chars] + "..."
                    
                vector_parts.append(f"[{doc_id}] {title}:\n{chunk}")
                if doc_id and doc_id not in sources:
                    sources.append(doc_id)
            vector_context = "\n\n".join(vector_parts)
        except Exception as exc:
            logger.warning("Hybrid: Vector retrieval failed: {}", exc)

        # 2. Get Graph Context
        graph_context = ""
        try:
            hybrid_top_k = int(rag_cfg.get("hybrid_graph_top_k", 3))
            hybrid_graph_context_chars = int(rag_cfg.get("hybrid_graph_context_chars", 2500))
            neo4j_client = getattr(self.graph_rag, "_neo4j_client", None)
            if neo4j_client and neo4j_client.verify_connection():
                graph_context = neo4j_client.get_multi_hop_context(query, top_k=hybrid_top_k)
                if graph_context and NO_GRAPH_CONTEXT not in graph_context:
                    kg_source = "neo4j"

            no_graph_msg = get_template(self._config.get("language", "vi"), "no_graph_found")
            if (not graph_context or no_graph_msg in graph_context
                    or NO_GRAPH_CONTEXT in graph_context):
                graph_context = self.graph_rag._sqlite_kg.multi_hop_context(query, top_k=hybrid_top_k)
                kg_source = "sqlite" if graph_context and no_graph_msg not in graph_context else "none"
            
            # Limit graph context length to avoid noise/token overflow
            if len(graph_context) > hybrid_graph_context_chars:
                graph_context = graph_context[:hybrid_graph_context_chars] + "..."
                
            nodes = self.graph_rag._sqlite_kg.search_nodes(query, top_k=hybrid_top_k)
            for n in nodes:
                nid = n.get("article_id")
                if nid and nid not in sources:
                    sources.append(nid)
        except Exception as exc:
            logger.warning("Hybrid: Graph traversal failed: {}", exc)

        vrag = self.vector_rag
        if not vector_context and not graph_context:
            return get_template(vrag.language, "no_context_found"), sources, "", kg_source

        # 3. Merge contexts
        merged_context = ""
        if vector_context:
            merged_context += f"{get_template(vrag.language, 'vector_header')}\n{vector_context}\n\n"
        
        # Check if graph_context contains localized "No information found"
        no_graph_msg = get_template(vrag.language, "no_graph_found")
        if graph_context and no_graph_msg not in graph_context:
            merged_context += f"{get_template(vrag.language, 'graph_header')}\n{graph_context}\n"

        # Select correct template and system prompt from vector_rag
        # Fixed: Using the dynamic prompt builders from VectorRAG instead of missing attributes
        system_prompt = vrag._build_system_prompt()
        history_str = history or ("None" if vrag.language == "en" else "Không có")
        prompt = vrag._build_prompt(merged_context.strip(), history_str, query)

        try:
            llm = vrag.llm
            answer = llm.generate(prompt, system_prompt=system_prompt)
            if hasattr(llm, "_strip_thinking"):
                answer = llm._strip_thinking(answer)
            
            # HotpotQA optimization: Normalize answer
            answer = answer.strip().replace(".", "").replace('"', "").replace("'", "")
            if (
                self._config.get("rag", {}).get("hybrid_fallback_to_vector", True)
                and self._is_uninformative_answer(answer)
                and vector_context
            ):
                logger.info("Hybrid answer lacks evidence; falling back to VectorRAG answer")
                vec_answer, vec_sources, vec_ctx = self._handle_vector(query, history)
                fallback_source = f"{kg_source}->vector" if kg_source else "vector"
                return vec_answer, vec_sources, vec_ctx, fallback_source
            return answer, sources, merged_context.strip(), kg_source

        except Exception as exc:
            logger.error("Hybrid generation failed: {}", exc)
            lang = self._config.get("language", "vi")
            err_msg = (f"Error synthesizing hybrid answer: {exc}" if lang == "en"
                       else f"Lỗi khi tổng hợp câu trả lời Hybrid: {exc}")
            return err_msg, sources, "", kg_source

    def _handle_clarify(self, query: str, history: str) -> str:
        """Handle ambiguous query by generating helpful clarification.

        Args:
            query: The ambiguous query.
            history: Conversation history.

        Returns:
            A helpful clarification with legal context.
        """
        report = self.ambiguity_detector.detect(query, history)
        
        lang = self._config.get("language", "vi")
        
        system_prompt = get_template(lang, "clarify_system")
        prompt = get_template(lang, "clarify_prompt", query=query, history=history)

        try:
            # Use OpenAI-compatible API for the clarification generation
            client = OpenAIClient(self._config.get("openai", self._config.get("ollama", {})))
            clarification = client.generate(prompt=prompt, system_prompt=system_prompt)
            if hasattr(client, "_strip_thinking"):
                clarification = client._strip_thinking(clarification)
            return clarification
        except Exception as exc:
            logger.error("Helpful clarification failed: {}", exc)
            if report.clarification_question:
                return report.clarification_question
            
            return (
                "Your question is unclear. Please provide more details for a precise answer."
                if lang == "en" else
                "Câu hỏi của bạn chưa đủ rõ ràng. "
                "Vui lòng cung cấp thêm chi tiết để tôi có thể trả lời chính xác hơn."
            )

    def _log_pipeline(
        self,
        query: str,
        resolved_query: str,
        router_output: Any,
        answer: str,
        latency_ms: float,
        session_id: str,
    ) -> None:
        """Log full pipeline execution to JSONL.

        Args:
            query: Original query.
            resolved_query: Query after coreference resolution.
            router_output: RouterOutput from two-stage router.
            answer: Generated answer.
            latency_ms: Total pipeline latency.
            session_id: Session identifier.
        """
        import datetime

        log_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "session_id": session_id,
            "query": query,
            "resolved_query": resolved_query if resolved_query != query else None,
            "query_features": {
                "entity_count": router_output.features.entity_count,
                "multi_hop_score": router_output.features.multi_hop_score,
                "ambiguity_score": router_output.features.ambiguity_score,
            },
            "stage1": {
                "route": router_output.stage1_route,
                "confidence": router_output.stage1_confidence,
            },
            "stage2_invoked": router_output.stage2_invoked,
            "stage2": {
                "route": router_output.route if router_output.stage2_invoked else None,
                "confidence": router_output.confidence if router_output.stage2_invoked else None,
                "reasoning": router_output.reasoning if router_output.stage2_invoked else None,
                "override": router_output.stage2_override,
                "override_reason": getattr(router_output, "stage2_override_reason", None),
                "complexity_level": getattr(router_output, "stage2_complexity_level", ""),
                "reasoning_steps": getattr(router_output, "stage2_reasoning_steps", []),
                "sub_questions": getattr(router_output, "stage2_sub_questions", []),
                "ambiguity_flags": getattr(router_output, "stage2_ambiguity_flags", {}),
                "clarify_question": getattr(router_output, "clarify_question", None),
                "parse_error": getattr(router_output, "stage2_parse_error", None),
            },
            "final_route": router_output.route,
            "pipeline_latency_ms": round(latency_ms, 1),
            "answer_length": len(answer),
            "is_ambiguous": router_output.is_ambiguous,
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except IOError as exc:
            logger.warning("Failed to write pipeline log: {}", exc)
