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

from llm.ollama_client import OllamaClient
from pipeline.conversation_manager import ConversationManager
from rag.graph_rag_adapter import GraphRAGAdapter
from rag.vector_rag import VectorRAG
from router.ambiguity_detector import AmbiguityDetector
from router.two_stage_router import TwoStageRouter


@dataclass
class PipelineResponse:
    """Complete response from the hybrid pipeline.

    Attributes:
        answer: Generated answer text.
        route_used: Which pipeline was used.
        confidence: Router confidence in the route.
        router_reasoning: Explanation of routing decision.
        stage2_invoked: Whether LLM verifier was used.
        sources: Document IDs referenced in the answer.
        latency_ms: Total pipeline latency.
        is_ambiguous: Whether the query was deemed ambiguous.
    """

    answer: str = ""
    route_used: str = ""
    confidence: float = 0.0
    router_reasoning: str = ""
    stage2_invoked: bool = False
    sources: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    is_ambiguous: bool = False


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

        if router_output.route == "vector":
            answer, sources = self._handle_vector(resolved_query, history_str)
        elif router_output.route == "graph":
            answer, sources = self._handle_graph(resolved_query, history_str)
        elif router_output.route == "clarify":
            answer = self._handle_clarify(resolved_query, history_str)
        else:
            logger.warning("Unknown route '{}', defaulting to vector", router_output.route)
            answer, sources = self._handle_vector(resolved_query, history_str)

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
            stage2_invoked=router_output.stage2_invoked,
            sources=sources,
            latency_ms=latency_ms,
            is_ambiguous=router_output.is_ambiguous,
        )

        logger.info(
            "Pipeline complete | route={} | latency={:.0f}ms | answer_len={}",
            response.route_used,
            response.latency_ms,
            len(response.answer),
        )

        return response

    def _handle_vector(self, query: str, history: str) -> tuple[str, list[str]]:
        """Handle query via Vector RAG pipeline.

        Args:
            query: Resolved query.
            history: Formatted conversation history.

        Returns:
            Tuple of (answer, sources).
        """
        try:
            result = self.vector_rag.answer(query, history=history)
            return result.answer, result.sources
        except Exception as exc:
            logger.error("VectorRAG failed: {}", exc)
            return f"Lỗi khi tra cứu: {exc}", []

    def _handle_graph(self, query: str, history: str) -> tuple[str, list[str]]:
        """Handle query via Graph RAG pipeline.

        Args:
            query: Resolved query.
            history: Formatted conversation history.

        Returns:
            Tuple of (answer, sources).
        """
        try:
            result = self.graph_rag.answer(query, history=history)
            return result.answer, result.sources
        except Exception as exc:
            logger.error("GraphRAG failed, falling back to VectorRAG: {}", exc)
            # Fallback to vector
            return self._handle_vector(query, history)

    def _handle_clarify(self, query: str, history: str) -> str:
        """Handle ambiguous query by generating helpful clarification.

        Args:
            query: The ambiguous query.
            history: Conversation history.

        Returns:
            A helpful clarification with legal context.
        """
        report = self.ambiguity_detector.detect(query, history)
        
        # Use LLM to generate a helpful clarification
        topics_str = ", ".join(report.detected_topics) if report.detected_topics else "pháp luật chung"
        
        prompt = f"""Bạn là một trợ lý luật sư ảo chuyên nghiệp và tận tâm.
Người dùng đã gửi một câu hỏi mơ hồ về chủ đề: {topics_str}.

Câu hỏi gốc: "{query}"
Lịch sử hội thoại: {history}

Yêu cầu:
1. Hãy tóm tắt ngắn gọn (1-2 câu) quy định chung của luật pháp Việt Nam về chủ đề người dùng đang quan tâm để cho họ thấy bạn vẫn có ích.
2. Sau đó, hãy giải thích tại sao câu hỏi hiện tại của họ chưa đủ rõ để bạn trả lời chính xác (ví dụ: thiếu chủ ngữ, thiếu văn bản luật cụ thể).
3. Đặt các câu hỏi gợi ý để họ cung cấp thêm thông tin cần thiết.

Lưu ý: Phản hồi phải chuyên nghiệp, lịch sự và BẮT BUỘC bằng tiếng Việt. Tuyệt đối không trả lời bằng tiếng Anh.
Giới hạn trong 150 chữ.

Câu trả lời của bạn:"""

        try:
            # Use Ollama for the clarification generation
            client = OllamaClient(self._config["ollama"])
            clarification = client.generate(prompt=prompt, system_prompt="Bạn là chuyên gia tư vấn pháp luật Việt Nam.")
            return clarification
        except Exception as exc:
            logger.error("Helpful clarification failed: {}", exc)
            if report.clarification_question:
                return report.clarification_question
            return (
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
                "route": router_output.route,
                "confidence": router_output.stage1_confidence,
            },
            "stage2_invoked": router_output.stage2_invoked,
            "stage2": {
                "route": router_output.route if router_output.stage2_invoked else None,
                "confidence": router_output.confidence if router_output.stage2_invoked else None,
                "reasoning": router_output.reasoning if router_output.stage2_invoked else None,
                "override": router_output.stage2_override,
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
