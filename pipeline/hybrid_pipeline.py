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
from pipeline.legal_citation_prompts import (
    LEGAL_SYSTEM_PROMPT,
    CLARIFY_SYSTEM_PROMPT,
    build_user_prompt,
    build_clarify_prompt,
    wrap_evidence_string,
)
from pipeline.llm_retry_utils import RateLimiter, call_llm_with_backoff


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

    def __init__(self, config_path: str | Path | None = None, disable_stage2: bool = False) -> None:
        """Initialize hybrid pipeline from config file.

        Args:
            config_path: Path to config.yaml. Defaults to configs/config.yaml.
            disable_stage2: If True, completely disables Stage 2 regardless of config.
        """
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"

        config_path = Path(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)
            
        if disable_stage2:
            if "router" not in self._config:
                self._config["router"] = {}
            if "stage2" not in self._config["router"]:
                self._config["router"]["stage2"] = {}
            self._config["router"]["stage2"]["enabled"] = False

        # Initialize components
        logger.info("Initializing HybridPipeline...")

        self.router = TwoStageRouter(self._config)
        self.conversation_manager = ConversationManager(self._config.get("conversation"))
        self.ambiguity_detector = AmbiguityDetector(self._config.get("ambiguity"))

        # RAG pipelines (lazy init to save memory)
        self._vector_rag: VectorRAG | None = None
        self._graph_rag: GraphRAGAdapter | None = None

        # Shared rate limiter — one instance for the whole pipeline so all
        # LLM calls (vector, hybrid, clarify) are paced together.
        llm_cfg = self._config.get("openai", self._config.get("ollama", {}))
        self._llm_limiter = RateLimiter(
            min_interval_sec=float(llm_cfg.get("min_request_interval_sec", 1.5))
        )

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
        llm_cfg = instance._config.get("openai", instance._config.get("ollama", {}))
        instance._llm_limiter = RateLimiter(
            min_interval_sec=float(llm_cfg.get("min_request_interval_sec", 1.5))
        )
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
        session_id: str | None = None,
        username: str = "default_user",
        force_route: str | None = None,
        force_stage1_route: str | None = None,
        force_stage2: bool = False,
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
            force_route: Optional override for Oracle evaluation.

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
        if force_route:
            from router.two_stage_router import RouterOutput
            from router.two_stage_router import QueryFeatures
            router_output = RouterOutput(
                route=force_route,
                confidence=1.0,
                reasoning="Oracle forced route",
                stage1_route=force_route,
                stage2_invoked=False,
                stage2_override=False,
                is_ambiguous=False,
                features=QueryFeatures(feature_dict={})
            )
        else:
            router_output = self.router.route(
                query=resolved_query,
                history=history_str,
                session_id=session_id,
                force_stage1_route=force_stage1_route,
                force_stage2=force_stage2,
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
        elif router_output.route == "chitchat":
            answer = "Xin chào! Tôi là trợ lý AI am hiểu Pháp luật Việt Nam. Bạn có câu hỏi nào về quy định, thủ tục pháp lý cần tôi giải đáp không?"
            context = ""
            actual_pipeline = "chitchat"
        else:
            logger.warning("Unknown route '{}', defaulting to dense_retrieval", router_output.route)
            answer, sources, context = self._handle_vector(resolved_query, history_str)
            actual_pipeline = "dense_retrieval"

        # Emit canonical IDs (M3 fix)
        try:
            from evaluation.metrics.id_normalizer import normalize_legal_id
            canonical_sources = []
            for src in sources:
                norm_id = normalize_legal_id(src)
                canonical_sources.append(norm_id.key if norm_id.is_resolvable else "UNRESOLVABLE")
            sources = canonical_sources
        except ImportError:
            pass

        latency_ms = (time.perf_counter() - start) * 1000

        # Step 4: Update conversation
        self.conversation_manager.add_turn(
            session_id=session_id,
            query=query,
            response=answer,
            route=router_output.route,
            username=username,
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

    def query_stream(
        self,
        query: str,
        session_id: str = "default",
        verbose: bool = False,
        force_route: str | None = None,
        username: str = "default",
    ):
        """Streaming version of query. Yields text chunks, then yields a final JSON metadata object."""
        start = time.perf_counter()

        # Step 1: Resolve coreference
        history_turns = self.conversation_manager.get_history(session_id)
        resolved_query = self.conversation_manager.resolve_coreference(query, history_turns)
        history_str = self.conversation_manager.get_history_string(session_id)

        # Step 2: Route
        router_start = time.perf_counter()
        if force_route:
            from router.two_stage_router import RouterOutput, QueryFeatures
            router_output = RouterOutput(
                route=force_route, confidence=1.0, reasoning="Oracle forced route",
                stage1_route=force_route, stage2_invoked=False, stage2_override=False,
                is_ambiguous=False, features=QueryFeatures(feature_dict={})
            )
        else:
            router_output = self.router.route(query=resolved_query, history=history_str, session_id=session_id)
        router_latency_ms = (time.perf_counter() - router_start) * 1000

        # Step 3: Stream from appropriate pipeline
        sources = []
        actual_pipeline = router_output.route
        full_answer = ""

        if router_output.route == "clarify":
            report = self.ambiguity_detector.detect(resolved_query, history_str)
            base_question = report.clarification_question or "Câu hỏi của bạn chưa đủ rõ ràng. Bạn vui lòng làm rõ ý định hoặc cung cấp thêm thông tin."
            
            yield base_question + "\n\n"
            full_answer += base_question + "\n\n"
            
            prompt = f"""Người dùng đặt câu hỏi pháp luật nhưng bị mơ hồ.
Câu hỏi: "{resolved_query}"
Lý do mơ hồ: {router_output.reasoning}
Câu hỏi làm rõ: {base_question}

Nhiệm vụ: Tạo 2-4 lựa chọn để người dùng click. MỖI lựa chọn PHẢI là MỘT CÂU HỎI PHÁP LUẬT HOÀN CHỈNH, ĐỘC LẬP, có thể tra cứu và trả lời được NGAY — nêu rõ đối tượng / văn bản / nội dung cụ thể mà người dùng có khả năng đang muốn hỏi (kèm số hiệu văn bản hoặc thuật ngữ pháp lý nếu suy ra được từ ngữ cảnh). TUYỆT ĐỐI KHÔNG viết dạng "Tôi muốn hỏi về ...". Mỗi lựa chọn bắt đầu bằng "- [Option] ". Không giải thích thêm.
Ví dụ (câu hoàn chỉnh, tự đủ nghĩa):
- [Option] Mức phạt vi phạm nồng độ cồn với xe mô tô theo Nghị định 100/2019/NĐ-CP là bao nhiêu?
- [Option] Nghị định 100/2019/NĐ-CP hiện còn hiệu lực không và văn bản nào đã sửa đổi nó?"""
            
            try:
                for chunk in self.vector_rag.llm.generate_stream(prompt, system_prompt="Chỉ in ra các lựa chọn bắt đầu bằng '- [Option] '."):
                    full_answer += chunk
                    yield chunk
            except Exception as e:
                logger.error("Failed to stream options: {}", e)
        elif router_output.route == "dense_retrieval":
            # Query Rewriting Step: Map common terms to legal terms
            vrag = self.vector_rag
            rewrite_prompt = f"""Bạn là một hệ thống tiền xử lý truy vấn pháp luật. Nhiệm vụ của bạn là dịch TẤT CẢ các từ ngữ thông dụng sang thuật ngữ pháp luật CHÍNH XÁC trong văn bản luật.
BẮT BUỘC ÁP DỤNG CÁC QUY TẮC SAU:
- Các loại bằng lái/xe cụ thể (B1, B2, C, D, E, F...) -> BẮT BUỘC dịch thành "xe ô tô" hoặc "người điều khiển xe ô tô". KHÔNG giữ lại từ "B2" hay "bằng B2".
- "xe máy" -> "xe mô tô, xe gắn máy".
- "sổ đỏ" -> "Giấy chứng nhận quyền sử dụng đất".
- "vượt đèn đỏ" -> "không chấp hành hiệu lệnh của đèn tín hiệu giao thông".
- "tước bằng" -> "tước quyền sử dụng Giấy phép lái xe".

Chỉ trả về 1 câu hỏi đã được chuẩn hóa theo quy tắc trên, tuyệt đối không giải thích thêm.
Câu hỏi gốc: {resolved_query}"""
            try:
                rewritten_query = vrag.llm.generate(rewrite_prompt, system_prompt="Chỉ trả về 1 câu.")
                if hasattr(vrag.llm, "_strip_thinking"):
                    rewritten_query = vrag.llm._strip_thinking(rewritten_query)
                logger.info("Rewritten query for dense retrieval: {}", rewritten_query)
                search_query = rewritten_query
            except Exception as e:
                logger.warning("Query rewrite failed: {}", e)
                search_query = resolved_query

            top_k = int(self._config.get("rag", {}).get("vector_top_k", 8))
            results = vrag.retriever.retrieve(search_query, top_k=top_k)
            context_parts = []
            current_len = 0
            for r in results:
                doc_id = getattr(r, "doc_id", "") or (r.metadata.get("article_id") if hasattr(r, "metadata") else "")
                title = r.metadata.get("title", f"Văn bản {doc_id}") if hasattr(r, "metadata") else f"Văn bản {doc_id}"
                if doc_id and doc_id not in sources:
                    sources.append(doc_id)
                part = f"[{doc_id}] {title}:\n{r.chunk_text}"
                if current_len + len(part) > 12000:
                    break
                context_parts.append(part)
                current_len += len(part)
            ctx = "\n\n".join(context_parts)
            evidence_chunks = wrap_evidence_string(ctx, source="Vector")
            user_msg = build_user_prompt(search_query, evidence_chunks, [{"query": history_str}] if history_str and history_str not in ("None", "Không có", "") else None)
            
            buffer = ""
            fallback_triggered = False
            stream_started = False
            
            for chunk in vrag.llm.generate_stream(user_msg, system_prompt=LEGAL_SYSTEM_PROMPT):
                if not fallback_triggered and not stream_started:
                    buffer += chunk
                    # Check if buffer contains fallback triggers
                    lower_buf = buffer.lower()
                    if any(p in lower_buf for p in ["không tìm thấy", "không có", "không được đề cập", "không đủ thông tin", "không xác định"]):
                        fallback_triggered = True
                        break
                    # If buffer gets large enough and no trigger, flush buffer and start normal stream
                    if len(buffer) > 50:
                        stream_started = True
                        full_answer += buffer
                        yield buffer
                else:
                    full_answer += chunk
                    yield chunk

            if fallback_triggered:
                # Do GraphRAG fallback
                # We don't yield the fallback message to UI to keep it clean
                logger.info("Stream buffer detected uninformative answer. Falling back to GraphRAG/Hybrid.")                
                # We use Hybrid logic for fallback to get the best of both worlds
                grag = self.graph_rag
                graph_context = ""
                try:
                    graph_context = grag._sqlite_kg.multi_hop_context(resolved_query, top_k=3)
                    nodes = grag._sqlite_kg.search_nodes(resolved_query, top_k=3)
                    for n in nodes:
                        if n.get("article_id") and n.get("article_id") not in sources:
                            sources.append(n.get("article_id"))
                except Exception:
                    pass
                
                merged_context = ""
                if ctx:
                    merged_context += f"TÀI LIỆU VĂN BẢN:\n{ctx}\n\n"
                if graph_context:
                    merged_context += f"LIÊN KẾT ĐỒ THỊ:\n{graph_context}\n"
                    
                # Truncate merged context to avoid exceeding limits
                if len(merged_context) > 16000:
                    merged_context = merged_context[:16000] + "..."
                    
                evidence_chunks = wrap_evidence_string(merged_context, source="Tổng hợp")
                user_msg2 = build_user_prompt(resolved_query, evidence_chunks, [{"query": history_str}] if history_str and history_str not in ("None", "Không có", "") else None)
                
                try:
                    for chunk in vrag.llm.generate_stream(user_msg2, system_prompt=LEGAL_SYSTEM_PROMPT):
                        full_answer += chunk
                        yield chunk
                except Exception as e:
                    logger.error("Fallback generation failed: {}", e)
                    yield "\n*(Lỗi: Hệ thống quá tải, không thể hoàn thành câu trả lời)*"
                
                actual_pipeline = "dense_retrieval->hybrid_fallback"

        elif router_output.route == "hybrid_reasoning" or router_output.route == "graph_traversal":
            # For simplicity, fallback to _handle_hybrid logic for both in streaming
            vrag = self.vector_rag
            grag = self.graph_rag
            
            # Vector context
            v_results = vrag.retriever.retrieve(resolved_query, top_k=5)
            v_ctx_parts = []
            current_len = 0
            for r in v_results:
                doc_id = getattr(r, "doc_id", "") or (r.metadata.get("article_id") if hasattr(r, "metadata") else "")
                title = r.metadata.get("title", "") if hasattr(r, "metadata") else ""
                if doc_id and doc_id not in sources:
                    sources.append(doc_id)
                
                label = f"[{doc_id}] {title}" if title else f"[{doc_id}]"
                part = f"{label}:\n{r.chunk_text}"
                if current_len + len(part) > 10000:
                    break
                v_ctx_parts.append(part)
                current_len += len(part)
            vector_context = "\n".join(v_ctx_parts)
            
            # Graph context
            graph_context = ""
            try:
                graph_context = grag._sqlite_kg.multi_hop_context(resolved_query, top_k=3)
                nodes = grag._sqlite_kg.search_nodes(resolved_query, top_k=3)
                for n in nodes:
                    if n.get("article_id") and n.get("article_id") not in sources:
                        sources.append(n.get("article_id"))
            except Exception:
                pass
            
            merged_context = ""
            if vector_context:
                merged_context += f"TÀI LIỆU VĂN BẢN:\n{vector_context}\n\n"
            if graph_context:
                merged_context += f"LIÊN KẾT ĐỒ THỊ:\n{graph_context}\n"
                
            if len(merged_context) > 16000:
                merged_context = merged_context[:16000] + "..."
                
            evidence_chunks = wrap_evidence_string(merged_context, source="Tổng hợp")
            user_msg = build_user_prompt(resolved_query, evidence_chunks, [{"query": history_str}] if history_str and history_str not in ("None", "Không có", "") else None)
            
            buffer = ""
            fallback_triggered = False
            stream_started = False
            
            for chunk in vrag.llm.generate_stream(user_msg, system_prompt=LEGAL_SYSTEM_PROMPT):
                if not fallback_triggered and not stream_started:
                    buffer += chunk
                    lower_buf = buffer.lower()
                    if any(p in lower_buf for p in ["không tìm thấy", "không có", "không được đề cập"]):
                        fallback_triggered = True
                        break
                    if len(buffer) > 50:
                        stream_started = True
                        full_answer += buffer
                        yield buffer
                else:
                    full_answer += chunk
                    yield chunk
                    
            if fallback_triggered:
                logger.info("Hybrid/Graph stream buffer detected uninformative answer. Falling back to simple VectorRAG.")
                yield "\n\n*(Luồng kết hợp không tìm thấy. Đang chuyển về tra cứu cơ bản...)*\n\n"
                
                # Fetch more vector chunks for fallback
                v_results2 = vrag.retriever.retrieve(resolved_query, top_k=8)
                v_ctx_parts2 = []
                current_len = 0
                for r in v_results2:
                    doc_id = getattr(r, "doc_id", "") or (r.metadata.get("article_id") if hasattr(r, "metadata") else "")
                    title = r.metadata.get("title", "") if hasattr(r, "metadata") else ""
                    if doc_id and doc_id not in sources:
                        sources.append(doc_id)
                    
                    label = f"[{doc_id}] {title}" if title else f"[{doc_id}]"
                    part = f"{label}:\n{r.chunk_text}"
                    if current_len + len(part) > 12000:
                        break
                    v_ctx_parts2.append(part)
                    current_len += len(part)
                vector_context2 = "\n".join(v_ctx_parts2)
                
                evidence_chunks2 = wrap_evidence_string(vector_context2, source="Vector")
                user_msg2 = build_user_prompt(resolved_query, evidence_chunks2, [{"query": history_str}] if history_str and history_str not in ("None", "Không có", "") else None)
                
                try:
                    for chunk in vrag.llm.generate_stream(user_msg2, system_prompt=LEGAL_SYSTEM_PROMPT):
                        full_answer += chunk
                        yield chunk
                except Exception as e:
                    logger.error("Fallback generation failed: {}", e)
                    yield "\n*(Lỗi: Hệ thống quá tải, không thể hoàn thành câu trả lời)*"
                    
                actual_pipeline = f"{router_output.route}->vector_fallback"

        elif router_output.route == "chitchat":
            ans = "Xin chào! Tôi là trợ lý AI am hiểu Pháp luật Việt Nam. Bạn có câu hỏi nào về quy định, thủ tục pháp lý cần tôi giải đáp không?"
            full_answer = ans
            yield ans

        else:
            ans = "Định tuyến không xác định."
            full_answer = ans
            yield ans

        latency_ms = (time.perf_counter() - start) * 1000

        self.conversation_manager.add_turn(
            session_id=session_id, query=query, response=full_answer, route=router_output.route, username=username
        )

        metadata = {
            "__metadata__": True,
            "route_used": router_output.route,
            "confidence": router_output.confidence,
            "stage2_invoked": router_output.stage2_invoked,
            "sources": sources,
            "latency_ms": latency_ms,
            "router_latency_ms": router_latency_ms,
            "is_ambiguous": router_output.is_ambiguous,
            "resolved_query": resolved_query
        }
        yield "\n\n" + json.dumps(metadata, ensure_ascii=False)

    def _handle_vector(self, query: str, history: str) -> tuple[str, list[str], str]:
        """Handle query via standard Vector RAG pipeline.

        Args:
            query: Resolved query.
            history: Formatted conversation history.

        Returns:
            Tuple of (answer, sources, context).
        """
        try:
            # Query Rewriting Step: Map common terms to legal terms
            vrag = self.vector_rag
            rewrite_prompt = f"""Bạn là một hệ thống tiền xử lý truy vấn pháp luật. Nhiệm vụ của bạn là dịch TẤT CẢ các từ ngữ thông dụng sang thuật ngữ pháp luật CHÍNH XÁC trong văn bản luật.
BẮT BUỘC ÁP DỤNG CÁC QUY TẮC SAU:
- Các loại bằng lái/xe cụ thể (B1, B2, C, D, E, F...) -> BẮT BUỘC dịch thành "xe ô tô" hoặc "người điều khiển xe ô tô". KHÔNG giữ lại từ "B2" hay "bằng B2".
- "xe máy" -> "xe mô tô, xe gắn máy".
- "sổ đỏ" -> "Giấy chứng nhận quyền sử dụng đất".
- "vượt đèn đỏ" -> "không chấp hành hiệu lệnh của đèn tín hiệu giao thông".
- "tước bằng" -> "tước quyền sử dụng Giấy phép lái xe".

Chỉ trả về 1 câu hỏi đã được chuẩn hóa theo quy tắc trên, tuyệt đối không giải thích thêm.
Câu hỏi gốc: {query}"""
            try:
                rewritten_query = vrag.llm.generate(rewrite_prompt, system_prompt="Chỉ trả về 1 câu.")
                if hasattr(vrag.llm, "_strip_thinking"):
                    rewritten_query = vrag.llm._strip_thinking(rewritten_query)
                logger.info("Rewritten query for _handle_vector: {}", rewritten_query)
                search_query = rewritten_query
            except Exception as e:
                logger.warning("Query rewrite failed: {}", e)
                search_query = query

            result = self.vector_rag.answer(search_query, history=history)
            return result.answer, result.sources, getattr(result, "context", "")
        except Exception as exc:
            logger.error("VectorRAG failed: {}", exc)
            raise

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
                metadata = getattr(result, "metadata", {}) or {}
                raw_doc_id = (
                    metadata.get("canonical_id")
                    or metadata.get("doc_id")
                    or metadata.get("article_id")
                    or metadata.get("title")
                    or getattr(result, "doc_id", "")
                )
                
                import re
                if isinstance(raw_doc_id, str) and "::" in raw_doc_id and re.match(r"^[^:]+\::\d", raw_doc_id):
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
            
            graph_engine = getattr(self.graph_rag, "graph_engine", "neo4j")
            graphrag_client = getattr(self.graph_rag, "_graphrag_client", None)
            
            if graph_engine == "microsoft" and graphrag_client and graphrag_client.is_available():
                logger.info("Hybrid: Getting Graph context from Microsoft GraphRAG")
                ms_answer = graphrag_client.answer(query, history)
                graph_context = ms_answer
                kg_source = "microsoft_graphrag"
                # Limit graph context length to avoid noise/token overflow
                if len(graph_context) > hybrid_graph_context_chars:
                    graph_context = graph_context[:hybrid_graph_context_chars] + "..."
            else:
                neo4j_client = getattr(self.graph_rag, "_neo4j_client", None)
                neo4j_available = getattr(self.graph_rag, "_neo4j_available", False)
                llm = getattr(self.graph_rag, "llm", None)
                if neo4j_client and neo4j_available:
                    graph_context = neo4j_client.get_cypher_context(query, llm_client=llm, top_k=hybrid_top_k)
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

        # --- Citation prompt: wrap merged_context into structured chunks ---
        evidence_chunks = wrap_evidence_string(
            merged_context.strip(),
            source="Tổng hợp (Vector + Graph)",
        )
        history_list: list[dict] = []
        if history and history not in ("None", "Không có", ""):
            history_list = [{"query": history}]

        try:
            llm = vrag.llm
            _user_msg = build_user_prompt(query, evidence_chunks, history_list or None)
            hybrid_system_prompt = (
                vrag._build_system_prompt()
                if hasattr(vrag, "_build_system_prompt")
                else LEGAL_SYSTEM_PROMPT
            )
            answer = call_llm_with_backoff(
                fn=lambda: llm.generate(_user_msg, system_prompt=hybrid_system_prompt),
                limiter=self._llm_limiter,
                max_retries=8,
                base_delay=2.0,
            )
            if hasattr(llm, "_strip_thinking"):
                answer = llm._strip_thinking(answer)

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
            # IMPORTANT: do NOT write str(exc) as a prediction string.
            # Re-raise so the outer handler in query() can catch it and the
            # eval runner can record this as a skipped/failed sample.
            logger.error("Hybrid generation failed after retries: {}", exc)
            raise

    def _handle_clarify(self, query: str, history: str) -> str:
        """Handle ambiguous query by generating helpful clarification.

        Uses CLARIFY_SYSTEM_PROMPT from legal_citation_prompts to produce
        a single, focused clarification question.

        Args:
            query: The ambiguous query.
            history: Conversation history.

        Returns:
            A clarification question string.
        """
        report = self.ambiguity_detector.detect(query, history)

        lang = self._config.get("language", "vi")

        # Build history_list for build_clarify_prompt
        history_list: list[dict] | None = None
        if history and history not in ("None", "Không có", ""):
            history_list = [{"query": history}]

        clarify_prompt = build_clarify_prompt(query, history_list)

        try:
            client = OpenAIClient(self._config.get("openai", self._config.get("ollama", {})))
            clarification = client.generate(
                prompt=clarify_prompt,
                system_prompt=CLARIFY_SYSTEM_PROMPT,
            )
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
                "entity_count": getattr(router_output.features, "entity_count", 0),
                "multi_hop_score": getattr(router_output.features, "multi_hop_score", 0),
                "ambiguity_score": getattr(router_output.features, "ambiguity_score", 0),
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
