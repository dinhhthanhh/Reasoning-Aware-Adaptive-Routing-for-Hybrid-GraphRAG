"""Stage 2 LLM verifier for the two-stage router.

Uses an OpenAI-compatible client via :class:`llm.openai_client.OpenAIClient`
to verify and potentially refine Stage 1 routing decisions. The concrete
model is read from the ``openai`` config block; the current legal QA setup
uses ``Qwen/Qwen3-32B-AWQ``.

Stage 2 is invoked by the router verification policy, not only by low
Stage 1 confidence. The policy can consider confidence, ambiguity, and
reasoning signals. The prompt uses ``/no_think`` and asks for short
``reasoning_steps`` summaries in JSON, not long free-form chain-of-thought.

This is a key novelty component of the thesis.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from loguru import logger

from llm.openai_client import OpenAIClient
from router.router_model import RouterOutput


@dataclass
class Stage2RouterOutput:
    """Stage 2 output with reasoning metadata for logs and analysis."""

    route: Literal["dense_retrieval", "graph_traversal", "hybrid_reasoning", "clarify"] = "dense_retrieval"
    confidence: float = 0.0
    feature_importances: dict[str, float] | None = None
    override_stage1: bool = False
    override_reason: str | None = None
    complexity_level: str = "unknown"
    reasoning_steps: list[str] | None = None
    sub_questions: list[str] | None = None
    ambiguity_flags: dict[str, bool] | None = None
    clarify_question: str | None = None
    stage1_route: str = ""
    stage1_confidence: float = 0.0
    parse_error: str | None = None
    raw_route: str = ""
    guardrail_applied: bool = False
    guardrail_reason: str | None = None
    resolved_referent: str | None = None
    candidate_referents: list[dict[str, Any]] | None = None
    history_resolution_status: str | None = None
    clarification_reason_type: str | None = None
    suggested_resolved_query: str | None = None


class LLMReasoningVerifier:
    """Stage 2 of the two-stage router: LLM-based verification.

    Uses a structured step-by-step verification prompt through the
    OpenAI-compatible ``OpenAIClient``. The verifier returns JSON containing
    the final route, confidence, short reasoning summaries, ambiguity flags,
    and optional clarification question.

    Activation is controlled by ``TwoStageRouter``'s verification policy:
    low confidence, ambiguity risk, and explicit reasoning signals can all
    trigger Stage 2 depending on config. The prompt intentionally includes
    ``/no_think`` and does not request long chain-of-thought text.
    """

    SYSTEM_PROMPT_VI = """Bạn là một hệ thống phân tích câu hỏi pháp lý tiếng Việt.
Nhiệm vụ: xác minh và tinh chỉnh quyết định định tuyến (routing) từ Stage 1.

Bạn PHẢI trả về JSON hợp lệ theo schema sau, KHÔNG có thêm văn bản nào khác:
{
  "final_route": "<dense_retrieval|graph_traversal|hybrid_reasoning|clarify>",
  "confidence": <0.0-1.0>,
  "override_stage1": <true|false>,
  "override_reason": "<lý do nếu override, null nếu không>",
  "complexity_level": "<simple|moderate|complex|ambiguous>",
  "reasoning_steps": [
    "<bước phân tích ngắn 1>",
    "<bước phân tích ngắn 2>"
  ],
  "sub_questions": ["<câu hỏi con nếu multi-hop, [] nếu không>"],
  "ambiguity_flags": {
    "missing_entity": <true|false>,
    "pronoun_reference": <true|false>,
    "multi_interpretation": <true|false>,
    "incomplete_context": <true|false>
  },
  "clarify_question": "<câu hỏi làm rõ nếu final_route=clarify, null nếu không>",
  "resolved_referent": "<tham chiếu đã được giải quyết nếu có, null nếu không>",
  "candidate_referents": [{"text": "<ứng viên>", "type": "<loại>", "confidence": <0.0-1.0>}],
  "history_resolution_status": "<resolved|no_history|irrelevant_history|conflicting_history|not_needed>",
  "clarification_reason_type": "<missing_entity|multi_interpretation|incomplete_context|unresolved_history|conflicting_history|null>",
  "suggested_resolved_query": "<câu hỏi đã thay thế tham chiếu bằng thực thể cụ thể nếu có, null nếu không>"
}"""

    ROUTING_PROMPT_TEMPLATE_VI = """/no_think

Phân tích câu hỏi sau đây:

CÂU HỎI:
{query}

LỊCH SỬ HỘI THOẠI:
{history}

PHÂN GIẢI THAM CHIẾU TỪ LỊCH SỬ:
{history_resolution_block}

QUYẾT ĐỊNH STAGE 1:
- Route: {stage1_route}
- Confidence: {stage1_confidence:.3f}
- Top features: {feature_importances}

HƯỚNG DẪN PHÂN TÍCH:

Bạn là verifier tư vấn, không phải router toàn quyền. Hãy đề xuất route tốt nhất theo phân tích của bạn, nhưng chỉ override Stage 1 khi có bằng chứng rõ ràng rằng Stage 1 không phù hợp với loại suy luận của câu hỏi. Nếu Stage 1 đã hợp lý, hãy giữ nguyên route Stage 1. Không đổi route chỉ vì có một từ khóa pháp lý đơn lẻ.

Bước 1 - Kiểm tra tính đầy đủ của câu hỏi:
- Xác định câu hỏi có đủ "retrieval target" hay không: chủ thể/hành vi, đối tượng pháp lý, lĩnh vực, văn bản/điều khoản nếu người dùng đã nêu, hoặc mô tả đủ hẹp để truy xuất.
- Chỉ xem là ambiguous khi thiếu thông tin làm thay đổi đáng kể nguồn luật cần truy xuất hoặc cách áp dụng luật.
- Có đại từ hoặc tham chiếu chưa giải quyết như "anh ấy", "họ", "việc đó", "quy định này", "luật đó", "hai văn bản đã nêu" mà lịch sử hội thoại không giải quyết được không?
- Có thể hiểu theo nhiều cách pháp lý loại trừ nhau không, hay chỉ là câu hỏi rộng nhưng vẫn có một mục tiêu truy xuất rõ?
- Đây có phải câu hỏi nối tiếp như "vậy thì sao", "bước tiếp theo là gì", "có đúng không" nhưng không có ngữ cảnh không?
- KHÔNG chọn clarify chỉ vì câu hỏi dài, có điều kiện giả định, cần suy luận nhiều bước, hoặc có số hiệu/năm văn bản trông mới/lạ. Hãy coi tính tồn tại của văn bản là vấn đề truy xuất trong corpus, trừ khi chính câu hỏi thiếu nguồn cần truy xuất.
- Nếu câu hỏi đã nêu rõ số hiệu văn bản, điều khoản, chủ thể, hành vi hoặc bối cảnh áp dụng, không yêu cầu người dùng xác nhận lại chỉ vì câu hỏi phức tạp.
- Nếu history_resolution_status = resolved, hãy coi đại từ/chỉ định như "văn bản đó", "quy định này", "điều này" là đã được giải quyết. Không chọn clarify chỉ vì còn đại từ; hãy chọn dense/graph/hybrid theo loại truy xuất cần thiết.
- Nếu history_resolution_status = no_history, irrelevant_history, hoặc conflicting_history và câu hỏi có đại từ/chỉ định, ưu tiên clarify.
- Nếu có nhiều candidate_referents phù hợp, hỏi người dùng muốn nói đến ứng viên nào.
- Nếu câu hỏi rõ nhưng relation-heavy (bãi bỏ/sửa đổi/hiệu lực/thẩm quyền/căn cứ/áp dụng đồng thời), ưu tiên graph_traversal hoặc hybrid_reasoning thay vì dense_retrieval.
Nếu thiếu retrieval target nghiêm trọng hoặc có tham chiếu không thể giải quyết, chọn final_route = "clarify".

Bước 2 - Phân loại độ phức tạp:
- simple: hỏi 1 định nghĩa pháp lý chung, 1 sự kiện chung không gắn với số hiệu văn bản cụ thể (general factoid).
- moderate: hỏi quy trình, điều kiện, mức phạt, hoặc so sánh đơn giản trong cùng bối cảnh.
- complex: cần tổng hợp nhiều điều khoản, văn bản, hoặc quan hệ sửa đổi/bãi bỏ. Việc tra cứu nội dung của một văn bản/điều khoản cụ thể theo tên/số hiệu (doc-specific lookup) cũng được tính vào mức độ này vì đòi hỏi phải truy vấn theo metadata.
- ambiguous: thiếu thông tin nên không thể trả lời chính xác.

Bước 3 - Xác minh route theo nguyên tắc bảo toàn tín hiệu pháp lý:

- dense_retrieval: chọn khi câu hỏi hỏi một thông tin chung (general factoid) không chỉ định rõ số hiệu văn bản cụ thể. Không nâng lên graph_traversal nếu không cần nối nhiều điều khoản, nhiều chủ thể, hoặc nhiều văn bản. KHÔNG CHỌN dense_retrieval nếu câu hỏi có chứa số hiệu văn bản/điều khoản cụ thể.

- graph_traversal: chọn khi cần đi qua quan hệ giữa ít nhất hai node/chứng cứ trong cùng văn bản hoặc cùng lĩnh vực, chẳng hạn Điều A liên quan Điều B, cơ quan được phân cấp theo một điều và trách nhiệm theo điều khác, văn bản hiện tại bãi bỏ/sửa đổi/thay thế văn bản khác, hoặc câu hỏi yêu cầu cơ sở pháp lý dựa trên nhiều điều khoản. ĐẶC BIỆT: Nếu câu hỏi tra cứu nội dung/cơ quan ban hành của một văn bản hoặc điều khoản cụ thể dựa trên số hiệu (ví dụ: "Điều 1 Quyết định 25/2026/QĐ-UBND nói về gì?", "Ai ban hành VBHN-BXD 12?"), phải ưu tiên đề xuất graph_traversal với confidence cao. Không chọn dense_retrieval cho các trường hợp này.

- hybrid_reasoning: chọn khi câu hỏi cần đối chiếu hoặc tổng hợp từ nhiều văn bản, nhiều lĩnh vực, nhiều căn cứ pháp lý, hoặc tình huống giả định có ít nhất hai nguồn luật độc lập. Không hạ từ hybrid_reasoning xuống graph_traversal nếu câu hỏi có nhiều văn bản hoặc nhiều nguồn luật rõ ràng.

- clarify: chỉ chọn khi thật sự thiếu retrieval target. Không chọn clarify nếu câu hỏi dùng "Thông tư này", "Quyết định này", "Nghị định này", "văn bản này" nhưng trong câu hỏi, lịch sử hội thoại, hoặc route/context trước đó đã có số hiệu văn bản, điều khoản, lĩnh vực, chủ thể, hoặc tên thủ tục đủ để truy xuất. Trong trường hợp có anchor đủ hẹp nhưng vẫn cần suy luận, chọn graph_traversal hoặc hybrid_reasoning.

Các tín hiệu pháp lý như "thẩm quyền", "trách nhiệm", "chịu trách nhiệm", "hiệu lực", "bãi bỏ", "sửa đổi", "thay thế", "chuyển tiếp", "theo Điều", "theo Thông tư này", "theo Quyết định này" chỉ là soft evidence cho relational reasoning. Không tự động chọn graph_traversal chỉ vì xuất hiện một tín hiệu pháp lý đơn lẻ trong câu hỏi CHUNG (không có số hiệu văn bản). Tuy nhiên, nếu câu hỏi CÓ số hiệu văn bản hoặc điều khoản cụ thể (doc-specific), phải ưu tiên đề xuất graph_traversal với confidence cao ngay cả khi chỉ hỏi một thông tin đơn lẻ — vì graph đã index LegalDoc và LegalArticle theo doc_id, còn vector store không có metadata index theo số hiệu. Quyết định override cuối cùng vẫn do policy bảo toàn của router xử lý.

Ví dụ dense_retrieval (general factoid, không có số hiệu):
- "Hành lang an toàn đường bộ là gì?"
- "Quỹ Đổi mới công nghệ quốc gia là gì?"
- "Xe cứu thương cần trang thiết bị nào?"

Ví dụ graph_traversal (doc-specific, CÓ số hiệu hoặc điều khoản):
- "Điều 1 Quyết định 25/2026/QĐ-UBND nói về nội dung gì?" (dù chỉ hỏi 1 fact)
- "Ai là cơ quan ban hành VBHN-BXD 12?" (dù chỉ hỏi 1 fact)
- "Thông tư số 77/2017/TT-BTC còn hiệu lực không?" (doc-specific lookup)
- "Quyết định 846/QĐ-BNNMT bãi bỏ gì?" (doc-specific + legal effect)
- "Theo Điều 2, ai có trách nhiệm thi hành?" (cần nối chủ thể với điều khoản)

Quy tắc override:
- QUAN TRỌNG NHẤT: Nếu Stage 1 chọn dense_retrieval nhưng câu hỏi CÓ số hiệu văn bản hoặc điều khoản cụ thể (ví dụ: Thông tư, Quyết định, Nghị định + số hiệu, hoặc Điều + số), hãy đề xuất graph_traversal với confidence cao. Quyết định override cuối cùng sẽ do policy bảo toàn của router xử lý.
- Nếu Stage 1 đã chọn graph_traversal và câu hỏi có tín hiệu thẩm quyền/trách nhiệm/hiệu lực/bãi bỏ/sửa đổi/chuyển tiếp/căn cứ pháp lý, chỉ override sang dense_retrieval khi bạn chắc rằng đáp án nằm trực tiếp trong một điều duy nhất VÀ câu hỏi không chứa số hiệu văn bản cụ thể.
- Nếu Stage 1 đã chọn hybrid_reasoning và câu hỏi có từ hai văn bản/lĩnh vực/điều kiện pháp lý trở lên, không override sang clarify chỉ vì có giả định phức tạp.
- Nếu Stage 1 chọn dense_retrieval nhưng câu hỏi có ít nhất hai tín hiệu quan hệ pháp lý, hãy cân nhắc graph_traversal.
- Nếu Stage 1 là graph_traversal, không đổi sang dense_retrieval hoặc clarify trừ khi câu hỏi thật sự chỉ có một fact trực tiếp CHUNG (không gắn số hiệu) hoặc thiếu retrieval target nghiêm trọng. Nếu Stage 1 là hybrid_reasoning, không đổi sang graph_traversal hoặc clarify khi câu hỏi có nhiều văn bản, nhiều điều kiện, hoặc nhiều nguồn luật rõ ràng.

Negative examples - KHÔNG chọn dense_retrieval:
- Câu hỏi hỏi cơ sở pháp lý cho một quyền, trách nhiệm hoặc thẩm quyền dựa trên hai điều luật.
- Câu hỏi hỏi một chủ thể có trách nhiệm gì "theo Điều 2", "theo Thông tư này", hoặc "theo Quyết định này" khi cần nối chủ thể với thủ tục/văn bản/điều khoản.
- Câu hỏi hỏi văn bản nào bị bãi bỏ, sửa đổi, thay thế hoặc còn hiệu lực theo một văn bản khác.
- Câu hỏi hỏi quan hệ giữa thủ tục hành chính được phân cấp và cơ quan/cá nhân có thẩm quyền giải quyết.

Negative examples - KHÔNG chọn clarify:
- Câu hỏi có "Thông tư này" hoặc "Quyết định này" nhưng đã có số hiệu văn bản trong query/history hoặc retrieval context.
- Câu hỏi có tình huống giả định dài nhưng đã nêu rõ chủ thể, hành vi, lĩnh vực và văn bản liên quan.
- Câu hỏi cần đối chiếu nhiều điều kiện hoặc nhiều văn bản; đây là hybrid_reasoning, không phải ambiguity.

Positive examples - chọn clarify:
- "Quy định này áp dụng thế nào?" nhưng không có lịch sử, không có số hiệu văn bản, không có lĩnh vực.
- "Cơ quan nào có thẩm quyền?" nhưng không nêu thẩm quyền về việc gì, lĩnh vực nào, thủ tục nào.
- "Trường hợp đó xử lý ra sao?" nhưng không có lịch sử để xác định trường hợp đang nói tới.

YÊU CẦU:
- Trả về JSON ngay, không bọc markdown.
- reasoning_steps chỉ ghi tóm tắt ngắn gọn, không viết chain-of-thought dài.
- Nếu final_route = "clarify", phải có clarify_question cụ thể bằng tiếng Việt.
- Trả về JSON đúng schema, bao gồm history_resolution_status nếu đã được cung cấp."""

    SYSTEM_PROMPT_EN = """You are a legal question classification system.
Return valid JSON only, with no extra text."""

    ROUTING_PROMPT_TEMPLATE_EN = """/no_think

You are a legal question analysis system implementing the **Adaptive-RAG + SelfRAG** methodology.

## Question to Analyze:
{query}

## Conversation History (if any):
{history}

## Stage 1 Results (XGBoost):
- Suggested Route: **{stage1_route}**
- Confidence: **{stage1_confidence:.2f}**
- Top features: {feature_importances}

---

## Task: Perform Chain-of-Thought Analysis

Think step-by-step according to the Adaptive-RAG framework:

**Step 1 — Analyze Question Structure:**
How many sub-questions can this query be split into?
Are there ambiguous pronouns (he, she, it, that law, etc.)?
Does the query provide a sufficient retrieval target, such as the entity,
event, document, relation, or constrained context needed to search evidence?
Do not choose clarify only because the question is long, multi-hop,
cross-document, or uses a rare/recent-looking identifier.

**Step 2 — Analyze Legal Entities/Facts:**
How many specific legal documents or entities are mentioned?
Are there relationships between these entities (e.g., cross-references, comparisons)?

**Step 3 — Analyze Complexity (Adaptive-RAG taxonomy):**
- Level 1 (Simple): Requires looking up 1 document/fact → dense_retrieval
- Level 2 (Multi-hop): Requires comparing facts within the same domain/context → graph_traversal
- Level 3 (Complex): Requires synthesizing from MULTIPLE different sources → hybrid_reasoning

**Step 4 — Decision:**
Use `clarify` only when missing context would change the target evidence or
make several mutually exclusive interpretations plausible. If the target is
clear but reasoning is hard, choose `graph_traversal` or `hybrid_reasoning`.

Negative examples — not clarify:
- A question names two documents/entities and asks for synthesis or comparison → `hybrid_reasoning`.
- A question names an entity and asks for one fact, date, definition, or responsible body → `dense_retrieval`.
- A question needs relation following, amendment/reference tracking, or multi-hop evidence but has explicit targets → `graph_traversal`.

Positive examples — clarify:
- Unresolved references such as "that rule", "the previous document", or pronouns with no conversation history.
- A request for authority, penalty, legality, or next step without the relevant event, subject, domain, or source context.

Answer EXACTLY in the following JSON format (no text other than JSON):
{{
    "step1_sub_questions": <number of sub-questions>,
    "step1_has_ambiguous_pronoun": <true/false>,
    "step2_law_count": <number of entities mentioned>,
    "step2_has_cross_reference": <true/false>,
    "step3_complexity_level": <1, 2, or 3>,
    "step3_reasoning": "<short explanation for this level>",
    "route": "<dense_retrieval | graph_traversal | hybrid_reasoning | clarify>",
    "final_route": "<dense_retrieval | graph_traversal | hybrid_reasoning | clarify>",
    "confidence": <float 0.0-1.0>,
    "override_stage1": <true/false>,
    "override_reason": "<reason for override, or empty if none>",
    "complexity_level": "<simple|moderate|complex|ambiguous>",
    "reasoning_steps": ["<short summary step 1>", "<short summary step 2>"],
    "sub_questions": ["<sub-question if any>"],
    "ambiguity_flags": {{
        "missing_entity": <true/false>,
        "pronoun_reference": <true/false>,
        "multi_interpretation": <true/false>,
        "incomplete_context": <true/false>
    }},
    "clarify_question": "<clarifying question if route is clarify, null otherwise>"
}}"""

    ROUTING_PROMPT_TEMPLATE = ROUTING_PROMPT_TEMPLATE_VI # Default for backward compatibility

    VALID_ROUTES: set[str] = {"dense_retrieval", "graph_traversal", "hybrid_reasoning", "clarify"}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize LLM Reasoning Verifier.

        Args:
            config: Full config dict. If None, loads from configs/config.yaml.
        """
        if config is None:
            config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

        self.llm = OpenAIClient(config.get("openai", config.get("ollama", {})))
        self.max_reasoning_tokens: int = config["router"]["stage2"].get(
            "max_reasoning_tokens", 512
        )
        self.language: str = config.get("language", "vi")

        logger.info("LLMReasoningVerifier initialized | language={}", self.language)

    def verify(
        self,
        query: str,
        history: str | None,
        stage1_output: RouterOutput,
        history_resolution: Any | None = None,
    ) -> Stage2RouterOutput:
        """Verify Stage 1 routing decision using LLM reasoning.

        Sends a structured prompt to the LLM requesting chain-of-thought
        analysis. Parses the JSON response and determines whether to
        override Stage 1. Falls back to Stage 1 on any failure.

        Args:
            query: The Vietnamese legal query.
            history: Conversation history string, or None.
            stage1_output: The Stage 1 prediction to verify.

        Returns:
            RouterOutput with potentially overridden route and reasoning.
        """
        start = time.perf_counter()

        # Select template and system prompt based on language
        if self.language == "en":
            template = self.ROUTING_PROMPT_TEMPLATE_EN
            system_prompt = self.SYSTEM_PROMPT_EN
            history_text = history or "None"
        else:
            template = self.ROUTING_PROMPT_TEMPLATE_VI
            system_prompt = self.SYSTEM_PROMPT_VI
            history_text = history or "Không có"

        feature_importances = stage1_output.feature_importances or {}
        top_features = dict(
            sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)[:8]
        )

        prompt = template.format(
            query=query,
            history=history_text,
            history_resolution_block=self._format_history_resolution(history_resolution),
            stage1_route=stage1_output.route,
            stage1_confidence=stage1_output.confidence,
            feature_importances=top_features,
        )

        try:
            raw_response = self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
            )

            # Parse and validate response
            result = self._parse_raw_response(raw_response, stage1_output)

            latency = (time.perf_counter() - start) * 1000
            logger.info(
                "Stage 2 verification | route={} | confidence={:.3f} | "
                "override={} | latency={:.0f}ms",
                result.route,
                result.confidence,
                result.route != stage1_output.route,
                latency,
            )
            return result

        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000
            logger.warning(
                "Stage 2 LLM verification failed (falling back to Stage 1): {} | latency={:.0f}ms",
                exc,
                latency,
            )
            return self._fallback(stage1_output, parse_error=str(exc))

    def _parse_raw_response(
        self,
        raw_response: str,
        stage1_output: RouterOutput,
    ) -> Stage2RouterOutput:
        """Parse raw LLM text, including Qwen think blocks and code fences."""
        cleaned = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()

        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        try:
            response = json.loads(cleaned)
        except json.JSONDecodeError:
            brace_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not brace_match:
                return self._fallback(
                    stage1_output,
                    parse_error=f"no_json_found: {raw_response[:200]}",
                )
            try:
                response = json.loads(self._repair_json(brace_match.group(0)))
            except json.JSONDecodeError as exc:
                return self._fallback(
                    stage1_output,
                    parse_error=f"json_decode_error: {exc}",
                )

        if not isinstance(response, dict):
            return self._fallback(stage1_output, parse_error="json_root_not_object")

        return self._parse_response(response, stage1_output)

    def _parse_response(
        self,
        response: dict[str, Any],
        stage1_output: RouterOutput,
    ) -> Stage2RouterOutput:
        """Parse and validate the LLM JSON response.

        Args:
            response: Parsed JSON dict from the LLM.
            stage1_output: Stage 1 output for fallback values.

        Returns:
            Validated RouterOutput.
        """
        raw_route = str(response.get("final_route", response.get("route", stage1_output.route))).strip()
        route = raw_route
        invalid_route = route not in self.VALID_ROUTES
        if invalid_route:
            logger.warning("Invalid route '{}' from LLM, using Stage 1", raw_route)
            route = stage1_output.route

        # Extract and clamp confidence
        confidence = response.get("confidence", stage1_output.confidence)
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = stage1_output.confidence

        override = bool(response.get("override_stage1", route != stage1_output.route))
        override_reason = response.get("override_reason")
        if override_reason in ("", "null"):
            override_reason = None

        complexity_level = str(response.get("complexity_level", "unknown"))
        reasoning_steps = self._as_string_list(
            response.get("reasoning_steps", response.get("step3_reasoning", []))
        )
        sub_questions = self._as_string_list(response.get("sub_questions", []))
        ambiguity_flags = self._normalize_ambiguity_flags(response.get("ambiguity_flags", {}))
        clarify_question = response.get("clarify_question")
        if clarify_question in ("", "null"):
            clarify_question = None
        resolved_referent = response.get("resolved_referent")
        if resolved_referent in ("", "null"):
            resolved_referent = None
        candidate_referents = response.get("candidate_referents")
        if not isinstance(candidate_referents, list):
            candidate_referents = []
        history_resolution_status = response.get("history_resolution_status")
        if history_resolution_status in ("", "null"):
            history_resolution_status = None
        clarification_reason_type = response.get("clarification_reason_type")
        if clarification_reason_type in ("", "null"):
            clarification_reason_type = None
        suggested_resolved_query = response.get("suggested_resolved_query")
        if suggested_resolved_query in ("", "null"):
            suggested_resolved_query = None

        route, guardrail_applied, guardrail_reason = self._apply_safety_guardrails(
            route=route,
            confidence=confidence,
            ambiguity_flags=ambiguity_flags,
            clarify_question=clarify_question,
            complexity_level=complexity_level,
            sub_questions=sub_questions,
            stage1_output=stage1_output,
        )

        if guardrail_applied:
            override = False
            override_reason = f"guardrail_kept_stage1: {guardrail_reason}"
            confidence = stage1_output.confidence
        elif invalid_route:
            override = False
            override_reason = f"invalid_stage2_route: {raw_route}"
            confidence = stage1_output.confidence

        return Stage2RouterOutput(
            route=route,
            confidence=confidence,
            feature_importances=stage1_output.feature_importances or {},
            override_stage1=override,
            override_reason=override_reason,
            complexity_level=complexity_level,
            reasoning_steps=reasoning_steps,
            sub_questions=sub_questions,
            ambiguity_flags=ambiguity_flags,
            clarify_question=clarify_question,
            stage1_route=stage1_output.route,
            stage1_confidence=stage1_output.confidence,
            raw_route=raw_route,
            guardrail_applied=guardrail_applied,
            guardrail_reason=guardrail_reason,
            resolved_referent=str(resolved_referent) if resolved_referent else None,
            candidate_referents=[
                item for item in candidate_referents if isinstance(item, dict)
            ],
            history_resolution_status=str(history_resolution_status) if history_resolution_status else None,
            clarification_reason_type=str(clarification_reason_type) if clarification_reason_type else None,
            suggested_resolved_query=str(suggested_resolved_query) if suggested_resolved_query else None,
        )

    def _apply_safety_guardrails(
        self,
        route: str,
        confidence: float,
        ambiguity_flags: dict[str, bool],
        clarify_question: str | None,
        complexity_level: str,
        sub_questions: list[str],
        stage1_output: RouterOutput,
    ) -> tuple[str, bool, str | None]:
        """Apply conservative override policy to high-risk Stage 2 changes.

        This guardrail reduces route instability and over-correction by the
        LLM verifier. It does not replace Stage 1 or Stage 2; it only blocks
        risky overrides when the LLM evidence is not strong enough. The rules
        are intentionally small and explicit so they can be reported or
        removed in an ablation study.
        """
        has_ambiguity = any(bool(value) for value in ambiguity_flags.values())
        normalized_complexity = str(complexity_level or "").strip().lower()

        if route == "clarify" and not clarify_question:
            return stage1_output.route, True, "clarify_without_question"

        if stage1_output.route == "dense_retrieval" and route == "graph_traversal":
            # Allow upgrade when LLM finds multi-evidence OR doc-specific complexity.
            # "moderate" complexity includes doc-specific lookup which needs graph.
            has_upgrade_evidence = (
                len(sub_questions) >= 2
                or normalized_complexity in ("complex", "moderate")
            )
            allow_upgrade = confidence >= 0.80 and has_upgrade_evidence
            if not allow_upgrade:
                return (
                    stage1_output.route,
                    True,
                    "blocked_dense_to_graph_without_strong_multi_evidence",
                )

        if stage1_output.route == "graph_traversal" and route == "dense_retrieval":
            allow_downgrade = confidence >= 0.85 and not has_ambiguity
            if not allow_downgrade:
                return (
                    stage1_output.route,
                    True,
                    "blocked_graph_to_dense_without_high_confidence_and_clean_ambiguity",
                )

        if stage1_output.route == "hybrid_reasoning" and route == "graph_traversal":
            allow_downgrade = confidence >= 0.88 and not has_ambiguity
            if not allow_downgrade:
                return (
                    stage1_output.route,
                    True,
                    "blocked_hybrid_to_graph_without_high_confidence",
                )

        if stage1_output.route == "hybrid_reasoning" and route == "clarify":
            severe_ambiguity = bool(
                ambiguity_flags.get("missing_entity")
                or ambiguity_flags.get("pronoun_reference")
                or ambiguity_flags.get("multi_interpretation")
            )
            allow_clarify = confidence >= 0.90 and severe_ambiguity
            if not allow_clarify:
                return (
                    stage1_output.route,
                    True,
                    "blocked_hybrid_to_clarify_without_severe_ambiguity",
                )

        return route, False, None

    def _fallback(
        self,
        stage1_output: RouterOutput,
        parse_error: str | None = None,
    ) -> Stage2RouterOutput:
        """Return a non-crashing Stage 2 result that preserves failure context."""
        confidence = max(0.0, min(1.0, stage1_output.confidence * 0.9))
        return Stage2RouterOutput(
            route=stage1_output.route,
            confidence=confidence,
            feature_importances=stage1_output.feature_importances or {},
            override_stage1=False,
            override_reason=f"stage2_fallback: {parse_error}" if parse_error else None,
            complexity_level="unknown",
            reasoning_steps=[],
            sub_questions=[],
            ambiguity_flags={},
            clarify_question=None,
            stage1_route=stage1_output.route,
            stage1_confidence=stage1_output.confidence,
            parse_error=parse_error,
            raw_route=stage1_output.route,
        )

    @staticmethod
    def _as_string_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value else []
        if isinstance(value, list):
            return [str(item) for item in value if item is not None and str(item)]
        return [str(value)]

    @staticmethod
    def _normalize_ambiguity_flags(value: Any) -> dict[str, bool]:
        keys = ("missing_entity", "pronoun_reference", "multi_interpretation", "incomplete_context")
        if not isinstance(value, dict):
            return {key: False for key in keys}
        return {key: bool(value.get(key, False)) for key in keys}

    @staticmethod
    def _format_history_resolution(history_resolution: Any | None) -> str:
        if history_resolution is None:
            return "not_available"
        if hasattr(history_resolution, "to_dict"):
            payload = history_resolution.to_dict()
        elif isinstance(history_resolution, dict):
            payload = history_resolution
        else:
            payload = {"value": str(history_resolution)}
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @staticmethod
    def _repair_json(json_str: str) -> str:
        repaired = re.sub(r",\s*([}\]])", r"\1", json_str)
        repaired = re.sub(r"\bTrue\b", "true", repaired)
        repaired = re.sub(r"\bFalse\b", "false", repaired)
        repaired = re.sub(r"\bNone\b", "null", repaired)
        return repaired
