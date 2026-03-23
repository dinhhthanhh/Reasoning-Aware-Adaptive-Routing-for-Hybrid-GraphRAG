"""Stage 2 LLM Reasoning Verifier for the two-stage router.

Uses Ollama (llama3:8b) with structured prompts to verify and
potentially override Stage 1 routing decisions. Only invoked
when Stage 1 confidence is below the threshold.

This is a KEY NOVELTY component of the thesis.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from loguru import logger

from llm.ollama_client import OllamaClient
from router.router_model import RouterOutput


class LLMReasoningVerifier:
    """Stage 2 of the two-stage router: LLM-based verification.

    Uses chain-of-thought reasoning via a structured prompt to verify
    Stage 1's routing decision. Can override Stage 1 if the LLM
    determines a different route is more appropriate.

    Only activated when Stage 1 confidence < threshold to balance
    statistical speed (Stage 1) with reasoning depth (Stage 2).
    """

    ROUTING_PROMPT_TEMPLATE = """Bạn là một hệ thống phân tích câu hỏi pháp luật tiếng Việt.

## Câu hỏi hiện tại:
{query}

## Lịch sử hội thoại:
{history}

## Phân tích sơ bộ (Stage 1):
- Route đề xuất: {stage1_route}
- Confidence: {stage1_confidence:.2f}

## Nhiệm vụ của bạn:
Phân tích câu hỏi và quyết định pipeline phù hợp nhất.

### Định nghĩa:
- **vector**: Câu hỏi đơn giản, tra cứu thẳng, 1 văn bản
- **graph**: Câu hỏi phức tạp, cần so sánh nhiều điều luật, suy luận nhiều bước
- **clarify**: Câu hỏi nhập nhằng, thiếu thông tin, không thể trả lời chắc chắn

### Quy tắc:
1. Nếu câu hỏi nhắc đến đại từ không rõ → clarify
2. Nếu câu hỏi so sánh 2+ điều luật → graph
3. Nếu câu hỏi hỏi về mối quan hệ giữa các điều khoản → graph
4. Nếu câu hỏi tra cứu đơn giản 1 điều → vector

Trả lời ĐÚNG theo format JSON sau và không có gì khác:
{{
    "route": "vector" hoặc "graph" hoặc "clarify",
    "confidence": <số thực từ 0.0 đến 1.0>,
    "reasoning": "<giải thích ngắn gọn tại sao chọn route này>",
    "override_stage1": true hoặc false,
    "override_reason": "<nếu override, giải thích tại sao>"
}}"""

    VALID_ROUTES: set[str] = {"vector", "graph", "clarify"}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize LLM Reasoning Verifier.

        Args:
            config: Full config dict. If None, loads from configs/config.yaml.
        """
        if config is None:
            config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

        self.llm = OllamaClient(config["ollama"])
        self.max_reasoning_tokens: int = config["router"]["stage2"].get(
            "max_reasoning_tokens", 512
        )

        logger.info("LLMReasoningVerifier initialized")

    def verify(
        self,
        query: str,
        history: str | None,
        stage1_output: RouterOutput,
    ) -> RouterOutput:
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

        prompt = self.ROUTING_PROMPT_TEMPLATE.format(
            query=query,
            history=history or "Không có",
            stage1_route=stage1_output.route,
            stage1_confidence=stage1_output.confidence,
        )

        try:
            response = self.llm.generate_json(
                prompt=prompt,
                system_prompt="Bạn là hệ thống phân loại câu hỏi pháp luật. Trả lời bằng JSON.",
            )

            # Parse and validate response
            result = self._parse_response(response, stage1_output)

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
            # Fallback: return Stage 1 output unchanged
            return stage1_output

    def _parse_response(
        self,
        response: dict[str, Any],
        stage1_output: RouterOutput,
    ) -> RouterOutput:
        """Parse and validate the LLM JSON response.

        Args:
            response: Parsed JSON dict from the LLM.
            stage1_output: Stage 1 output for fallback values.

        Returns:
            Validated RouterOutput.
        """
        # Extract and validate route
        route = response.get("route", stage1_output.route)
        if route not in self.VALID_ROUTES:
            logger.warning("Invalid route '{}' from LLM, using Stage 1", route)
            route = stage1_output.route

        # Extract and clamp confidence
        confidence = response.get("confidence", stage1_output.confidence)
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = stage1_output.confidence

        # Extract reasoning
        reasoning = response.get("reasoning", "")
        override = response.get("override_stage1", False)
        override_reason = response.get("override_reason", "")

        # Build feature importances from stage 1 if available
        importances = stage1_output.feature_importances

        return RouterOutput(
            route=route,
            confidence=confidence,
            feature_importances=importances,
        )
