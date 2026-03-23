"""Two-stage router orchestrator — core novelty of the thesis.

Combines Stage 1 (XGBoost classifier) with Stage 2 (LLM Reasoning
Verifier) for adaptive routing decisions. When Stage 1 is confident,
skips Stage 2 for speed. When uncertain, invokes LLM for deeper analysis.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from loguru import logger

from router.ambiguity_detector import AmbiguityDetector
from router.features import FeatureExtractor, QueryFeatures
from router.llm_reasoning_verifier import LLMReasoningVerifier
from router.router_model import RouterModel, RouterOutput as Stage1Output


@dataclass
class RouterOutput:
    """Complete output from the two-stage routing process.

    Captures the full routing decision with provenance: which stage
    made the decision, whether Stage 2 overrode Stage 1, reasoning,
    latency breakdown, and the underlying features.

    Attributes:
        route: Final routing destination.
        confidence: Final confidence score.
        is_ambiguous: Whether the query was deemed ambiguous.
        reasoning: Explanation of the routing decision.
        stage1_confidence: Stage 1 model confidence.
        stage2_invoked: Whether Stage 2 was activated.
        stage2_override: Whether Stage 2 overrode Stage 1.
        latency_ms: Total router latency.
        features: Extracted query features.
    """

    route: Literal["vector", "graph", "clarify"] = "vector"
    confidence: float = 0.0
    is_ambiguous: bool = False
    reasoning: str = ""
    stage1_confidence: float = 0.0
    stage2_invoked: bool = False
    stage2_override: bool = False
    latency_ms: float = 0.0
    features: QueryFeatures = field(default_factory=QueryFeatures)


class TwoStageRouter:
    """Two-stage adaptive router: XGBoost + LLM Verifier.

    This is the core novelty of the thesis. Combines:
    - Stage 1: Fast XGBoost classifier for initial routing
    - Stage 2: LLM chain-of-thought verifier for uncertain cases

    When Stage 1 confidence >= threshold, Stage 2 is skipped.
    When confidence < threshold, Stage 2 provides deeper analysis
    and may override Stage 1's decision.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the two-stage router.

        Args:
            config: Full config dict. If None, loads from configs/config.yaml.
        """
        if config is None:
            config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

        self._config = config
        self.confidence_threshold: float = config["router"]["stage1"].get(
            "confidence_threshold", 0.85
        )
        self.stage2_enabled: bool = config["router"]["stage2"].get("enabled", True)

        # Initialize components
        self.feature_extractor = FeatureExtractor(config=config)
        self.ambiguity_detector = AmbiguityDetector(config.get("ambiguity"))
        self.router_model = RouterModel(config)
        self.llm_verifier = LLMReasoningVerifier(config) if self.stage2_enabled else None

        # Logging
        self.log_path = Path(config["logging"]["routing_log_path"])
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "TwoStageRouter initialized | threshold={} | stage2_enabled={}",
            self.confidence_threshold,
            self.stage2_enabled,
        )

    def route(
        self,
        query: str,
        history: str | None = None,
        session_id: str = "",
    ) -> RouterOutput:
        """Route a query through the two-stage process.

        Pipeline:
        1. Detect ambiguity
        2. Extract query features
        3. Stage 1: XGBoost classifier prediction
        4. If confidence >= threshold → use Stage 1 result
        5. If confidence < threshold → invoke Stage 2 LLM verifier
        6. Log the routing decision
        7. Return final RouterOutput

        Args:
            query: Vietnamese legal query.
            history: Optional conversation history.
            session_id: Session identifier for logging.

        Returns:
            RouterOutput with complete routing information.
        """
        start = time.perf_counter()

        # Step 1: Detect ambiguity
        ambiguity_report = self.ambiguity_detector.detect(query, history)

        # Step 2: Extract features (with ambiguity info)
        features = self.feature_extractor.extract(
            query=query,
            history=history,
            ambiguity_score=ambiguity_report.score,
            has_pronoun="pronoun" in ambiguity_report.ambiguity_types,
            missing_entity_type=ambiguity_report.missing_entity_type,
        )

        # Step 3: Stage 1 prediction
        stage1_output = self.router_model.predict(features)
        stage1_confidence = stage1_output.confidence
        stage1_route = stage1_output.route

        # If ambiguity is high, override to clarify
        if ambiguity_report.is_ambiguous and ambiguity_report.score >= 0.8:
            stage1_output.route = "clarify"
            stage1_output.confidence = ambiguity_report.score

        # Step 4/5: Decide whether to invoke Stage 2
        stage2_invoked = False
        stage2_override = False
        final_route = stage1_output.route
        final_confidence = stage1_output.confidence
        reasoning = f"Stage 1: route={stage1_route}, confidence={stage1_confidence:.3f}"

        # Specific logic for thesis: Stage 1 is often overconfident on 'clarify'.
        # We force Stage 2 if Stage 1 says 'clarify' but ambiguity score is low,
        # or if we want extra verification for any 'clarify' decision.
        is_uncertain = stage1_output.confidence < self.confidence_threshold
        if stage1_output.route == "clarify" and ambiguity_report.score < 0.6:
            is_uncertain = True
            logger.info("Stage 1 predicted 'clarify' but ambiguity score is low ({:.2f}). Forcing Stage 2.", ambiguity_report.score)

        if (
            self.stage2_enabled
            and self.llm_verifier is not None
            and is_uncertain
        ):
            # Stage 2: LLM Verification
            stage2_invoked = True
            logger.info(
                "Stage 1 confidence {:.3f} < threshold {:.3f}, invoking Stage 2",
                stage1_output.confidence,
                self.confidence_threshold,
            )

            stage2_output = self.llm_verifier.verify(query, history, stage1_output)

            if stage2_output.route != stage1_output.route:
                stage2_override = True
                reasoning = (
                    f"Stage 1: {stage1_route}({stage1_confidence:.3f}) → "
                    f"Stage 2 override: {stage2_output.route}({stage2_output.confidence:.3f})"
                )
            else:
                reasoning = (
                    f"Stage 1: {stage1_route}({stage1_confidence:.3f}) → "
                    f"Stage 2 confirmed: {stage2_output.route}({stage2_output.confidence:.3f})"
                )

            final_route = stage2_output.route
            final_confidence = stage2_output.confidence
        else:
            reasoning += " (high confidence, Stage 2 skipped)"

        latency_ms = (time.perf_counter() - start) * 1000

        output = RouterOutput(
            route=final_route,
            confidence=final_confidence,
            is_ambiguous=ambiguity_report.is_ambiguous,
            reasoning=reasoning,
            stage1_confidence=stage1_confidence,
            stage2_invoked=stage2_invoked,
            stage2_override=stage2_override,
            latency_ms=latency_ms,
            features=features,
        )

        # Step 6: Log routing decision
        self._log_routing(output, query, session_id)

        logger.info(
            "Routing complete | route={} | confidence={:.3f} | "
            "stage2={} | override={} | latency={:.0f}ms",
            output.route,
            output.confidence,
            stage2_invoked,
            stage2_override,
            latency_ms,
        )

        return output

    def _log_routing(
        self,
        output: RouterOutput,
        query: str,
        session_id: str,
    ) -> None:
        """Log routing decision to JSONL file for research analysis.

        Args:
            output: Final routing output.
            query: Original query.
            session_id: Session identifier.
        """
        import datetime

        log_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "session_id": session_id,
            "query": query,
            "query_features": {
                "entity_count": output.features.entity_count,
                "multi_hop_score": output.features.multi_hop_score,
                "ambiguity_score": output.features.ambiguity_score,
                "graph_keyword_count": output.features.graph_keyword_count,
                "legal_reference_count": output.features.legal_reference_count,
                "has_comparison": output.features.has_comparison,
                "has_pronoun": output.features.has_pronoun,
                "query_length": output.features.query_length,
            },
            "stage1": {
                "route": output.features.question_word or "n/a",
                "confidence": output.stage1_confidence,
            },
            "stage2_invoked": output.stage2_invoked,
            "stage2": {
                "route": output.route if output.stage2_invoked else None,
                "confidence": output.confidence if output.stage2_invoked else None,
                "reasoning": output.reasoning if output.stage2_invoked else None,
                "override": output.stage2_override,
            },
            "final_route": output.route,
            "is_ambiguous": output.is_ambiguous,
            "pipeline_latency_ms": round(output.latency_ms, 1),
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except IOError as exc:
            logger.warning("Failed to write routing log: {}", exc)
