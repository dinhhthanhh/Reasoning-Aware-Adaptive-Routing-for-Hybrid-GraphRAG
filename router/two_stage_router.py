"""Two-stage router orchestrator — core novelty of the thesis.

Combines Stage 1 (XGBoost classifier) with Stage 2 (LLM Reasoning
Verifier) for adaptive routing decisions. Stage 2 is invoked only when
the configurable verification policy sees uncertainty or ambiguity risk.
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
        stage1_route: Stage 1 model route before any post-processing.
        stage1_confidence: Stage 1 model confidence.
        stage2_invoked: Whether Stage 2 was activated.
        stage2_override: Whether Stage 2 overrode Stage 1.
        latency_ms: Total router latency.
        features: Extracted query features.
    """

    route: Literal["dense_retrieval", "graph_traversal", "hybrid_reasoning", "clarify"] = "dense_retrieval"
    confidence: float = 0.0
    is_ambiguous: bool = False
    reasoning: str = ""
    stage1_route: str = ""
    stage1_confidence: float = 0.0
    base_route: str = ""
    base_confidence: float = 0.0
    ambiguity_candidate_route: str | None = None
    ambiguity_override_allowed: bool = False
    ambiguity_override_reason: str | None = None
    stage2_invoked: bool = False
    stage2_override: bool = False
    stage2_complexity_level: str = ""
    stage2_reasoning_steps: list[str] = field(default_factory=list)
    stage2_sub_questions: list[str] = field(default_factory=list)
    stage2_ambiguity_flags: dict[str, bool] = field(default_factory=dict)
    stage2_override_reason: str | None = None
    stage2_raw_route: str = ""
    stage2_guardrail_applied: bool = False
    stage2_guardrail_reason: str | None = None
    stage2_override_allowed: bool = False
    stage2_override_policy_reason: str | None = None
    stage2_trigger_reasons: list[str] = field(default_factory=list)
    clarify_question: str | None = None
    stage2_parse_error: str | None = None
    latency_ms: float = 0.0
    features: QueryFeatures = field(default_factory=QueryFeatures)


class TwoStageRouter:
    """Two-stage adaptive router: XGBoost + LLM Verifier.

    This is the core novelty of the thesis. Combines:
    - Stage 1: Fast XGBoost classifier for initial routing
    - Stage 2: LLM chain-of-thought verifier for uncertain cases

    High-confidence dense retrieval with low ambiguity skips Stage 2.
    Uncertain, ambiguity-sensitive, or suspicious clarify decisions can
    invoke Stage 2 for deeper analysis and possible override.
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

        # Configurable verification policy. These thresholds keep the
        # Stage 2 trigger research-friendly: ambiguity and reasoning signals
        # are measured separately, and reasoning does not force clarification.
        override_cfg = config.get("router", {}).get("override_rules", {})
        self.entity_count_threshold: int = override_cfg.get("entity_count_threshold", 1)
        self.multi_hop_score_threshold: float = override_cfg.get("multi_hop_score_threshold", 0.3)
        self.ambiguity_clarify_threshold: float = override_cfg.get("ambiguity_clarify_threshold", 0.8)
        self.ambiguity_force_stage2_threshold: float = override_cfg.get("ambiguity_force_stage2_threshold", 0.6)
        self.high_confidence_dense_skip_threshold: float = override_cfg.get(
            "high_confidence_dense_skip_threshold", self.confidence_threshold
        )
        self.dense_skip_max_ambiguity: float = override_cfg.get("dense_skip_max_ambiguity", 0.4)
        self.reasoning_force_stage2_enabled: bool = override_cfg.get(
            "reasoning_force_stage2_enabled", False
        )
        self.reasoning_force_stage2_threshold: float = override_cfg.get(
            "reasoning_force_stage2_threshold", 0.6
        )
        self.reasoning_force_confidence_ceiling: float = override_cfg.get(
            "reasoning_force_confidence_ceiling", 0.7
        )

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
        4. Apply ambiguity and high-confidence dense fast-path policy
        5. Invoke Stage 2 only when the verification policy requires it
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

        # Stage 1 is the base decision. Ambiguity and Stage 2 can only
        # change it through explicit conservative override policies.
        base_route = stage1_route
        base_confidence = stage1_confidence
        final_route = base_route
        final_confidence = base_confidence

        ambiguity_candidate_route: str | None = None
        ambiguity_candidate_confidence = 0.0
        ambiguity_override_allowed = False
        ambiguity_override_reason: str | None = None
        if ambiguity_report.is_ambiguous and ambiguity_report.score >= self.ambiguity_clarify_threshold:
            ambiguity_candidate_route = "clarify"
            ambiguity_candidate_confidence = ambiguity_report.score
            ambiguity_override_allowed, ambiguity_override_reason = self._allow_ambiguity_override(
                stage1_route=stage1_route,
                stage1_confidence=stage1_confidence,
                ambiguity_score=ambiguity_report.score,
                features=features,
            )
            if ambiguity_override_allowed:
                final_route = ambiguity_candidate_route
                final_confidence = ambiguity_candidate_confidence

        # Step 4/5: Decide whether to invoke Stage 2
        stage2_invoked = False
        stage2_override = False
        reasoning = f"Stage 1: route={stage1_route}, confidence={stage1_confidence:.3f}"
        stage2_complexity_level = ""
        stage2_reasoning_steps: list[str] = []
        stage2_sub_questions: list[str] = []
        stage2_ambiguity_flags: dict[str, bool] = {}
        stage2_override_reason: str | None = None
        stage2_raw_route = ""
        stage2_guardrail_applied = False
        stage2_guardrail_reason: str | None = None
        stage2_override_allowed = False
        stage2_override_policy_reason: str | None = None
        clarify_question: str | None = None
        stage2_parse_error: str | None = None

        is_uncertain, stage2_reasons = self._should_invoke_stage2(
            final_route=final_route,
            final_confidence=final_confidence,
            ambiguity_score=ambiguity_report.score,
            ambiguity_is_detected=ambiguity_report.is_ambiguous,
            features=features,
        )

        if (
            self.stage2_enabled
            and self.llm_verifier is not None
            and is_uncertain
        ):
            # Stage 2: LLM Verification
            stage2_invoked = True
            logger.info(
                "Invoking Stage 2 verifier | stage1_confidence={:.3f} | reasons={}",
                stage1_output.confidence,
                "; ".join(stage2_reasons) or "unspecified",
            )

            stage2_output = self.llm_verifier.verify(query, history, stage1_output)
            stage2_complexity_level = getattr(stage2_output, "complexity_level", "")
            stage2_reasoning_steps = getattr(stage2_output, "reasoning_steps", None) or []
            stage2_sub_questions = getattr(stage2_output, "sub_questions", None) or []
            stage2_ambiguity_flags = getattr(stage2_output, "ambiguity_flags", None) or {}
            stage2_override_reason = getattr(stage2_output, "override_reason", None)
            stage2_raw_route = getattr(stage2_output, "raw_route", "") or ""
            stage2_guardrail_applied = bool(getattr(stage2_output, "guardrail_applied", False))
            stage2_guardrail_reason = getattr(stage2_output, "guardrail_reason", None)
            clarify_question = getattr(stage2_output, "clarify_question", None)
            stage2_parse_error = getattr(stage2_output, "parse_error", None)

            if stage2_parse_error:
                if not ambiguity_override_allowed:
                    final_route = stage1_route
                    final_confidence = stage1_confidence
                reasoning = (
                    f"Stage 1: {stage1_route}({stage1_confidence:.3f}) → "
                    f"Stage 2 parse fallback kept {final_route}({final_confidence:.3f})"
                )
            elif stage2_output.route != final_route:
                stage2_override_allowed, stage2_override_policy_reason = self._allow_stage2_override(
                    stage1_route=stage1_route,
                    stage1_confidence=stage1_confidence,
                    stage2_route=stage2_output.route,
                    stage2_confidence=stage2_output.confidence,
                    ambiguity_score=ambiguity_report.score,
                    features=features,
                    stage2_ambiguity_flags=stage2_ambiguity_flags,
                    stage2_sub_questions=stage2_sub_questions,
                )
                if stage2_override_allowed:
                    stage2_override = True
                    reasoning = (
                        f"Stage 1: {stage1_route}({stage1_confidence:.3f}) → "
                        f"Stage 2 rescue override: {stage2_output.route}({stage2_output.confidence:.3f})"
                        f" | policy={stage2_override_policy_reason}"
                    )
                    final_route = stage2_output.route
                    final_confidence = stage2_output.confidence
                else:
                    stage2_override = False
                    if not ambiguity_override_allowed:
                        final_route = stage1_route
                        final_confidence = stage1_confidence
                    reasoning = (
                        f"Stage 1 kept: {final_route}({final_confidence:.3f}); "
                        f"Stage 2 suggested {stage2_output.route}({stage2_output.confidence:.3f}) "
                        f"but override blocked | policy={stage2_override_policy_reason}"
                    )
            else:
                stage2_override = False
                stage2_override_allowed = False
                stage2_override_policy_reason = "stage2_confirmed_stage1"
                reasoning = (
                    f"Stage 1: {final_route}({final_confidence:.3f}) → "
                    f"Stage 2 confirmed: {stage2_output.route}({stage2_output.confidence:.3f})"
                )
            if stage2_complexity_level:
                reasoning += f" | complexity={stage2_complexity_level}"
            if stage2_override_reason:
                reasoning += f" | reason={stage2_override_reason}"
        else:
            reasoning += " (Stage 2 skipped by policy)"

        latency_ms = (time.perf_counter() - start) * 1000

        output = RouterOutput(
            route=final_route,
            confidence=final_confidence,
            is_ambiguous=ambiguity_report.is_ambiguous,
            reasoning=reasoning,
            stage1_route=stage1_route,
            stage1_confidence=stage1_confidence,
            base_route=base_route,
            base_confidence=base_confidence,
            ambiguity_candidate_route=ambiguity_candidate_route,
            ambiguity_override_allowed=ambiguity_override_allowed,
            ambiguity_override_reason=ambiguity_override_reason,
            stage2_invoked=stage2_invoked,
            stage2_override=stage2_override,
            stage2_complexity_level=stage2_complexity_level,
            stage2_reasoning_steps=stage2_reasoning_steps,
            stage2_sub_questions=stage2_sub_questions,
            stage2_ambiguity_flags=stage2_ambiguity_flags,
            stage2_override_reason=stage2_override_reason,
            stage2_raw_route=stage2_raw_route,
            stage2_guardrail_applied=stage2_guardrail_applied,
            stage2_guardrail_reason=stage2_guardrail_reason,
            stage2_override_allowed=stage2_override_allowed,
            stage2_override_policy_reason=stage2_override_policy_reason,
            stage2_trigger_reasons=stage2_reasons,
            clarify_question=clarify_question,
            stage2_parse_error=stage2_parse_error,
            latency_ms=latency_ms,
            features=features,
        )

        # Step 6: Log routing decision
        self._log_routing(output, query, session_id)

        logger.info(
            "Routing complete | route={} | confidence={:.3f} | "
            "stage2={} | stage2_override={} | latency={:.0f}ms",
            output.route,
            output.confidence,
            stage2_invoked,
            stage2_override,
            latency_ms,
        )

        return output

    def _allow_ambiguity_override(
        self,
        stage1_route: str,
        stage1_confidence: float,
        ambiguity_score: float,
        features: QueryFeatures,
    ) -> tuple[bool, str]:
        """Decide whether rule-based ambiguity can override Stage 1 to clarify.

        Ambiguity override is intentionally conservative. It should not turn
        answerable graph/hybrid legal questions into clarification requests
        merely because they contain demonstratives such as "Thông tư này" or
        complex conditions. Stage 1 remains the base route unless ambiguity
        clearly points to an unresolved retrieval target.
        """
        if ambiguity_score < self.ambiguity_clarify_threshold:
            return False, "ambiguity_below_clarify_threshold"

        has_pronoun = bool(getattr(features, "has_pronoun", False))
        history_resolves = bool(getattr(features, "history_resolves_ambiguity", False))

        if stage1_route in {"graph_traversal", "hybrid_reasoning"}:
            if ambiguity_score >= 0.95 and has_pronoun and not history_resolves:
                return True, "extreme_unresolved_pronoun_on_reasoning_route"
            return False, "blocked_auto_clarify_for_reasoning_route"

        if stage1_route == "dense_retrieval":
            allow = (
                ambiguity_score >= 0.95
                and stage1_confidence <= 0.65
                and has_pronoun
                and not history_resolves
            )
            if allow:
                return True, "high_unresolved_pronoun_on_dense_route"
            return False, "blocked_auto_clarify_without_unresolved_pronoun"

        return False, "no_ambiguity_override_rule_matched"

    def _allow_stage2_override(
        self,
        stage1_route: str,
        stage1_confidence: float,
        stage2_route: str,
        stage2_confidence: float,
        ambiguity_score: float,
        features: QueryFeatures,
        stage2_ambiguity_flags: dict[str, bool],
        stage2_sub_questions: list[str],
    ) -> tuple[bool, str]:
        """Rescue-only override policy for Stage 2 suggestions.

        Stage 2 is treated as an advisory verifier. Overrides are denied by
        default and allowed only for high-confidence rescue scenarios where
        Stage 1 is likely underestimating graph or hybrid reasoning demand.
        """
        severe_ambiguity = bool(
            stage2_ambiguity_flags.get("missing_entity")
            or stage2_ambiguity_flags.get("pronoun_reference")
            or stage2_ambiguity_flags.get("multi_interpretation")
        )
        strong_graph_signal = (
            features.authority_chain_count >= 1
            or features.legal_effect_count >= 1
            or features.procedural_count >= 1
            or features.multi_entity_relation_count >= 1
            or features.graph_keyword_count >= 2
            or len(stage2_sub_questions) >= 2
        )

        if (
            stage1_route == "dense_retrieval"
            and stage2_route == "graph_traversal"
            and stage1_confidence <= 0.65
            and stage2_confidence >= 0.85
            and strong_graph_signal
        ):
            return True, "rescue_dense_to_graph_strong_reasoning_signal"

        if (
            stage1_route == "graph_traversal"
            and stage2_route == "hybrid_reasoning"
            and stage2_confidence >= 0.88
            and (
                features.cross_doc_signals
                or features.legal_reference_count >= 2
                or features.complexity_level >= 3
            )
        ):
            return True, "rescue_graph_to_hybrid_cross_document_signal"

        if (
            stage1_route == "dense_retrieval"
            and stage2_route == "hybrid_reasoning"
            and stage1_confidence <= 0.65
            and stage2_confidence >= 0.88
            and (
                features.cross_doc_signals
                or features.legal_reference_count >= 2
                or features.complexity_level >= 3
                or features.sub_question_count >= 2
            )
        ):
            return True, "rescue_dense_to_hybrid_strong_structural_signal"

        if (
            stage2_route == "clarify"
            and stage2_confidence >= 0.92
            and ambiguity_score >= 0.75
            and severe_ambiguity
        ):
            return True, "rescue_to_clarify_severe_ambiguity"

        # Rescue rule: dense → graph for doc-specific lookup.
        # When query has explicit legal references or law specificity,
        # Stage 2 should be allowed to upgrade even at moderate Stage 1
        # confidence (the existing rule requires stage1_confidence <= 0.65
        # which is too strict for overconfident Stage 1 dense predictions).
        doc_specific_signal = (
            features.legal_reference_count >= 1
            or features.law_specificity >= 1
        )
        if (
            stage1_route == "dense_retrieval"
            and stage2_route == "graph_traversal"
            and stage2_confidence >= 0.82
            and doc_specific_signal
        ):
            return True, "rescue_dense_to_graph_doc_specific_lookup"

        return False, "blocked_by_rescue_only_policy"

    def _should_invoke_stage2(
        self,
        final_route: str,
        final_confidence: float,
        ambiguity_score: float,
        ambiguity_is_detected: bool,
        features: QueryFeatures,
    ) -> tuple[bool, list[str]]:
        """Decide whether the expensive verifier is worth invoking.

        The policy separates three signals:
        - statistical uncertainty from Stage 1 confidence;
        - ambiguity risk from the ambiguity detector;
        - reasoning complexity from multi-hop/cross-document features.

        Reasoning signals are advisory by default. They can escalate retrieval
        strategy, but they should not by themselves turn an answerable question
        into a clarification request.
        """
        reasons: list[str] = []

        # Dense fast-path: skip Stage 2 only when there is NO legal signal.
        # Problem B fix: queries with doc-specific or relational legal features
        # must not be blindly trusted as dense even at high confidence.
        strong_legal_signal = (
            features.legal_reference_count >= 1
            or features.graph_keyword_count >= 1
            or features.law_specificity >= 1
            or features.authority_chain_count >= 1
            or features.legal_effect_count >= 1
        )
        dense_fast_path_candidate = (
            final_route == "dense_retrieval"
            and final_confidence >= self.high_confidence_dense_skip_threshold
            and ambiguity_score <= self.dense_skip_max_ambiguity
        )
        if dense_fast_path_candidate:
            if not strong_legal_signal:
                reasons.append(
                    "skip high-confidence dense route with low ambiguity and no legal signal "
                    f"(confidence={final_confidence:.3f}, ambiguity={ambiguity_score:.2f})"
                )
                return False, reasons
            else:
                reasons.append("dense route has legal signal; Stage 2 remains eligible")
        dense_fast_path = dense_fast_path_candidate and not strong_legal_signal

        should_invoke = False
        if final_confidence < self.confidence_threshold:
            should_invoke = True
            reasons.append(
                f"confidence {final_confidence:.3f} < threshold {self.confidence_threshold:.3f}"
            )

        if ambiguity_is_detected and ambiguity_score >= self.ambiguity_force_stage2_threshold:
            should_invoke = True
            reasons.append(
                f"ambiguity {ambiguity_score:.2f} >= {self.ambiguity_force_stage2_threshold:.2f}"
            )
            logger.info(
                "Ambiguity score {:.2f} >= {:.2f}. Forcing Stage 2 verification.",
                ambiguity_score,
                self.ambiguity_force_stage2_threshold,
            )

        if (
            self.reasoning_force_stage2_enabled
            and final_route == "dense_retrieval"
            and strong_legal_signal
            and final_confidence <= self.reasoning_force_confidence_ceiling
        ):
            should_invoke = True
            reasons.append(
                "dense route has legal signal inside confidence ceiling "
                f"(confidence={final_confidence:.3f}, ceiling={self.reasoning_force_confidence_ceiling:.3f})"
            )
            logger.info(
                "Dense route has legal signal inside confidence ceiling "
                "(confidence={:.3f}, legal_refs={}, graph_kw={}, law_specificity={}, "
                "authority={}, legal_effect={}).",
                final_confidence,
                features.legal_reference_count,
                features.graph_keyword_count,
                features.law_specificity,
                features.authority_chain_count,
                features.legal_effect_count,
            )

        reasoning_signal = self._has_reasoning_signal(features)
        if (
            self.reasoning_force_stage2_enabled
            and reasoning_signal
            and final_confidence <= self.reasoning_force_confidence_ceiling
            and not dense_fast_path
        ):
            should_invoke = True
            reasons.append("reasoning signal inside uncertainty band")
            logger.info(
                "Reasoning signal inside uncertainty band "
                "(confidence={:.3f}, multi_hop={:.2f}, cross_doc={}, graph_kw={}).",
                final_confidence,
                features.multi_hop_score,
                features.cross_doc_signals,
                features.graph_keyword_count,
            )

        if final_route == "clarify" and ambiguity_score < self.ambiguity_force_stage2_threshold:
            should_invoke = True
            reasons.append("clarify route with low ambiguity score")
            logger.info(
                "Stage 1 predicted 'clarify' but ambiguity score is low ({:.2f}). Forcing Stage 2.",
                ambiguity_score,
            )

        if reasoning_signal and not self.reasoning_force_stage2_enabled:
            reasons.append("reasoning signal observed but kept advisory")

        return should_invoke, reasons

    def _has_reasoning_signal(self, features: QueryFeatures) -> bool:
        """Return whether query features suggest multi-hop, legal-relation, or hybrid reasoning."""
        strong_structural_signal = (
            features.cross_doc_signals
            or features.complexity_level >= 3
            or features.sub_question_count >= 2
            or features.conditional_depth >= 2
        )
        # legal_reference_count >= 1 and graph_keyword_count >= 1 only make Stage 2 eligible 
        # when reasoning_force_stage2_enabled and confidence ceiling allows it.
        legal_relation_signal = (
            features.multi_hop_score >= self.reasoning_force_stage2_threshold
            or features.legal_reference_count >= 1
            or features.law_specificity >= 1
            or features.graph_keyword_count >= 1
            or features.authority_chain_count >= 1
            or features.legal_effect_count >= 1
            or features.procedural_count >= 1
            or features.multi_entity_relation_count >= 1
        )
        return strong_structural_signal or legal_relation_signal

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
                "law_specificity": output.features.law_specificity,
                "complexity_level": output.features.complexity_level,
                "sub_question_count": output.features.sub_question_count,
                "conditional_depth": output.features.conditional_depth,
                "authority_chain_count": output.features.authority_chain_count,
                "legal_effect_count": output.features.legal_effect_count,
                "procedural_count": output.features.procedural_count,
                "multi_entity_relation_count": output.features.multi_entity_relation_count,
                "cross_doc_signals": output.features.cross_doc_signals,
                "is_factoid": output.features.is_factoid,
                "multi_hop_verb_count": output.features.multi_hop_verb_count,
                "comparative_depth": output.features.comparative_depth,
            },
            "stage1": {
                "route": output.stage1_route,
                "confidence": output.stage1_confidence,
            },
            "base_route": output.base_route,
            "base_confidence": output.base_confidence,
            "ambiguity": {
                "score": output.features.ambiguity_score,
                "candidate_route": output.ambiguity_candidate_route,
                "override_allowed": output.ambiguity_override_allowed,
                "override_reason": output.ambiguity_override_reason,
            },
            "stage2_invoked": output.stage2_invoked,
            "stage2_trigger_reasons": output.stage2_trigger_reasons,
            "stage2": {
                "route": output.route if output.stage2_invoked else None,
                "confidence": output.confidence if output.stage2_invoked else None,
                "reasoning": output.reasoning if output.stage2_invoked else None,
                "trigger_reasons": output.stage2_trigger_reasons,
                "override": output.stage2_override,
                "override_reason": output.stage2_override_reason,
                "complexity_level": output.stage2_complexity_level,
                "reasoning_steps": output.stage2_reasoning_steps,
                "sub_questions": output.stage2_sub_questions,
                "ambiguity_flags": output.stage2_ambiguity_flags,
                "clarify_question": output.clarify_question,
                "parse_error": output.stage2_parse_error,
                "raw_route": output.stage2_raw_route,
                "stage2_raw_route": output.stage2_raw_route,
                "guardrail_applied": output.stage2_guardrail_applied,
                "guardrail_reason": output.stage2_guardrail_reason,
                "override_allowed": output.stage2_override_allowed,
                "override_policy_reason": output.stage2_override_policy_reason,
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
