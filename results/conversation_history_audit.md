# Conversation History Audit

## Scope

This audit checks whether the current routing and retrieval pipeline can accept conversation history. It is a static code audit plus artifact inspection; no router logic, prompts, model checkpoint, dataset, vector DB, or graph was changed.

## Current Support

- `pipeline/hybrid_pipeline.py` passes the resolved `history_str` into `TwoStageRouter.route(...)`, graph retrieval, hybrid retrieval, and clarification generation.
- `router/two_stage_router.py` accepts `history: str | None` and passes it to `AmbiguityDetector.detect(...)`, `FeatureExtractor.extract(...)`, and `LLMReasoningVerifier.verify(...)`.
- `router/features.py` includes two history-aware Stage 1 features: `history_length` and `history_resolves_ambiguity`.
- `router/ambiguity_detector.py` lowers pronoun ambiguity when some history exists.
- `router/llm_reasoning_verifier.py` includes an explicit conversation-history block in the Stage 2 prompt.
- `rag/graph_rag_adapter.py` includes `History:` in the graph answer prompt, so graph-generation can use history after routing.

## Strengths

- History is threaded through the main pipeline instead of being handled only at the UI level.
- Stage 2 has access to history, which is the right place to resolve conversational references such as "văn bản đó" or "quy định này".
- Clarification generation has a fallback path, so the system can still ask a follow-up question if LLM clarification fails.

## Risks

- The strict QA split and the constructed ambiguity benchmark currently use empty histories, so the reported metrics do not prove conversation-history resolution.
- `history_resolves_ambiguity` is a shallow heuristic. It only checks for a few literal pronoun terms in history, not whether a specific legal document, article, agency, or time period has been resolved.
- `AmbiguityDetector` reduces pronoun ambiguity if any history exists. Irrelevant history can therefore make an unresolved query look less risky.
- Stage 1 has no trained clarify class in the strict training data, so history-sensitive clarification still depends on heuristic ambiguity detection and Stage 2.
- There is no dedicated regression test for: unresolved history, resolved history, irrelevant history, or conflicting history.

## Phase 2 Recommendations

1. Add a small conversation-aware benchmark with paired cases:
   unresolved query with empty history, the same query with resolving history, and the same query with irrelevant history.
2. Store structured history fields in eval data, not only free-form transcript text.
3. Add entity-linking uncertainty features such as candidate legal document count, article candidate count, and whether the last referenced legal target is unique.
4. Add tests around `AmbiguityDetector.detect(query, history)` and `FeatureExtractor.extract(query, history)` for pronoun and demonstrative-reference cases.
5. Report conversation-history metrics separately from the current strict routing benchmark.

## Current Conclusion

Conversation history is architecturally supported, but it is not yet experimentally validated. For the current thesis demo, history can be shown as a capability, while the paper should continue to describe semantic ambiguity and conversation grounding as future work unless a dedicated history benchmark is added.
