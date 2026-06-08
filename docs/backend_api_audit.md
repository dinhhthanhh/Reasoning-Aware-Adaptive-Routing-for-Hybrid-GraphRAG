# Backend API Readiness Audit

Generated for Phase 5 reproducibility and GitHub readiness.

## Scope

This audit checks the current FastAPI wrapper used by the web demo. It does
not run full benchmarks, rebuild indexes, or modify model outputs.

## Current API

- Entry point: `api/main.py`.
- Main endpoint: `POST /query`.
- Request model fields: `query`, `session_id`, and `verbose`.
- Response model fields: `answer`, `route_used`, `confidence`,
  `router_reasoning`, `stage2_invoked`, `stage2_override`, `sources`,
  `latency_ms`, and `is_ambiguous`.
- Backend pipeline: initializes `HybridGraphRAGPipeline` from
  `configs/config.yaml`.
- Compile check: `python -m compileall api` passed.

## Demo Suitability

The backend API is suitable for a basic one-turn web demo when the local
configuration, vector store, Neo4j, and LLM endpoint are available. It can
return the selected route, confidence, Stage 2 metadata, answer text, and
sources.

The API is not yet the best surface for the Phase 3 conversation-history
demo because it does not expose an explicit `history` request field and does
not return dedicated fields such as `resolved_referent` or
`history_resolution_status`. Those details are currently demonstrated more
clearly by:

```bash
python scripts/demo_conversation_routing.py --config configs/config.yaml
```

## Recommendation

For thesis defense:

- primary demo: CLI conversation-routing demo;
- optional UI demo: FastAPI plus Next.js frontend for a one-turn query;
- avoid presenting the frontend as the definitive Phase 3 history-resolution
  interface until the API and UI expose the history-resolution fields.

Suggested future API additions:

- add an explicit `history` field to `QueryRequest`;
- return `stage1_route`, `final_route`, `history_resolution_status`,
  `resolved_referent`, and `query_has_contextual_reference`;
- make the frontend API URL configurable through `NEXT_PUBLIC_API_BASE_URL`.
