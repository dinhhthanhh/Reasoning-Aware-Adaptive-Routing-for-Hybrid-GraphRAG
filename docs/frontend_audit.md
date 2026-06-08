# Frontend Readiness Audit

Generated for Phase 5 reproducibility and GitHub readiness.

## Scope

This audit checks whether the current frontend is suitable for a thesis
defense demo. It does not change benchmark results and does not start a
long-running development server.

## Current Frontend

- Framework: Next.js 15.1.6 with React 19.
- Package manager command used: `npm run build` in `frontend/`.
- Build result: PASS. The production build completed successfully.
- Main page: `frontend/src/app/page.tsx`.
- Backend endpoint: hard-coded `http://localhost:8000/query`.
- Request payload: sends the current message as `query`.
- Displayed metadata: route, confidence, Stage 2 flag, reasoning,
  latency, and sources when returned by the backend.

## Demo Suitability

The frontend is suitable for a basic single-turn visual demo:

- enter a Vietnamese legal question;
- show the generated answer;
- show the selected route and Stage 2 metadata;
- show latency and sources if available.

For the defense demo of the Phase 3 contribution, the CLI demo remains the
recommended path. The reason is that the current frontend does not send an
explicit conversation history payload and does not display the full
history-resolution metadata that the conversation stress test uses.

## Gaps

- The API base URL is hard-coded instead of using a variable such as
  `NEXT_PUBLIC_API_BASE_URL`.
- The request payload does not include explicit conversation history.
- The UI does not display `history_resolution_status`,
  `resolved_referent`, or `query_has_contextual_reference`.
- The UI is therefore not the best surface for demonstrating why
  "van ban do" can be resolved when history contains a concrete legal
  document and clarified when history is absent or irrelevant.

## Recommendation

Use the CLI demo as the primary defense demo:

```bash
python scripts/demo_conversation_routing.py --config configs/config.yaml
```

Use the frontend as an optional product-style demo only after starting the
backend API locally:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
cd frontend
npm run dev
```

Future frontend improvement should add configurable API base URL, explicit
history payload support, and display fields for resolved referents and
history-resolution status.
