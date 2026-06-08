# Demo Guide

Recommended defense demo: CLI routing demo. It is clearer than the current
frontend because it exposes route decisions, Stage 2 behavior, and resolved
conversation referents.

## Primary Demo

```bash
python scripts/demo_conversation_routing.py --config configs/config.yaml
```

Expected behavior:

| Scenario | Expected route behavior |
|---|---|
| Direct dense lookup | `dense_retrieval` / vector backend |
| Relation-heavy query | `graph_traversal` / graph backend |
| Pronoun with valid history | resolves `Nghị định 100/2019/NĐ-CP`, routes retrieval |
| Pronoun without history | `clarify` |
| Pronoun with irrelevant history | `clarify` |
| Missing entity | `clarify` |
| Multi-interpretation | `clarify` |

Pre-generated output:

```text
results_phase3/demo_conversation_routing_output.md
docs/final_results_snapshot/demo_conversation_routing_output.md
```

## Limited Conversation Evaluation

```bash
python scripts/evaluate_conversation_ambiguity.py --config configs/config.yaml --eval-file evaluation/conversation_ambiguity_eval.json --output-dir results_demo --limit 10 --use-cache
```

This is a diagnostic/stress test. It does not replace the strict 600-query
end-to-end benchmark.

## Strict Routing-only Sanity

```bash
python scripts/evaluate_strict_routing_only.py --config configs/config.yaml --test-file qa_pipeline/data/legal_strict/test.json --output-dir results_demo
```

This command checks routing only. It does not run retrieval or answer
generation and must not be used to replace `tab:end_to_end_results`.

## Optional Frontend

Frontend folder: `frontend/`.

```bash
cd frontend
npm ci
npm run build
npm run dev
```

The frontend posts to `http://localhost:8000/query` and can display route and
Stage 2 metadata, but it does not currently send conversation history or show
resolved referents. Use it only as an optional UI; use the CLI demo for the
research defense.

## Demo Script for Speaking

"Em demo phần routing vì đây là đóng góp chính. Câu tra cứu trực tiếp đi theo
dense retrieval; câu hỏi quan hệ pháp lý đi theo graph traversal. Khi câu hỏi
dùng 'văn bản đó' và history có Nghị định 100/2019/NĐ-CP, hệ thống resolve được
referent nên không hỏi lại. Nếu không có history hoặc history không liên quan,
router chuyển sang clarify để tránh trả lời nhầm. Điểm chính là hệ thống không
always GraphRAG, mà chọn route theo reasoning demand và ambiguity."
