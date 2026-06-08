# Conversation Ambiguity Summary

- Eval file: `evaluation\conversation_ambiguity_eval.json`
- Mode: `two_stage_current`
- Limit: `None`
- Total samples: `160`
- Route accuracy: `0.750`
- Clarify precision: `0.963`
- Clarify recall: `0.780`
- Clarify F1: `0.862`
- Stage 2 trigger rate: `0.825`
- Stage 2 override rate: `0.22727272727272727`
- Avg latency ms: `3127.231151250089`
- History resolution accuracy: `0.85`
- History non-clarify proxy: `0.85`

## Category Distribution

| Category | Count |
|---|---:|
| `answerable_with_history` | 20 |
| `clarify_without_history` | 20 |
| `irrelevant_history` | 20 |
| `conflicting_history` | 20 |
| `missing_entity` | 20 |
| `multi_interpretation` | 20 |
| `clear_dense_control` | 20 |
| `clear_graph_or_hybrid_control` | 20 |

## Metrics by Type

| Type | Total | Route Acc. | Clarify R | Stage2 rate | Pred routes |
|---|---:|---:|---:|---:|---|
| `answerable_with_history` | 20 | 0.600 | n/a | 0.85 | `{'graph_traversal': 9, 'dense_retrieval': 8, 'clarify': 3}` |
| `clarify_without_history` | 20 | 0.850 | 0.850 | 1.0 | `{'clarify': 17, 'dense_retrieval': 3}` |
| `clear_dense_control` | 20 | 1.000 | n/a | 0.15 | `{'dense_retrieval': 20}` |
| `clear_graph_or_hybrid_control` | 20 | 0.500 | n/a | 0.95 | `{'graph_traversal': 13, 'dense_retrieval': 4, 'hybrid_reasoning': 3}` |
| `conflicting_history` | 20 | 0.700 | 0.700 | 0.9 | `{'clarify': 14, 'dense_retrieval': 5, 'graph_traversal': 1}` |
| `irrelevant_history` | 20 | 0.750 | 0.750 | 0.9 | `{'clarify': 15, 'dense_retrieval': 4, 'graph_traversal': 1}` |
| `missing_entity` | 20 | 0.850 | 0.850 | 1.0 | `{'clarify': 17, 'dense_retrieval': 3}` |
| `multi_interpretation` | 20 | 0.750 | 0.750 | 0.85 | `{'clarify': 15, 'dense_retrieval': 5}` |

## False Counts

- `false_clarification_on_answerable_with_history`: `3`
- `false_answer_on_clarify_without_history`: `3`
- `false_answer_on_irrelevant_history`: `5`
- `false_answer_on_conflicting_history`: `6`

## Top Failure Examples

- `conv_0012` `conflicting_history` expected `clarify`, predicted `dense_retrieval`: Quy định đó áp dụng cho đối tượng nào?
- `conv_0013` `answerable_with_history` expected `graph_traversal`, predicted `dense_retrieval`: Thủ tục đó do cơ quan nào giải quyết?
- `conv_0015` `irrelevant_history` expected `clarify`, predicted `dense_retrieval`: Thủ tục đó do cơ quan nào giải quyết?
- `conv_0016` `conflicting_history` expected `clarify`, predicted `dense_retrieval`: Thủ tục đó do cơ quan nào giải quyết?
- `conv_0017` `answerable_with_history` expected `hybrid_reasoning`, predicted `clarify`: Nội dung trên có phải căn cứ để xử phạt không?
- `conv_0020` `conflicting_history` expected `clarify`, predicted `dense_retrieval`: Nội dung trên có phải căn cứ để xử phạt không?
- `conv_0025` `answerable_with_history` expected `hybrid_reasoning`, predicted `graph_traversal`: Quy định này có được áp dụng đồng thời với quy định về ngân sách không?
- `conv_0033` `answerable_with_history` expected `graph_traversal`, predicted `dense_retrieval`: Cơ quan đó có thẩm quyền ban hành quyết định không?
- `conv_0034` `clarify_without_history` expected `clarify`, predicted `dense_retrieval`: Cơ quan đó có thẩm quyền ban hành quyết định không?
- `conv_0036` `conflicting_history` expected `clarify`, predicted `graph_traversal`: Cơ quan đó có thẩm quyền ban hành quyết định không?
- `conv_0037` `answerable_with_history` expected `hybrid_reasoning`, predicted `dense_retrieval`: Trường hợp này có được miễn giảm tiền thuê đất không?
- `conv_0039` `irrelevant_history` expected `clarify`, predicted `dense_retrieval`: Trường hợp này có được miễn giảm tiền thuê đất không?
- `conv_0046` `clarify_without_history` expected `clarify`, predicted `dense_retrieval`: Quy định đó bị bãi bỏ bởi văn bản nào?
- `conv_0047` `irrelevant_history` expected `clarify`, predicted `graph_traversal`: Quy định đó bị bãi bỏ bởi văn bản nào?
- `conv_0051` `irrelevant_history` expected `clarify`, predicted `dense_retrieval`: Nội dung đó yêu cầu hồ sơ gồm những gì?
