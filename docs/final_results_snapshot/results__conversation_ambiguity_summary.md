# Conversation Ambiguity Summary

- Eval file: `evaluation\conversation_ambiguity_eval.json`
- Mode: `two_stage_current`
- Limit: `None`
- Total samples: `160`
- Route accuracy: `0.475`
- Clarify precision: `0.907`
- Clarify recall: `0.390`
- Clarify F1: `0.545`
- Stage 2 trigger rate: `0.7125`
- Stage 2 override rate: `0.3333333333333333`
- Avg latency ms: `2238.7177831249574`
- History resolution accuracy: `not_available`
- History non-clarify proxy: `0.8`

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
| `answerable_with_history` | 20 | 0.550 | n/a | 0.8 | `{'clarify': 4, 'dense_retrieval': 10, 'graph_traversal': 6}` |
| `clarify_without_history` | 20 | 0.650 | 0.650 | 0.85 | `{'clarify': 13, 'dense_retrieval': 7}` |
| `clear_dense_control` | 20 | 1.000 | n/a | 0.05 | `{'dense_retrieval': 20}` |
| `clear_graph_or_hybrid_control` | 20 | 0.300 | n/a | 0.9 | `{'graph_traversal': 9, 'dense_retrieval': 10, 'hybrid_reasoning': 1}` |
| `conflicting_history` | 20 | 0.150 | 0.150 | 0.8 | `{'clarify': 3, 'dense_retrieval': 9, 'graph_traversal': 8}` |
| `irrelevant_history` | 20 | 0.300 | 0.300 | 0.8 | `{'clarify': 6, 'dense_retrieval': 14}` |
| `missing_entity` | 20 | 0.600 | 0.600 | 0.9 | `{'clarify': 12, 'dense_retrieval': 8}` |
| `multi_interpretation` | 20 | 0.250 | 0.250 | 0.6 | `{'clarify': 5, 'dense_retrieval': 15}` |

## False Counts

- `false_clarification_on_answerable_with_history`: `4`
- `false_answer_on_clarify_without_history`: `7`
- `false_answer_on_irrelevant_history`: `14`
- `false_answer_on_conflicting_history`: `17`

## Top Failure Examples

- `conv_0001` `answerable_with_history` expected `graph_traversal`, predicted `clarify`: Văn bản đó còn hiệu lực không?
- `conv_0005` `answerable_with_history` expected `dense_retrieval`, predicted `clarify`: Điều này quy định điều kiện gì?
- `conv_0011` `irrelevant_history` expected `clarify`, predicted `dense_retrieval`: Quy định đó áp dụng cho đối tượng nào?
- `conv_0012` `conflicting_history` expected `clarify`, predicted `dense_retrieval`: Quy định đó áp dụng cho đối tượng nào?
- `conv_0013` `answerable_with_history` expected `graph_traversal`, predicted `dense_retrieval`: Thủ tục đó do cơ quan nào giải quyết?
- `conv_0014` `clarify_without_history` expected `clarify`, predicted `dense_retrieval`: Thủ tục đó do cơ quan nào giải quyết?
- `conv_0015` `irrelevant_history` expected `clarify`, predicted `dense_retrieval`: Thủ tục đó do cơ quan nào giải quyết?
- `conv_0016` `conflicting_history` expected `clarify`, predicted `dense_retrieval`: Thủ tục đó do cơ quan nào giải quyết?
- `conv_0017` `answerable_with_history` expected `hybrid_reasoning`, predicted `dense_retrieval`: Nội dung trên có phải căn cứ để xử phạt không?
- `conv_0018` `clarify_without_history` expected `clarify`, predicted `dense_retrieval`: Nội dung trên có phải căn cứ để xử phạt không?
- `conv_0019` `irrelevant_history` expected `clarify`, predicted `dense_retrieval`: Nội dung trên có phải căn cứ để xử phạt không?
- `conv_0020` `conflicting_history` expected `clarify`, predicted `dense_retrieval`: Nội dung trên có phải căn cứ để xử phạt không?
- `conv_0024` `conflicting_history` expected `clarify`, predicted `graph_traversal`: Văn bản đó sửa đổi quy định nào?
- `conv_0025` `answerable_with_history` expected `hybrid_reasoning`, predicted `dense_retrieval`: Quy định này có được áp dụng đồng thời với quy định về ngân sách không?
- `conv_0027` `irrelevant_history` expected `clarify`, predicted `dense_retrieval`: Quy định này có được áp dụng đồng thời với quy định về ngân sách không?
