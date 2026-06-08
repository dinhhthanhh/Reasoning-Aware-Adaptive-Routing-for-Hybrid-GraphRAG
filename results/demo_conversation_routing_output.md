# Demo Conversation Routing Output

This demo runs the current router with fixed scenarios. Retrieval and final answer generation are skipped; non-clarify routes show the backend that would be used.

## Direct dense lookup

- Expected: `dense_retrieval`
- History: [empty]
- User query: Điều kiện kết hôn theo Luật Hôn nhân và gia đình gồm những gì?
- Stage 1 route: `dense_retrieval`
- Stage 1 confidence: `0.9162303643953668`
- Ambiguity score: `0.0`
- Has pronoun: `False`
- History length: `0`
- History resolves ambiguity feature: `False`
- Stage 2 triggered: `False`
- Stage 2 override: `False`
- Final route: `dense_retrieval`
- Resolved referent: `not_available`
- Retrieved backend: `vector`
- Final answer / clarification: [generation skipped: routing-only demo]

<details>
<summary>Router reasoning</summary>

Stage 1: route=dense_retrieval, confidence=0.916 (Stage 2 skipped by policy)

</details>

## Relation-heavy graph/hybrid

- Expected: `graph_traversal`
- History: [empty]
- User query: Quyết định 732/QĐ-UBND bãi bỏ hoặc sửa đổi quy định nào?
- Stage 1 route: `dense_retrieval`
- Stage 1 confidence: `0.8727272738306976`
- Ambiguity score: `0.3`
- Has pronoun: `False`
- History length: `0`
- History resolves ambiguity feature: `False`
- Stage 2 triggered: `True`
- Stage 2 override: `True`
- Final route: `graph_traversal`
- Resolved referent: `not_available`
- Retrieved backend: `graph`
- Final answer / clarification: [generation skipped: routing-only demo]

<details>
<summary>Router reasoning</summary>

Stage 1: dense_retrieval(0.873) → Stage 2 rescue override: graph_traversal(0.950) | policy=rescue_dense_to_graph_doc_specific_lookup | complexity=moderate | reason=Câu hỏi chứa số hiệu văn bản cụ thể 'Quyết định 732/QĐ-UBND', thuộc trường hợp doc-specific lookup yêu cầu graph_traversal để truy xuất metadata và quan hệ bãi bỏ/sửa đổi, không phù hợp với dense_retrieval.

</details>

## Pronoun with valid history

- Expected: `not_clarify`
- History: Người dùng đang hỏi về Nghị định 100/2019/NĐ-CP về xử phạt vi phạm hành chính trong lĩnh vực giao thông đường bộ.
- User query: Văn bản đó còn hiệu lực không?
- Stage 1 route: `clarify`
- Stage 1 confidence: `0.6129032311094662`
- Ambiguity score: `0.6`
- Has pronoun: `False`
- History length: `1`
- History resolves ambiguity feature: `False`
- Stage 2 triggered: `True`
- Stage 2 override: `False`
- Final route: `clarify`
- Resolved referent: `not_available`
- Retrieved backend: `none`
- Final answer / clarification: Bạn đang hỏi về hiệu lực của văn bản nào? (Ví dụ: Nghị định 100/2019/NĐ-CP, hay một văn bản sửa đổi/bãi bỏ nó?)

<details>
<summary>Router reasoning</summary>

Stage 1: clarify(0.613) → Stage 2 confirmed: clarify(0.950) | complexity=ambiguous

</details>

## Pronoun without history

- Expected: `clarify`
- History: [empty]
- User query: Văn bản đó còn hiệu lực không?
- Stage 1 route: `clarify`
- Stage 1 confidence: `0.6129032311094662`
- Ambiguity score: `0.6`
- Has pronoun: `False`
- History length: `0`
- History resolves ambiguity feature: `False`
- Stage 2 triggered: `True`
- Stage 2 override: `False`
- Final route: `clarify`
- Resolved referent: `not_available`
- Retrieved backend: `none`
- Final answer / clarification: Bạn vui lòng cung cấp số hiệu, tên hoặc nội dung cụ thể của văn bản pháp luật mà bạn đang muốn hỏi về tình trạng hiệu lực.

<details>
<summary>Router reasoning</summary>

Stage 1: clarify(0.613) → Stage 2 confirmed: clarify(0.950) | complexity=ambiguous

</details>

## Pronoun with irrelevant history

- Expected: `clarify`
- History: Người dùng hỏi cách tra cứu văn bản pháp luật trên cổng thông tin điện tử, nhưng chưa nêu văn bản cụ thể.
- User query: Văn bản đó còn hiệu lực không?
- Stage 1 route: `clarify`
- Stage 1 confidence: `0.6129032311094662`
- Ambiguity score: `0.6`
- Has pronoun: `False`
- History length: `1`
- History resolves ambiguity feature: `False`
- Stage 2 triggered: `True`
- Stage 2 override: `False`
- Final route: `clarify`
- Resolved referent: `not_available`
- Retrieved backend: `none`
- Final answer / clarification: Bạn đang muốn kiểm tra hiệu lực của văn bản pháp luật nào? Vui lòng cung cấp tên hoặc số hiệu của văn bản (ví dụ: Nghị định 10/2020/NĐ-CP, Luật Đất đai 2024).

<details>
<summary>Router reasoning</summary>

Stage 1: clarify(0.613) → Stage 2 confirmed: clarify(0.950) | complexity=ambiguous

</details>

## Missing entity

- Expected: `clarify`
- History: [empty]
- User query: Mức phạt trong trường hợp này là bao nhiêu?
- Stage 1 route: `clarify`
- Stage 1 confidence: `0.6125486227663145`
- Ambiguity score: `0.9`
- Has pronoun: `True`
- History length: `0`
- History resolves ambiguity feature: `False`
- Stage 2 triggered: `True`
- Stage 2 override: `False`
- Final route: `clarify`
- Resolved referent: `not_available`
- Retrieved backend: `none`
- Final answer / clarification: Bạn vui lòng cung cấp thêm thông tin về 'trường hợp này' đang đề cập, bao gồm: hành vi cụ thể, chủ thể liên quan, hoặc lĩnh vực pháp lý để tôi có thể tra cứu mức phạt chính xác.

<details>
<summary>Router reasoning</summary>

Stage 1: clarify(0.613) → Stage 2 confirmed: clarify(0.950) | complexity=ambiguous

</details>

## Multi-interpretation

- Expected: `clarify`
- History: [empty]
- User query: Doanh nghiệp có bị phạt không?
- Stage 1 route: `clarify`
- Stage 1 confidence: `0.8285985448768098`
- Ambiguity score: `0.0`
- Has pronoun: `False`
- History length: `0`
- History resolves ambiguity feature: `False`
- Stage 2 triggered: `True`
- Stage 2 override: `False`
- Final route: `clarify`
- Resolved referent: `not_available`
- Retrieved backend: `none`
- Final answer / clarification: Bạn vui lòng cung cấp thêm thông tin về: 1) Lĩnh vực hoạt động của doanh nghiệp (ví dụ: xây dựng, tài chính, thương mại...); 2) Hành vi cụ thể mà doanh nghiệp đã thực hiện hoặc dự định thực hiện; 3) Văn bản pháp luật hoặc quy định cụ thể mà bạn đang quan tâm (nếu có).

<details>
<summary>Router reasoning</summary>

Stage 1: clarify(0.829) → Stage 2 confirmed: clarify(0.950) | complexity=ambiguous

</details>
