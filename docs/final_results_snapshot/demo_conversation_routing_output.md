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
- History resolution status: `not_needed`
- History resolution confidence: `0.0`
- Query has contextual reference: `False`
- Stage 2 triggered: `False`
- Stage 2 override: `False`
- Final route: `dense_retrieval`
- Resolved referent: `None`
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
- History resolution status: `not_needed`
- History resolution confidence: `0.0`
- Query has contextual reference: `False`
- Stage 2 triggered: `True`
- Stage 2 override: `True`
- Final route: `graph_traversal`
- Resolved referent: `None`
- Retrieved backend: `graph`
- Final answer / clarification: [generation skipped: routing-only demo]

<details>
<summary>Router reasoning</summary>

Stage 1: dense_retrieval(0.873) → Stage 2 rescue override: graph_traversal(0.950) | policy=rescue_dense_to_graph_strong_reasoning_signal | complexity=moderate | reason=Câu hỏi chứa số hiệu văn bản cụ thể 'Quyết định 732/QĐ-UBND'. Theo nguyên tắc bảo toàn tín hiệu pháp lý, các câu hỏi tra cứu nội dung, hiệu lực, hoặc quan hệ bãi bỏ/sửa đổi của một văn bản cụ thể (doc-specific lookup) phải ưu tiên graph_traversal thay vì dense_retrieval để đảm bảo truy xuất chính xác dựa trên metadata và quan hệ pháp lý.

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
- History resolves ambiguity feature: `True`
- History resolution status: `resolved`
- History resolution confidence: `0.92`
- Query has contextual reference: `True`
- Stage 2 triggered: `True`
- Stage 2 override: `False`
- Final route: `graph_traversal`
- Resolved referent: `Nghị định 100/2019/NĐ-CP`
- Retrieved backend: `graph`
- Final answer / clarification: [generation skipped: routing-only demo]

<details>
<summary>Router reasoning</summary>

Stage 1: graph_traversal(0.920) → Stage 2 confirmed: graph_traversal(0.950) | complexity=moderate | reason=Câu hỏi yêu cầu tra cứu hiệu lực của một văn bản pháp lý cụ thể (Nghị định 100/2019/NĐ-CP) đã được xác định từ lịch sử hội thoại. Theo nguyên tắc bảo toàn tín hiệu pháp lý, các câu hỏi doc-specific (có số hiệu văn bản) về hiệu lực, bãi bỏ, sửa đổi phải ưu tiên graph_traversal để truy xuất quan hệ pháp lý trong cơ sở dữ liệu, thay vì clarify hay dense_retrieval. | policy=resolved_history_unblocks_retrieval

</details>

## Pronoun without history

- Expected: `clarify`
- History: [empty]
- User query: Văn bản đó còn hiệu lực không?
- Stage 1 route: `clarify`
- Stage 1 confidence: `1.0`
- Ambiguity score: `0.9`
- Has pronoun: `True`
- History length: `0`
- History resolves ambiguity feature: `False`
- History resolution status: `no_history`
- History resolution confidence: `0.0`
- Query has contextual reference: `True`
- Stage 2 triggered: `True`
- Stage 2 override: `False`
- Final route: `clarify`
- Resolved referent: `None`
- Retrieved backend: `none`
- Final answer / clarification: Bạn đang muốn hỏi về hiệu lực của văn bản pháp lý nào? Vui lòng cung cấp tên hoặc số hiệu của văn bản (ví dụ: Luật X, Nghị định Y, Thông tư Z).

<details>
<summary>Router reasoning</summary>

Stage 1: clarify(0.900) → Stage 2 confirmed: clarify(1.000) | complexity=ambiguous

</details>

## Pronoun with irrelevant history

- Expected: `clarify`
- History: Người dùng hỏi cách tra cứu văn bản pháp luật trên cổng thông tin điện tử, nhưng chưa nêu văn bản cụ thể.
- User query: Văn bản đó còn hiệu lực không?
- Stage 1 route: `clarify`
- Stage 1 confidence: `1.0`
- Ambiguity score: `0.85`
- Has pronoun: `True`
- History length: `1`
- History resolves ambiguity feature: `False`
- History resolution status: `irrelevant_history`
- History resolution confidence: `0.0`
- Query has contextual reference: `True`
- Stage 2 triggered: `True`
- Stage 2 override: `False`
- Final route: `clarify`
- Resolved referent: `None`
- Retrieved backend: `none`
- Final answer / clarification: Bạn đang muốn kiểm tra hiệu lực của văn bản pháp luật cụ thể nào? Vui lòng cung cấp tên hoặc số hiệu của văn bản.

<details>
<summary>Router reasoning</summary>

Stage 1: clarify(0.850) → Stage 2 confirmed: clarify(0.950) | complexity=ambiguous

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
- History resolution status: `no_history`
- History resolution confidence: `0.0`
- Query has contextual reference: `True`
- Stage 2 triggered: `True`
- Stage 2 override: `False`
- Final route: `clarify`
- Resolved referent: `None`
- Retrieved backend: `none`
- Final answer / clarification: Bạn vui lòng cung cấp thêm thông tin về 'trường hợp này' (ví dụ: hành vi vi phạm, lĩnh vực pháp lý, hoặc văn bản pháp luật liên quan) để tôi có thể tra cứu mức phạt chính xác.

<details>
<summary>Router reasoning</summary>

Stage 1: clarify(0.900) → Stage 2 confirmed: clarify(0.950) | complexity=ambiguous

</details>

## Multi-interpretation

- Expected: `clarify`
- History: [empty]
- User query: Doanh nghiệp có bị phạt không?
- Stage 1 route: `clarify`
- Stage 1 confidence: `0.8285985448768098`
- Ambiguity score: `0.85`
- Has pronoun: `False`
- History length: `0`
- History resolves ambiguity feature: `False`
- History resolution status: `not_needed`
- History resolution confidence: `0.0`
- Query has contextual reference: `False`
- Stage 2 triggered: `True`
- Stage 2 override: `False`
- Final route: `clarify`
- Resolved referent: `None`
- Retrieved backend: `none`
- Final answer / clarification: Bạn vui lòng cung cấp thêm thông tin về hành vi cụ thể mà doanh nghiệp đã thực hiện hoặc lĩnh vực pháp lý liên quan để tôi có thể xác định xem có bị phạt hay không?

<details>
<summary>Router reasoning</summary>

Stage 1: clarify(0.850) → Stage 2 confirmed: clarify(0.950) | complexity=ambiguous

</details>
