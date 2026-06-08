#!/usr/bin/env python3
"""Build a deterministic synthetic clarification/ambiguity routing eval set."""

from __future__ import annotations

import json
from pathlib import Path


TOPICS = [
    "lao động",
    "bảo hiểm xã hội",
    "đất đai",
    "giao thông đường bộ",
    "hàng không dân dụng",
    "khám bệnh, chữa bệnh",
    "môi trường",
    "đầu tư công",
    "doanh nghiệp",
    "thuế",
    "xây dựng",
    "giáo dục",
    "hóa chất",
]

DOCS = [
    "Luật Đường bộ 2024",
    "Bộ luật Lao động 2019",
    "Luật Đất đai 2024",
    "Luật Khám bệnh, chữa bệnh 2023",
    "Nghị định 75/2026/NĐ-CP",
    "Thông tư 02/2026/TT-BYT",
    "Quyết định 447/QĐ-BCT",
    "Chỉ thị 09/CT-TTg",
]

AGENCIES = [
    "Ủy ban nhân dân cấp tỉnh",
    "Bộ Y tế",
    "Bộ Công Thương",
    "Cục Hàng không Việt Nam",
    "Sở Khoa học và Công nghệ",
    "Cục Quản lý hóa chất",
]


def add(rows: list[dict[str, object]], query: str, route: str, ambiguity_type: str = "", complexity: str = "low") -> None:
    rows.append(
        {
            "id": f"clarify_eval_{len(rows) + 1:03d}",
            "query": query,
            "expected_route": route,
            "ambiguity_type": ambiguity_type,
            "expected_complexity": complexity,
            "history": [],
        }
    )


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    # Positive clarify cases: underspecified references.
    for topic in TOPICS * 3:
        add(rows, f"Quy định này áp dụng cho trường hợp nào trong lĩnh vực {topic}?", "clarify", "incomplete_context")

    # Positive clarify cases: missing legal entity/document.
    for topic in TOPICS * 3:
        add(rows, f"Tôi cần nộp hồ sơ đó ở đâu khi làm thủ tục về {topic}?", "clarify", "missing_entity")

    # Positive clarify cases: pronoun/reference ambiguity.
    pronoun_templates = [
        "Ông ấy có quyền yêu cầu bồi thường theo quy định không?",
        "Bà ấy có được miễn hoặc giảm nghĩa vụ đó không?",
        "Họ phải gửi văn bản này cho cơ quan nào?",
        "Người đó có bị xử phạt trong trường hợp này không?",
        "Bên đó có phải chịu trách nhiệm theo hợp đồng không?",
        "Cơ quan đó có thẩm quyền giải quyết việc này không?",
    ]
    for text in pronoun_templates * 7:
        add(rows, text, "clarify", "pronoun_reference")

    # Positive clarify cases: multiple plausible interpretations.
    multi_templates = [
        "Ngân hàng có phải báo cáo khi xử lý tài sản bảo đảm không?",
        "Cảng có trách nhiệm gì khi tiếp nhận hàng hóa nguy hiểm?",
        "Quỹ có được cấp kinh phí cho nhiệm vụ này không?",
        "Ban đó có được phê chuẩn danh sách thành viên không?",
        "Cơ sở đó có được tiếp tục hoạt động sau thời điểm chuyển tiếp không?",
        "Đơn vị này có được ký thay khi cần thiết không?",
    ]
    for text in multi_templates * 6:
        add(rows, text, "clarify", "multi_interpretation", "medium")

    # Negative controls: clear dense retrieval questions.
    dense_templates = [
        "Theo {doc}, thời điểm có hiệu lực của văn bản là khi nào?",
        "Theo {doc}, hồ sơ đổi thẻ Sỹ quan kiểm tra tàu biển gồm những gì?",
        "Theo {doc}, nội dung của một phép bay bao gồm những gì?",
        "Theo {doc}, cơ sở dữ liệu đường bộ bao gồm những nội dung nào?",
        "Theo {doc}, tài khoản giao thông là gì?",
        "Theo {doc}, ai có trách nhiệm lập kế hoạch hoạt động bay dân dụng?",
    ]
    for i in range(26):
        add(rows, dense_templates[i % len(dense_templates)].format(doc=DOCS[i % len(DOCS)]), "dense_retrieval")

    # Negative controls: clear graph traversal questions.
    graph_templates = [
        "Theo {doc1} và {doc2}, cơ quan nào có thẩm quyền xử lý hồ sơ chuyển tiếp?",
        "Mối quan hệ giữa {doc1} và {doc2} về thẩm quyền giải quyết thủ tục là gì?",
        "Căn cứ {doc1} và {doc2}, quy định nào xác định trách nhiệm báo cáo của cơ quan nhà nước?",
        "Đối chiếu {doc1} với {doc2}, văn bản nào quy định việc bãi bỏ thủ tục hành chính?",
    ]
    for i in range(26):
        add(
            rows,
            graph_templates[i % len(graph_templates)].format(
                doc1=DOCS[i % len(DOCS)],
                doc2=DOCS[(i + 3) % len(DOCS)],
            ),
            "graph_traversal",
            complexity="medium",
        )

    # Negative controls: clear hybrid reasoning questions.
    hybrid_templates = [
        "Nếu {agency} triển khai dự án trong lĩnh vực {topic}, đồng thời hồ sơ đã nộp trước ngày văn bản mới có hiệu lực, thì có được tiếp tục xử lý theo quy định cũ không?",
        "Nếu {agency} vừa thực hiện thủ tục về {topic} vừa phải tuân thủ kế hoạch chuyển đổi số, thì cần căn cứ những nhóm quy định nào?",
        "Trường hợp {agency} sử dụng vốn vay ưu đãi cho nhiệm vụ thuộc lĩnh vực {topic}, có được áp dụng đồng thời quy định về ngân sách và thủ tục chuyên ngành không?",
    ]
    for i in range(26):
        add(
            rows,
            hybrid_templates[i % len(hybrid_templates)].format(
                agency=AGENCIES[i % len(AGENCIES)],
                topic=TOPICS[i % len(TOPICS)],
            ),
            "hybrid_reasoning",
            complexity="high",
        )

    if len(rows) != 234:
        raise RuntimeError(f"Expected 234 rows, got {len(rows)}")
    return rows


def main() -> None:
    output = Path("evaluation/legal_clarify_eval.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_rows(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output} with 234 rows")


if __name__ == "__main__":
    main()
