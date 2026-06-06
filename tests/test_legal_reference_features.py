"""Unit tests for legal reference feature extraction.

Validates that the regex patterns in router/features.py correctly
identify Vietnamese legal document references, article/clause numbers,
VBHN patterns, and graph keyword signals.

Uses a fake NER model to avoid loading the real Transformer model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router.features import FeatureExtractor


# Fake NER model to avoid loading the real Transformer
class FakeNER:
    def extract(self, texts: list[str]) -> list[list[Any]]:
        return [[] for _ in texts]


def _make_extractor() -> FeatureExtractor:
    """Create a FeatureExtractor with a fake NER model and minimal config."""
    return FeatureExtractor(
        ner_model=FakeNER(),
        config={
            "language": "vi",
            "router": {
                "scoring": {
                    "multi_hop_norm_divisor": 3.0,
                    "comparison_boost": 0.7,
                    "cross_doc_boost": 0.9,
                }
            },
        },
    )


class TestFeatureExtractorLegalRef:
    """Validate that the full extractor returns nonzero legal_reference_count
    and graph_keyword_count for the required test queries."""

    @pytest.fixture(scope="class")
    def extractor(self) -> FeatureExtractor:
        return _make_extractor()

    def test_legal_reference_count_nonzero(self, extractor: FeatureExtractor) -> None:
        cases = [
            "Thông tư số 77/2017/TT-BTC còn hiệu lực không?",
            "Quyết định 846/QĐ-BNNMT bãi bỏ thủ tục nào?",
            "Điều 1 Quyết định 25/2026/QĐ-UBND nói về nội dung gì?",
            "Ai là cơ quan ban hành VBHN-BXD 12?",
            "Theo khoản 22 Điều 13 Nghị định số 50/2026/NĐ-CP, doanh nghiệp có cần khai thuế không?",
            "Văn bản 47/VBHN-VPQH quy định gì về bảo vệ quyền lợi người tiêu dùng?",
            "Luật Kinh doanh Bảo hiểm quy định gì về lưu trữ hồ sơ nghiệp vụ?",
            "Luật Bảo vệ quyền lợi người tiêu dùng áp dụng trong trường hợp nào?",
        ]
        for query in cases:
            features = extractor.extract(query)
            assert features.legal_reference_count >= 1, (
                f"legal_reference_count should be >= 1 for: '{query}', "
                f"got {features.legal_reference_count}"
            )

    def test_legal_reference_count_multiple_refs(self, extractor: FeatureExtractor) -> None:
        cases = {
            "Điều 1 Quyết định 25/2026/QĐ-UBND nói về nội dung gì?": 2,
            "Theo khoản 22 Điều 13 Nghị định số 50/2026/NĐ-CP, doanh nghiệp có cần khai thuế không?": 2,
            "Nếu một cơ quan đã nộp hồ sơ trước ngày Thông tư số 02/2026/TT-BYT có hiệu lực, nhưng đồng thời cần kinh phí theo khoản 1, điểm c Điều 6 Nghị định 75/2026/NĐ-CP, thì có được tiếp tục không?": 3,
        }
        for query, min_count in cases.items():
            features = extractor.extract(query)
            assert features.legal_reference_count >= min_count, (
                f"legal_reference_count should be >= {min_count} for: '{query}', "
                f"got {features.legal_reference_count}"
            )

    def test_graph_keyword_count_nonzero(self, extractor: FeatureExtractor) -> None:
        graph_cases = [
            "Giám đốc Sở Y tế có trách nhiệm gì liên quan đến các thủ tục hành chính được phân cấp theo Điều 2?",
            "Các thủ tục hành chính nào đã bị bãi bỏ và thay thế bởi các thủ tục được công bố tại Quyết định này?",
            "Theo Quyết định này, cơ quan nào chịu trách nhiệm thi hành?",
        ]
        for query in graph_cases:
            features = extractor.extract(query)
            assert features.graph_keyword_count >= 1, (
                f"graph_keyword_count should be >= 1 for: '{query}', "
                f"got {features.graph_keyword_count}"
            )

    def test_remaining_false_negative_reasoning_signals(self, extractor: FeatureExtractor) -> None:
        cases = [
            "Cục Hóa chất có tư cách pháp nhân và được thành lập các đơn vị trực thuộc như thế nào?",
            "Nếu một xã có diện tích 8.000 ha và lập bản đồ hiện trạng sử dụng đất ở tỷ lệ 1/10.000, cần bao nhiêu cái quạt trần và bao nhiêu quyển sổ ghi chép theo định mức?",
            "Nếu Bộ Công Thương muốn sử dụng vốn ODA hoặc vốn vay ưu đãi nước ngoài để thực hiện các dự án trong Kế hoạch chuyển đổi số năm 2026, thì có được phép không?",
        ]
        for query in cases:
            features = extractor.extract(query)
            assert (
                features.multi_hop_score > 0
                or features.complexity_level >= 2
                or features.sub_question_count >= 1
                or features.authority_chain_count >= 1
                or features.procedural_count >= 1
                or features.multi_entity_relation_count >= 1
            ), f"Expected at least one reasoning signal for: {query}"

