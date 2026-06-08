from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router.history_resolver import resolve_history_referents


def test_empty_history_contextual_reference_is_no_history() -> None:
    result = resolve_history_referents("Văn bản đó còn hiệu lực không?", None)

    assert result.query_has_contextual_reference
    assert result.resolution_status == "no_history"
    assert result.resolved_referent is None


def test_single_document_history_resolves_pronoun() -> None:
    result = resolve_history_referents(
        "Văn bản đó còn hiệu lực không?",
        "Người dùng đang hỏi về Nghị định 100/2019/NĐ-CP.",
    )

    assert result.resolution_status == "resolved"
    assert result.resolved_referent
    assert "Nghị định 100/2019/NĐ-CP" in result.resolved_referent


def test_irrelevant_history_is_not_resolved() -> None:
    result = resolve_history_referents(
        "Văn bản đó còn hiệu lực không?",
        "Người dùng hỏi cách tra cứu văn bản pháp luật nhưng chưa nêu văn bản cụ thể.",
    )

    assert result.resolution_status == "irrelevant_history"
    assert result.resolved_referent is None


def test_two_documents_are_conflicting_for_singular_reference() -> None:
    result = resolve_history_referents(
        "Văn bản đó còn hiệu lực không?",
        "Người dùng nhắc đến Nghị định 100/2019/NĐ-CP và Luật Đường bộ 2024.",
    )

    assert result.resolution_status == "conflicting_history"
    assert len(result.candidate_referents) >= 2


def test_non_contextual_query_does_not_need_history() -> None:
    result = resolve_history_referents(
        "Điều kiện kết hôn theo Luật Hôn nhân và gia đình gồm những gì?",
        None,
    )

    assert not result.query_has_contextual_reference
    assert result.resolution_status == "not_needed"
