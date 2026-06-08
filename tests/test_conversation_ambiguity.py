"""Regression checks for conversation-aware ambiguity behavior.

These tests avoid the LLM verifier and focus on the lightweight ambiguity
detector. End-to-end router limitations are reported by the Phase 2 benchmark.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router.ambiguity_detector import AmbiguityDetector


def _detector() -> AmbiguityDetector:
    return AmbiguityDetector(
        {
            "score_threshold": 0.6,
            "pronoun_list": [
                "quy định này",
                "quy định đó",
                "trường hợp này",
                "trường hợp đó",
                "văn bản đó",
            ],
            "vague_terms": ["quy định", "văn bản", "điều khoản"],
        }
    )


def test_empty_history_pronoun_query_is_ambiguous() -> None:
    detector = _detector()
    report = detector.detect("Quy định này còn hiệu lực không?", history=None)

    assert report.score >= 0.6
    assert report.is_ambiguous
    assert "pronoun" in report.ambiguity_types


def test_resolving_history_reduces_unresolved_pronoun_ambiguity() -> None:
    detector = _detector()
    empty_history = detector.detect("Quy định này còn hiệu lực không?", history=None)
    resolving_history = detector.detect(
        "Quy định này còn hiệu lực không?",
        history="Người dùng đang hỏi về Nghị định 100/2019/NĐ-CP.",
    )

    assert resolving_history.score < empty_history.score


def test_irrelevant_history_should_not_fully_suppress_ambiguity() -> None:
    detector = _detector()
    report = detector.detect(
        "Quy định này còn hiệu lực không?",
        history="Người dùng hỏi cách tra cứu văn bản pháp luật nhưng không nêu số hiệu cụ thể.",
    )

    assert report.is_ambiguous


def test_conflicting_history_should_remain_ambiguous() -> None:
    detector = _detector()
    report = detector.detect(
        "Văn bản đó còn hiệu lực không?",
        history="Người dùng nhắc đến Nghị định 100/2019/NĐ-CP và Luật Đường bộ 2024.",
    )

    assert report.is_ambiguous
