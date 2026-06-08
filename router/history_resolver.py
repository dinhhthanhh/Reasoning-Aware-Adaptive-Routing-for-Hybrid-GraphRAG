"""Deterministic conversation-history referent resolution for legal QA.

The router should not treat any non-empty history as resolving a follow-up
question. This module extracts concrete legal referents from history and labels
whether the current query is resolved, unresolved, irrelevant, or conflicting.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Literal


ReferentType = Literal[
    "legal_document",
    "legal_article",
    "agency",
    "procedure",
    "concept",
    "factual_situation",
    "unknown",
]

ResolutionStatus = Literal[
    "resolved",
    "unresolved",
    "irrelevant_history",
    "conflicting_history",
    "no_history",
    "not_needed",
]


@dataclass
class HistoryReferentCandidate:
    text: str
    type: ReferentType
    source_span: str
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class HistoryResolutionResult:
    has_history: bool = False
    query_has_contextual_reference: bool = False
    candidate_referents: list[HistoryReferentCandidate] = field(default_factory=list)
    resolved_referent: str | None = None
    resolution_status: ResolutionStatus = "not_needed"
    history_resolution_confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["candidate_referents"] = [candidate.to_dict() for candidate in self.candidate_referents]
        return data


CONTEXTUAL_REFERENCE_PATTERNS: list[tuple[str, re.Pattern[str], tuple[ReferentType, ...]]] = [
    (
        "legal_document",
        re.compile(r"(?<!\w)(?:văn\s+bản|nghị\s+định|thông\s+tư|quyết\s+định|luật)\s+(?:đó|này|trên)(?!\w)", re.IGNORECASE),
        ("legal_document",),
    ),
    (
        "legal_article",
        re.compile(r"(?<!\w)(?:điều|điều\s+khoản|quy\s+định|nội\s+dung)\s+(?:đó|này|trên)(?!\w)", re.IGNORECASE),
        ("legal_article", "concept", "legal_document"),
    ),
    (
        "procedure",
        re.compile(r"(?<!\w)(?:thủ\s+tục|hồ\s+sơ)\s+(?:đó|này|trên)(?!\w)", re.IGNORECASE),
        ("procedure",),
    ),
    (
        "agency",
        re.compile(r"(?<!\w)cơ\s+quan\s+(?:đó|này)(?!\w)", re.IGNORECASE),
        ("agency",),
    ),
    (
        "situation",
        re.compile(r"(?<!\w)(?:trường\s+hợp|việc|nguồn\s+kinh\s+phí|vấn\s+đề)\s+(?:đó|này|trên)(?!\w)", re.IGNORECASE),
        ("factual_situation", "concept", "procedure"),
    ),
]

SINGULAR_DEMONSTRATIVE_PATTERN = re.compile(r"(?<!\w)(?:đó|này|trên)(?!\w)", re.IGNORECASE)

DOC_NUMBER_RE = r"\d+\s*/\s*\d{4}\s*/\s*[A-ZĐ]{2,}(?:-[A-ZĐ0-9]+)*"
SHORT_DECISION_RE = r"\d+\s*/\s*QĐ-[A-ZĐ]{2,}(?:-[A-ZĐ0-9]+)*"
VBHN_RE = r"\d+\s*/\s*VBHN-[A-ZĐ]{2,}(?:-[A-ZĐ0-9]+)*"

LEGAL_DOCUMENT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        rf"(?<!\w)(?:Nghị\s+định|Thông\s+tư|Quyết\s+định|Nghị\s+quyết|Chỉ\s+thị|Pháp\s+lệnh|Văn\s+bản\s+hợp\s+nhất)"
        rf"(?:\s+số)?\s+(?:{DOC_NUMBER_RE}|{SHORT_DECISION_RE}|{VBHN_RE})",
        re.IGNORECASE,
    ),
    re.compile(rf"(?<!\w)(?:{DOC_NUMBER_RE}|{SHORT_DECISION_RE}|{VBHN_RE})(?!\w)", re.IGNORECASE),
    re.compile(
        r"(?<!pháp\s)(?<!\w)(?:Luật|Bộ\s+luật|Hiến\s+pháp)\s+"
        r"[A-ZĐÀ-Ỹa-zđà-ỹ0-9\s,\-]+?(?:\s+\d{4})?"
        r"(?=(?:\s+(?:liên\s+quan|về|quy\s+định|được|còn|sửa|và|theo)|[.,;\n]|$))",
        re.IGNORECASE,
    ),
]

LEGAL_ARTICLE_PATTERN = re.compile(
    r"(?<!\w)(?:Điều|Khoản|Điểm)\s+\d+[A-Za-zÀ-ỹ0-9]*"
    r"(?:\s+(?:Luật|Bộ\s+luật|Nghị\s+định|Thông\s+tư|Quyết\s+định)[^.,;\n]{0,80})?",
    re.IGNORECASE,
)

AGENCY_PATTERN = re.compile(
    r"(?<!\w)(?:"
    r"Bộ\s+[A-ZĐÀ-Ỹ][^.,;\n]{1,60}|"
    r"Cục\s+[A-ZĐÀ-Ỹ][^.,;\n]{1,70}|"
    r"Sở\s+[A-ZĐÀ-Ỹ][^.,;\n]{1,70}|"
    r"Ủy\s+ban\s+nhân\s+dân[^.,;\n]{0,70}|"
    r"UBND(?:\s+(?:tỉnh|huyện|xã|thành\s+phố))?(?:\s+[A-ZĐÀ-Ỹ][^.,;\n]{0,40})?"
    r")",
    re.IGNORECASE,
)

PROCEDURE_PATTERN = re.compile(
    r"(?<!\w)(?:thủ\s+tục|hồ\s+sơ|cấp\s+giấy\s+phép|đổi\s+thẻ|giải\s+quyết\s+hồ\s+sơ)"
    r"[^.,;\n]{0,90}",
    re.IGNORECASE,
)

CONCEPT_PHRASES = [
    "miễn giảm tiền thuê đất",
    "miễn, giảm tiền thuê đất",
    "xử phạt vi phạm hành chính",
    "quản lý chất thải rắn sinh hoạt",
    "nghĩa vụ tài chính đất đai",
    "bảo vệ quyền lợi người tiêu dùng",
    "kinh doanh bảo hiểm",
    "lưu trữ hồ sơ",
    "nguồn kinh phí",
    "khám bệnh, chữa bệnh",
    "thủ tục hành chính",
]

FACTUAL_SITUATION_PATTERN = re.compile(
    r"(?<!\w)(?:trường\s+hợp|tình\s+huống|việc)\s+[^.,;\n]{8,110}",
    re.IGNORECASE,
)


def resolve_history_referents(query: str, history: str | None) -> HistoryResolutionResult:
    """Resolve contextual references in ``query`` against ``history``.

    The function is intentionally deterministic and conservative. It only marks
    a query as resolved when exactly one likely referent is available.
    """
    normalized_query = (query or "").strip()
    normalized_history = _history_to_text(history)
    query_has_context, likely_types = _query_contextual_reference_types(normalized_query)
    has_history = bool(normalized_history)

    if not query_has_context:
        return HistoryResolutionResult(
            has_history=has_history,
            query_has_contextual_reference=False,
            candidate_referents=_extract_candidates(normalized_history),
            resolution_status="not_needed",
            reason="query_does_not_require_history",
        )

    if not has_history:
        return HistoryResolutionResult(
            has_history=False,
            query_has_contextual_reference=True,
            resolution_status="no_history",
            history_resolution_confidence=0.0,
            reason="query_has_contextual_reference_but_history_is_empty",
        )

    candidates = _extract_candidates(normalized_history)
    if not candidates:
        return HistoryResolutionResult(
            has_history=True,
            query_has_contextual_reference=True,
            candidate_referents=[],
            resolution_status="irrelevant_history",
            history_resolution_confidence=0.0,
            reason="history_has_no_strong_legal_referent",
        )

    relevant = _select_relevant_candidates(candidates, likely_types)
    if not relevant:
        return HistoryResolutionResult(
            has_history=True,
            query_has_contextual_reference=True,
            candidate_referents=candidates,
            resolution_status="irrelevant_history",
            history_resolution_confidence=max(candidate.confidence for candidate in candidates),
            reason="history_referents_do_not_match_query_reference_type",
        )

    primary_type = likely_types[0] if likely_types else relevant[0].type
    same_primary_type = _dedupe_candidates([candidate for candidate in relevant if candidate.type == primary_type])
    conflict_pool = same_primary_type or _same_type_conflict_pool(relevant)

    if len(conflict_pool) > 1 and SINGULAR_DEMONSTRATIVE_PATTERN.search(normalized_query):
        return HistoryResolutionResult(
            has_history=True,
            query_has_contextual_reference=True,
            candidate_referents=candidates,
            resolution_status="conflicting_history",
            history_resolution_confidence=max(candidate.confidence for candidate in conflict_pool),
            reason="multiple_strong_candidates_for_singular_reference",
        )

    resolved_pool = conflict_pool if conflict_pool else relevant
    resolved = max(resolved_pool, key=lambda candidate: candidate.confidence)
    return HistoryResolutionResult(
        has_history=True,
        query_has_contextual_reference=True,
        candidate_referents=candidates,
        resolved_referent=resolved.text,
        resolution_status="resolved",
        history_resolution_confidence=resolved.confidence,
        reason=f"resolved_to_single_{resolved.type}",
    )


def _history_to_text(history: str | None) -> str:
    if not history:
        return ""
    if isinstance(history, str):
        return history.strip()
    return str(history).strip()


def _query_contextual_reference_types(query: str) -> tuple[bool, tuple[ReferentType, ...]]:
    matches: list[tuple[ReferentType, ...]] = []
    for _name, pattern, types in CONTEXTUAL_REFERENCE_PATTERNS:
        if pattern.search(query):
            matches.append(types)
    if not matches:
        return False, ()

    ordered: list[ReferentType] = []
    for types in matches:
        for item in types:
            if item not in ordered:
                ordered.append(item)
    return True, tuple(ordered)


def _extract_candidates(history: str) -> list[HistoryReferentCandidate]:
    if not history:
        return []

    candidates: list[HistoryReferentCandidate] = []
    for pattern in LEGAL_DOCUMENT_PATTERNS:
        candidates.extend(_matches_to_candidates(pattern, history, "legal_document", 0.92))
    candidates.extend(_matches_to_candidates(LEGAL_ARTICLE_PATTERN, history, "legal_article", 0.9))
    candidates.extend(_matches_to_candidates(AGENCY_PATTERN, history, "agency", 0.78))
    candidates.extend(_matches_to_candidates(PROCEDURE_PATTERN, history, "procedure", 0.74))
    candidates.extend(_extract_concept_candidates(history))
    candidates.extend(_matches_to_candidates(FACTUAL_SITUATION_PATTERN, history, "factual_situation", 0.62))
    return _dedupe_candidates(candidates)


def _matches_to_candidates(
    pattern: re.Pattern[str],
    text: str,
    candidate_type: ReferentType,
    confidence: float,
) -> list[HistoryReferentCandidate]:
    candidates: list[HistoryReferentCandidate] = []
    for match in pattern.finditer(text):
        value = _clean_text(match.group(0))
        if len(value) < 4:
            continue
        lowered = value.lower()
        if any(marker in lowered for marker in ("không nêu", "chưa nêu", "hỏi chung", "nói chung")):
            continue
        candidates.append(
            HistoryReferentCandidate(
                text=value,
                type=candidate_type,
                source_span=f"history[{match.start()}:{match.end()}]",
                confidence=confidence,
            )
        )
    return candidates


def _extract_concept_candidates(history: str) -> list[HistoryReferentCandidate]:
    candidates: list[HistoryReferentCandidate] = []
    lowered = history.lower()
    for phrase in CONCEPT_PHRASES:
        start = lowered.find(phrase.lower())
        if start == -1:
            continue
        end = start + len(phrase)
        candidates.append(
            HistoryReferentCandidate(
                text=history[start:end],
                type="concept",
                source_span=f"history[{start}:{end}]",
                confidence=0.7,
            )
        )
    return candidates


def _select_relevant_candidates(
    candidates: list[HistoryReferentCandidate],
    likely_types: tuple[ReferentType, ...],
) -> list[HistoryReferentCandidate]:
    if not likely_types:
        return candidates
    relevant = [candidate for candidate in candidates if candidate.type in likely_types]
    if relevant:
        return relevant

    # Article-like follow-up queries often refer to a document-level anchor when
    # history names a law/decree but not a specific article.
    if "legal_article" in likely_types:
        fallback = [candidate for candidate in candidates if candidate.type in {"legal_document", "concept", "procedure"}]
        if fallback:
            return fallback
    if "procedure" in likely_types:
        fallback = [candidate for candidate in candidates if candidate.type in {"procedure", "concept", "legal_document", "factual_situation"}]
        if fallback:
            return fallback
    if "factual_situation" in likely_types:
        fallback = [candidate for candidate in candidates if candidate.type in {"concept", "procedure", "legal_document"}]
        if fallback:
            return fallback
    return []


def _same_type_conflict_pool(candidates: list[HistoryReferentCandidate]) -> list[HistoryReferentCandidate]:
    by_type: dict[str, list[HistoryReferentCandidate]] = {}
    for candidate in _dedupe_candidates(candidates):
        by_type.setdefault(candidate.type, []).append(candidate)
    conflict_groups = [group for group in by_type.values() if len(group) > 1]
    if not conflict_groups:
        return []
    return max(conflict_groups, key=lambda group: (len(group), max(item.confidence for item in group)))


def _dedupe_candidates(candidates: list[HistoryReferentCandidate]) -> list[HistoryReferentCandidate]:
    ordered: list[HistoryReferentCandidate] = []
    seen: set[tuple[str, str]] = set()
    for candidate in sorted(candidates, key=lambda item: (-item.confidence, item.source_span)):
        normalized = _normalize(candidate.text)
        if any(
            candidate.type == existing.type
            and _contains_same_referent(normalized, _normalize(existing.text))
            for existing in ordered
        ):
            continue
        key = (normalized, candidate.type)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(candidate)
    return ordered


def _contains_same_referent(left: str, right: str) -> bool:
    if left == right:
        return True
    if len(left) < 8 or len(right) < 8:
        return False
    return left in right or right in left


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" .,:;()[]")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip(" .,:;()[]")
