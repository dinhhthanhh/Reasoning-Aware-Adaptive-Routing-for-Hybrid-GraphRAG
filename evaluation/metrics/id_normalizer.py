"""Canonical legal-article ID normalisation for retrieval evaluation.

Motivation
----------
Retrieval Hit@k was reported at ~2.5% because the retriever and the gold
QA dataset use *different* identifier systems:

* Gold ``article_key`` / ``relevant_articles`` use VBPL-style document numbers,
  e.g. ``"77/2026/NĐ-CP::Điều 18"`` -> ``{law_id: "77/2026/NĐ-CP", article_id: "Điều 18"}``.
* Retrieved articles often carry Pháp Điển structural codes, e.g.
  ``"19.2. Điều 19.2.TT.10.15. Đơn vị quản lý kinh phí ..."`` or opaque store IDs
  such as ``"Document 162387"`` / ``"12::VBHN-BXD"``.

This module defines ONE canonical form and a normaliser that maps both sides to
it, so retrieval metrics compare like-with-like. It does **not** touch Neo4j
storage — normalisation happens only at evaluation time.

Canonical form
--------------
``law::<document_code>::article::<number>``  e.g. ``law::77/2026/NĐ-CP::article::18``

A :class:`CanonicalID` keeps the document code and article number separately so
callers can choose strict (doc+article) or relaxed (article-only) matching and
can honestly report which retrieved IDs were *unresolvable* to the gold scheme.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

__all__ = [
    "CanonicalID",
    "normalize_legal_id",
    "normalize_gold_article",
    "canonical_key",
    "compute_hit_at_k",
    "compute_mrr",
]


# Document-number patterns (ordered most-specific first).
# NOTE: The first capture group allows trailing digits (e.g. QH11, QH12).
_DOC_FULL_RE = re.compile(
    r"\d+\s*/\s*\d{4}\s*/\s*[A-ZĐ]{2,}[A-ZĐ0-9]*(?:-[A-ZĐ0-9]+)*", re.IGNORECASE
)
_DOC_DECISION_RE = re.compile(r"\d+\s*/\s*QĐ-[A-ZĐ]{2,}(?:-[A-ZĐ0-9]+)*", re.IGNORECASE)
_DOC_VBHN_RE = re.compile(r"\d+\s*/\s*VBHN-[A-ZĐ]{2,}(?:-[A-ZĐ0-9]+)*", re.IGNORECASE)

# A "real" article number is a complete integer (optionally with a trailing
# letter, e.g. "Điều 4a"). It must NOT be followed by another digit (which would
# mean we matched only part of a longer number) nor by ".<digit>" (which signals
# a Pháp Điển structural code such as "Điều 19.2.TT.10.15").
_ARTICLE_RE = re.compile(
    r"Điều\s+(\d+[a-zđ]?)(?!\d)(?!\s*\.\s*\d)",
    re.IGNORECASE,
)

# Pháp Điển structural code, kept so we can flag (not match) these explicitly.
_PHAPDIEN_CODE_RE = re.compile(r"\d+\.\d+\.[A-ZĐ]{2}\.\d+\.\d+", re.IGNORECASE)


@dataclass(frozen=True)
class CanonicalID:
    """Canonical representation of a legal-article reference.

    Attributes:
        doc_code: Normalised document number (e.g. ``"77/2026/NĐ-CP"``) or ``""``.
        article_num: Article number as a string (e.g. ``"18"``, ``"4a"``) or ``""``.
        phapdien_code: Pháp Điển structural code if present (diagnostic only).
        raw: The original string this was parsed from.
    """

    doc_code: str = ""
    article_num: str = ""
    phapdien_code: str = ""
    raw: str = ""

    @property
    def is_resolvable(self) -> bool:
        """True if at least a document code or a real article number was found."""
        return bool(self.doc_code or self.article_num)

    @property
    def key(self) -> str:
        """Canonical string key ``law::<doc>::article::<num>`` (parts omitted if absent)."""
        return canonical_key(self.doc_code, self.article_num)


def _clean_doc_code(code: str) -> str:
    # Strip whitespace, uppercase, fix common Unicode issues
    code = re.sub(r"\s+", "", code).upper()
    code = code.replace("ÐY", "ĐY")
    # Fix Cyrillic С (U+0421) → Latin C, с (U+0441) → c
    code = code.replace('\u0421', 'C').replace('\u0441', 'C')
    return code


def canonical_key(doc_code: str, article_num: str) -> str:
    """Build the canonical key string from parts."""
    parts: list[str] = []
    if doc_code:
        parts.append(f"law::{doc_code}")
    if article_num:
        parts.append(f"article::{article_num}")
    return "::".join(parts) if parts else ""


def _extract_doc_code(text: str) -> str:
    # Pre-normalize Cyrillic characters before regex matching
    text = text.replace('\u0421', 'C').replace('\u0441', 'c')
    for pattern in (_DOC_FULL_RE, _DOC_DECISION_RE, _DOC_VBHN_RE):
        match = pattern.search(text)
        if match:
            return _clean_doc_code(match.group(0))
    return ""


def normalize_legal_id(raw: str) -> CanonicalID:
    """Normalise an arbitrary retrieved/stored article reference.

    Handles VBPL doc numbers, short decisions, VBHN, ``law::article`` strings,
    and Pháp Điển structural codes. Returns a :class:`CanonicalID`; check
    ``.is_resolvable`` to detect references that cannot be mapped to the gold
    identifier scheme.
    """
    if not raw:
        return CanonicalID(raw="")
    text = unicodedata.normalize("NFC", str(raw))

    doc_code = _extract_doc_code(text)
    phapdien = _PHAPDIEN_CODE_RE.search(text)
    phapdien_code = phapdien.group(0) if phapdien else ""

    article_match = _ARTICLE_RE.search(text)
    article_num = article_match.group(1).lower() if article_match else ""

    # "12/VBHN-BXD::8" or "12::VBHN-BXD" style: split on "::"
    if "::" in text:
        for part in text.split("::"):
            if not doc_code:
                doc_code = _extract_doc_code(part)
            if not article_num and re.match(r"^\d+[a-zđ]?$", part, re.IGNORECASE):
                article_num = part.lower()

    return CanonicalID(
        doc_code=doc_code,
        article_num=article_num,
        phapdien_code=phapdien_code,
        raw=str(raw),
    )


def normalize_gold_article(article: object) -> CanonicalID:
    """Normalise a gold article entry.

    Accepts either a dict ``{"law_id": ..., "article_id": ...}`` (from
    ``relevant_articles``) or a string ``"<law>::<article>"`` (from ``article_key``).
    """
    if isinstance(article, dict):
        law = str(article.get("law_id", "") or "")
        art = str(article.get("article_id", "") or "")
        doc_code = _extract_doc_code(law) or _clean_doc_code(law) if law else ""
        art_match = _ARTICLE_RE.search(art)
        article_num = art_match.group(1).lower() if art_match else ""
        return CanonicalID(doc_code=doc_code, article_num=article_num, raw=str(article))
    return normalize_legal_id(str(article))


def _match(gold: CanonicalID, pred: CanonicalID, mode: str) -> bool:
    """Return whether a predicted ID matches a gold ID under ``mode``.

    Modes:
        * ``"strict"``  -> document code AND article number must match.
        * ``"article"`` -> article number must match (document ignored). Use when
          retrieval cannot recover the document code (Pháp Điển vs VBPL).
        * ``"doc"``     -> document code must match.
    """
    if mode == "strict":
        return (
            bool(gold.doc_code)
            and gold.doc_code == pred.doc_code
            and bool(gold.article_num)
            and gold.article_num == pred.article_num
        )
    if mode == "article":
        return bool(gold.article_num) and gold.article_num == pred.article_num
    if mode == "doc":
        return bool(gold.doc_code) and gold.doc_code == pred.doc_code
    raise ValueError(f"unknown match mode: {mode}")


def compute_hit_at_k(
    retrieved: list[object],
    gold: list[object],
    k: int,
    mode: str = "strict",
) -> int:
    """Return 1 if any gold article appears in the top-k retrieved list.

    Args:
        retrieved: Ordered list of retrieved article references (any format).
        gold: List of gold article references (dict or string).
        k: Cut-off rank.
        mode: Matching mode (``"strict"`` | ``"article"`` | ``"doc"``).
    """
    if not gold or k <= 0:
        return 0
    gold_ids = [normalize_gold_article(g) for g in gold]
    top = [normalize_legal_id(str(r)) for r in retrieved[:k]]
    for pred in top:
        if any(_match(g, pred, mode) for g in gold_ids):
            return 1
    return 0


def compute_mrr(
    retrieved: list[object],
    gold: list[object],
    mode: str = "strict",
) -> float:
    """Reciprocal rank of the first correct retrieved article (0 if none)."""
    if not gold:
        return 0.0
    gold_ids = [normalize_gold_article(g) for g in gold]
    for rank, item in enumerate(retrieved, start=1):
        pred = normalize_legal_id(str(item))
        if any(_match(g, pred, mode) for g in gold_ids):
            return 1.0 / rank
    return 0.0
