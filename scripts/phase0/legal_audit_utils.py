"""Shared helpers for Phase 0 corpus / retrieval audits."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Document-number patterns (ordered most-specific first).
_DOC_PATTERNS = [
    re.compile(r"\d+\s*/\s*\d{4}\s*/\s*[A-ZĐ]{2,}(?:-[A-ZĐ0-9]+)+", re.IGNORECASE),
    re.compile(r"\d+\s*/\s*\d{4}\s*/\s*QH\d+", re.IGNORECASE),
    re.compile(r"\d+\s*/\s*QĐ-[A-ZĐ]{2,}(?:-[A-ZĐ0-9]+)*", re.IGNORECASE),
    re.compile(r"\d+\s*/\s*VBHN-[A-ZĐ]{2,}(?:-[A-ZĐ0-9]+)*", re.IGNORECASE),
    re.compile(r"\d+\s*/\s*HD-[A-ZĐ0-9]+", re.IGNORECASE),
    re.compile(r"\d+\s*/\s*CT-[A-ZĐ0-9]+", re.IGNORECASE),
    re.compile(r"\d+\s*/\s*CĐ-[A-ZĐ0-9]+", re.IGNORECASE),
]

_ARTICLE_FROM_GHI_CHU_RE = re.compile(
    r"^Điều\s+(\d+[a-zđ]?)",
    re.IGNORECASE,
)
_ARTICLE_GENERIC_RE = re.compile(
    r"Điều\s+(\d+[a-zđ]?)(?!\d)(?!\s*\.\s*\d)",
    re.IGNORECASE,
)

_DIEU_COUNT_RE = re.compile(
    r"Điều\s+\d+[a-zđ]?(?!\d)(?!\s*\.\s*\d)",
    re.IGNORECASE,
)


def clean_doc_code(code: str) -> str:
    return re.sub(r"\s+", "", code or "").upper().replace("ÐY", "ĐY")


def extract_doc_codes(text: str) -> list[str]:
    if not text:
        return []
    text = unicodedata.normalize("NFC", text)
    found: list[str] = []
    seen: set[str] = set()
    for pattern in _DOC_PATTERNS:
        for match in pattern.finditer(text):
            code = clean_doc_code(match.group(0))
            if code and code not in seen:
                seen.add(code)
                found.append(code)
    return found


def normalize_article_id(article_id: str) -> str:
    if not article_id:
        return ""
    text = unicodedata.normalize("NFC", str(article_id)).replace("\n", " ").strip()
    if text.lower() in {"toàn văn", "toan van", "full"}:
        return "TOAN_VAN"
    match = _ARTICLE_GENERIC_RE.search(text)
    if match:
        return match.group(1).lower()
    digits = re.sub(r"[^\d]", "", text)
    return digits.lower() if digits else text.lower()


def classify_law_type(law_id: str) -> str:
    lid = clean_doc_code(law_id)
    if "VBHN" in lid:
        return "VBHN"
    if "/NĐ" in lid or "NĐ-CP" in lid:
        return "Nghị định"
    if "/TT" in lid or "TT-" in lid:
        return "Thông tư"
    if "/QĐ" in lid or "QĐ-" in lid:
        return "Quyết định"
    if "/QH" in lid:
        return "Luật"
    if "/HD" in lid:
        return "other"
    if "/CT" in lid or "/CĐ" in lid:
        return "other"
    return "other"


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_test_questions(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return list(iter_jsonl(path))
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("data", [])


def gold_articles(record: dict[str, Any]) -> list[dict[str, str]]:
    articles: list[dict[str, str]] = []
    for key in ("relevant_articles", "supporting_facts"):
        for item in record.get(key) or []:
            law_id = str(item.get("law_id") or "").strip()
            article_id = str(item.get("article_id") or "").strip()
            if law_id:
                articles.append({"law_id": law_id, "article_id": article_id})
    if not articles and record.get("article_key"):
        for part in str(record["article_key"]).split(","):
            part = part.strip()
            if "::" in part:
                law_id, article_id = part.split("::", 1)
                articles.append({"law_id": law_id.strip(), "article_id": article_id.strip()})
    # dedupe
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, str]] = []
    for a in articles:
        key = (a["law_id"], a["article_id"])
        if key not in seen:
            seen.add(key)
            unique.append(a)
    return unique


def pd_record_law_and_article(record: dict[str, Any]) -> tuple[str, str]:
    meta = record.get("raw_metadata") or {}
    ghi_chu = str(meta.get("ghi_chu") or "")
    blob = " ".join(
        filter(
            None,
            [
                record.get("title", ""),
                record.get("source", ""),
                ghi_chu,
            ],
        )
    )
    codes = extract_doc_codes(blob)
    law = codes[0] if codes else ""
    article = ""
    match = _ARTICLE_FROM_GHI_CHU_RE.search(ghi_chu.strip())
    if match:
        article = match.group(1).lower()
    elif ghi_chu:
        match = _ARTICLE_GENERIC_RE.search(ghi_chu)
        if match:
            article = match.group(1).lower()
    return law, article


def build_phapdien_index(path: Path) -> tuple[dict[tuple[str, str], dict], set[str], dict[str, int]]:
    """Return (article_index, law_set, stats)."""
    article_index: dict[tuple[str, str], dict] = {}
    law_set: set[str] = set()
    stats = {
        "total_records": 0,
        "with_law_number": 0,
        "with_article_number": 0,
        "with_full_text": 0,
        "missing_law_number": 0,
        "duplicate_keys": 0,
    }
    for record in iter_jsonl(path):
        stats["total_records"] += 1
        content = str(record.get("content_markdown") or "")
        if len(content.strip()) >= 20:
            stats["with_full_text"] += 1
        law, article = pd_record_law_and_article(record)
        if law:
            stats["with_law_number"] += 1
            law_set.add(law)
        else:
            stats["missing_law_number"] += 1
        if article:
            stats["with_article_number"] += 1
        if law and article:
            key = (law, article)
            if key in article_index:
                stats["duplicate_keys"] += 1
            else:
                article_index[key] = record
    return article_index, law_set, stats


def scan_hf_for_laws(path: Path, target_laws: set[str]) -> set[str]:
    """Return subset of target_laws found in HF corpus content."""
    if not target_laws or not path.exists():
        return set()
    # Normalize targets for substring search
    needles = {law.replace(" ", "") for law in target_laws}
    found: set[str] = set()
    remaining = set(target_laws)
    for record in iter_jsonl(path):
        if not remaining:
            break
        blob = " ".join(
            filter(
                None,
                [
                    record.get("title", ""),
                    str(record.get("raw_metadata", "")),
                    str(record.get("content_markdown", ""))[:6000],
                ],
            )
        ).replace(" ", "")
        hit = {law for law in list(remaining) if law.replace(" ", "") in blob}
        if hit:
            found |= hit
            remaining -= hit
    return found


def count_dieu_in_text(text: str) -> int:
    return len(_DIEU_COUNT_RE.findall(text or ""))
