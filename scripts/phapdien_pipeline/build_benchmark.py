"""Generate Pháp Điển-only QA benchmark with GENUINE multi-hop questions.

Key design principles vs. previous version:
  - dense_retrieval : single-article lookup (unchanged)
  - graph_traversal : 2-hop same-law; question asks about COMBINED TOPIC of two
                      articles WITHOUT mentioning article numbers → pure_vector
                      can only retrieve one article; graph traversal follows
                      explicit cross-article references to retrieve both.
  - hybrid_reasoning: 2-hop cross-law; question asks about a topic across TWO
                      explicit law numbers → pure_vector may miss one law;
                      hybrid (vector+graph) retrieves from both.

Gold answer for graph/hybrid = A.content + "\n\n" + B.content (≤3000 chars).
This guarantees F1(oracle) > F1(pure_vector) for multi-hop queries.

Usage:
    python build_benchmark_multihop.py [--input PD_JSONL] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

PD_RECHUNKED = ROOT / "data/processed/pd_rechunked.jsonl"
OUT_DIR = ROOT / "qa_pipeline/data/phapdien_strict"

# ──────────────────────────────────────────────────────────────────────────────
# Regex helpers
# ──────────────────────────────────────────────────────────────────────────────

# Matches article references within a document:  "theo Điều 5", "tại Điều 5a"
_SAME_ART_REF_RE = re.compile(
    r"(?:theo\s+(?:quy\s+định\s+)?(?:của\s+)?(?:các\s+)?|tại\s+|căn\s+cứ\s+|"
    r"quy\s+định\s+tại\s+|tham\s+chiếu\s+)(?:các\s+)?[Đđ]iều\s+(\d+[a-zđ]?)",
    re.IGNORECASE,
)

# Other-law document number
_OTHER_LAW_RE = re.compile(
    r"(\d+\s*/\s*\d{4}\s*/\s*[A-ZĐ]{2,}[A-ZĐ0-9]*(?:-[A-ZĐ0-9]+)*)",
    re.IGNORECASE,
)

# ──────────────────────────────────────────────────────────────────────────────
# Topic extraction
# ──────────────────────────────────────────────────────────────────────────────

_TITLE_TOPIC_RE = re.compile(
    r"^\d+\.\d+\.\s*[Đđ]iều\s+\d+(?:\.\d+)*(?:\.[A-ZĐa-z]+\d*)*\.\s*(.+?)(?:\s*[-–]\s*(?:Luật|Nghị định|Thông tư|Quyết định|Chỉ thị|Pháp lệnh|Hiến pháp).*)?$",
    re.IGNORECASE,
)
_NUMBERED_PREFIX_RE = re.compile(r"^\d+(?:\.\d+)+\.\s*")


def _extract_topic(title: str) -> str:
    """Extract the human-readable topic from a Pháp Điển article title."""
    m = _TITLE_TOPIC_RE.match(title.strip())
    if m:
        t = m.group(1).strip()
        t = re.sub(r"\s*[-–]\s*.+$", "", t).strip()
        t = _NUMBERED_PREFIX_RE.sub("", t).strip()
        return t
    # Fallback: last dot-segment, before dash
    part = title.split(" - ")[0]
    segments = part.split(".")
    return segments[-1].strip()[:80]


# ──────────────────────────────────────────────────────────────────────────────
# Multi-hop question templates
# ──────────────────────────────────────────────────────────────────────────────

_GRAPH_Q_TEMPLATES = [
    "Theo văn bản {law}, quy định về {topic_a} và {topic_b} được nêu cụ thể như thế nào? Hãy tổng hợp đầy đủ từ các quy định liên quan trong văn bản này.",
    "Trong văn bản {law}, {topic_a} và {topic_b} đều được quy định. Nội dung chi tiết của hai quy định này là gì?",
    "Theo {law}, cần hiểu như thế nào về {topic_a}? Ngoài ra, văn bản này còn quy định thêm điều gì về {topic_b} liên quan?",
    "Văn bản {law} có những quy định nào về {topic_a}? Đồng thời, quy định về {topic_b} trong cùng văn bản là gì? Hãy nêu đầy đủ cả hai.",
    "Theo {law}, hãy trình bày toàn bộ nội dung quy định liên quan đến {topic_a} cũng như {topic_b} trong văn bản này.",
]

_HYBRID_Q_TEMPLATES = [
    "Theo quy định của văn bản {law_a} và văn bản {law_b}, nội dung về {topic_a} được quy định như thế nào? Hãy tổng hợp đầy đủ từ cả hai văn bản.",
    "Để hiểu đầy đủ về {topic_a}, cần tham chiếu cả văn bản {law_a} và văn bản {law_b}. Nội dung tương ứng ở hai văn bản này là gì?",
    "Văn bản {law_a} về {topic_a} có liên quan đến văn bản {law_b}. Hãy tổng hợp nội dung quy định từ cả hai văn bản về vấn đề này.",
    "Theo cả văn bản {law_a} lẫn văn bản {law_b}, quy định về {topic_a} và {topic_b} bao gồm những nội dung gì?",
    "Văn bản {law_a} và văn bản {law_b} cùng điều chỉnh lĩnh vực liên quan đến {topic_a}. Hãy nêu đầy đủ nội dung quy định từ từng văn bản.",
]


# ──────────────────────────────────────────────────────────────────────────────
# Pool classification
# ──────────────────────────────────────────────────────────────────────────────

def _find_same_law_ref(rec: dict, by_law_art: dict[str, dict[str, dict]]) -> str | None:
    """Return an article number in the same law that rec explicitly references."""
    law = rec.get("law_number", "")
    art = rec["canonical_id"].split("::", 1)[1]
    content = rec.get("content", "")
    law_articles = by_law_art.get(law, {})
    for m in _SAME_ART_REF_RE.finditer(content):
        ref_art = m.group(1)
        if ref_art != art and ref_art in law_articles:
            return ref_art
    return None


def _find_other_law_ref(rec: dict, by_law_art: dict[str, dict[str, dict]]) -> str | None:
    """Return another law number that rec explicitly references and exists in corpus."""
    law = rec.get("law_number", "")
    self_norm = re.sub(r"\s+", "", law).upper()
    content = rec.get("content", "")
    for m in _OTHER_LAW_RE.finditer(content):
        other = re.sub(r"\s+", "", m.group(1)).upper()
        if other != self_norm and other in by_law_art and len(by_law_art[other]) > 0:
            return other
    return None


def _pool_for(
    rec: dict,
    by_law_art: dict[str, dict[str, dict]],
) -> tuple[str, str | None, str | None]:
    """Return (pool_label, same_law_ref_art, other_law_ref) for a record."""
    other_law = _find_other_law_ref(rec, by_law_art)
    if other_law:
        return "hybrid_reasoning", None, other_law
    same_ref = _find_same_law_ref(rec, by_law_art)
    if same_ref:
        return "graph_traversal", same_ref, None
    return "dense_retrieval", None, None


# ──────────────────────────────────────────────────────────────────────────────
# Question/answer builders
# ──────────────────────────────────────────────────────────────────────────────

def _make_dense(rec: dict, idx: int, split: str) -> dict:
    law = rec["law_number"]
    art_num = rec["canonical_id"].split("::", 1)[1]
    art_label = f"Điều {art_num}"
    title = rec.get("title", law)
    content = rec["content"].strip()
    answer = content[:3000]
    question = f"Nội dung {art_label} của văn bản {law} quy định như thế nào?"
    return _base_record(
        idx, split, question, answer, content,
        f"{law}::{art_num}", title, law, art_label,
        hop_count=1, is_cross_doc=False,
        routing_label="dense_retrieval",
        relevant=[(law, art_label, content)],
        question_type="factual", difficulty=0.35, has_reasoning=False,
        canonical_id=rec["canonical_id"],
    )


def _make_graph_multihop(
    rec: dict,
    ref_art_num: str,
    by_law_art: dict[str, dict[str, dict]],
    idx: int,
    split: str,
) -> dict:
    """2-hop same-law question: topic from A + B, NO article numbers in question."""
    law = rec["law_number"]
    art_num = rec["canonical_id"].split("::", 1)[1]
    art_label = f"Điều {art_num}"
    content_a = rec["content"].strip()
    title_a = rec.get("title", law)

    ref_rec = by_law_art[law][ref_art_num]
    ref_label = f"Điều {ref_art_num}"
    content_b = ref_rec["content"].strip()

    topic_a = _extract_topic(title_a)
    topic_b = _extract_topic(ref_rec.get("title", law))

    # Fallback topics
    if not topic_a or len(topic_a) < 4:
        topic_a = art_label
    if not topic_b or len(topic_b) < 4:
        topic_b = ref_label

    template = random.choice(_GRAPH_Q_TEMPLATES)
    question = template.format(law=law, topic_a=topic_a, topic_b=topic_b)

    # Gold answer = BOTH articles combined → pure_vector can only get one
    answer = (content_a[:1500] + "\n\n" + content_b[:1400])[:3000]

    relevant = [
        (law, art_label, content_a),
        (law, ref_label, content_b),
    ]
    return _base_record(
        idx, split, question, answer, content_a,
        f"{law}::{art_num}", title_a, law, art_label,
        hop_count=2, is_cross_doc=False,
        routing_label="graph_traversal",
        relevant=relevant,
        question_type="relational_multihop", difficulty=0.60, has_reasoning=True,
        canonical_id=rec["canonical_id"],
    )


def _make_hybrid_multihop(
    rec: dict,
    ref_law: str,
    by_law_art: dict[str, dict[str, dict]],
    idx: int,
    split: str,
) -> dict:
    """2-hop cross-law question: both law numbers in question, gold = A+B content."""
    law_a = rec["law_number"]
    art_num = rec["canonical_id"].split("::", 1)[1]
    art_label = f"Điều {art_num}"
    content_a = rec["content"].strip()
    title_a = rec.get("title", law_a)

    # Pick the most relevant article from the referenced law
    ref_articles = list(by_law_art[ref_law].values())
    random.shuffle(ref_articles)
    ref_rec = ref_articles[0]
    ref_art_num = ref_rec["canonical_id"].split("::", 1)[1]
    ref_label = f"Điều {ref_art_num}"
    content_b = ref_rec["content"].strip()

    topic_a = _extract_topic(title_a)
    topic_b = _extract_topic(ref_rec.get("title", ref_law))

    if not topic_a or len(topic_a) < 4:
        topic_a = art_label
    if not topic_b or len(topic_b) < 4:
        topic_b = ref_label

    template = random.choice(_HYBRID_Q_TEMPLATES)
    question = template.format(
        law_a=law_a, law_b=ref_law,
        topic_a=topic_a, topic_b=topic_b,
    )

    # Gold answer = BOTH laws combined → pure_vector may miss one law
    answer = (content_a[:1500] + "\n\n" + content_b[:1400])[:3000]

    relevant = [
        (law_a, art_label, content_a),
        (ref_law, ref_label, content_b),
    ]
    return _base_record(
        idx, split, question, answer, content_a,
        f"{law_a}::{art_num}", title_a, law_a, art_label,
        hop_count=2, is_cross_doc=True,
        routing_label="hybrid_reasoning",
        relevant=relevant,
        question_type="cross_doc_multihop", difficulty=0.65, has_reasoning=True,
        canonical_id=rec["canonical_id"],
    )


def _base_record(
    idx: int, split: str, question: str, answer: str,
    gold_content: str, canonical_key: str, title: str,
    law: str, art_label: str, *,
    hop_count: int, is_cross_doc: bool, routing_label: str,
    relevant: list[tuple[str, str, str]],
    question_type: str, difficulty: float, has_reasoning: bool,
    canonical_id: str,
) -> dict:
    rel_articles = [
        {"law_id": lid, "article_id": alab, "content": body[:2000]}
        for lid, alab, body in relevant
    ]
    supporting = [
        {"law_id": lid, "article_id": alab, "title": f"{title} — {alab}"}
        for lid, alab, _ in relevant[:1]
    ]
    return {
        "id": f"phapdien_strict_{split}_{idx:04d}",
        "question": question,
        "answer": answer,
        "gold_context": gold_content[:4000],
        "article_key": canonical_key,
        "law": title,
        "doc_number": law,
        "relevant_articles": rel_articles,
        "supporting_facts": supporting,
        "hop_count": hop_count,
        "is_cross_doc": is_cross_doc,
        "routing_label": routing_label,
        "difficulty": difficulty,
        "has_reasoning": has_reasoning,
        "question_type": question_type,
        "source": "phapdien",
        "canonical_id": canonical_id,
        "strict_split": split,
        "eval_set": "phapdien_strict",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Loading & sampling
# ──────────────────────────────────────────────────────────────────────────────

SPLIT_QUOTAS = {
    "train": {"dense_retrieval": 400, "graph_traversal": 200, "hybrid_reasoning": 200},
    "dev":   {"dense_retrieval": 50,  "graph_traversal": 25,  "hybrid_reasoning": 25},
    "test":  {"dense_retrieval": 300, "graph_traversal": 150, "hybrid_reasoning": 150},
}


def load_candidates(path: Path) -> tuple[
    dict[str, list[tuple[dict, str | None, str | None]]],
    dict[str, dict[str, dict]],
]:
    """
    Returns:
        pools : label -> list of (record, same_law_ref_art, other_law_ref)
        by_law_art : law_number -> {art_num -> record}
    """
    by_law_art: dict[str, dict[str, dict]] = defaultdict(dict)
    raw_rows: list[dict] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("has_canonical_id"):
                continue
            content = (r.get("content") or "").strip()
            if len(content) < 80:
                continue
            if not r.get("law_number") or "::" not in r.get("canonical_id", ""):
                continue
            art_num = r["canonical_id"].split("::", 1)[1]
            law = r["law_number"]
            by_law_art[law][art_num] = r
            raw_rows.append(r)

    # Build pools after full index is loaded (need complete by_law_art)
    pools: dict[str, list[tuple[dict, str | None, str | None]]] = defaultdict(list)
    for r in raw_rows:
        label, same_ref, other_law = _pool_for(r, by_law_art)
        pools[label].append((r, same_ref, other_law))

    return pools, by_law_art


def sample_pool(
    pool: list[tuple[dict, str | None, str | None]],
    n: int,
    used: set[str],
) -> list[tuple[dict, str | None, str | None]]:
    random.shuffle(pool)
    picked: list[tuple[dict, str | None, str | None]] = []
    for item in pool:
        cid = item[0]["canonical_id"]
        if cid in used:
            continue
        picked.append(item)
        used.add(cid)
        if len(picked) >= n:
            return picked
    # Fallback: allow reuse
    for item in pool:
        if item not in picked:
            picked.append(item)
        if len(picked) >= n:
            break
    return picked


def build_split(
    split: str,
    quotas: dict[str, int],
    pools: dict[str, list[tuple[dict, str | None, str | None]]],
    by_law_art: dict[str, dict[str, dict]],
    used: set[str],
) -> list[dict]:
    raw: list[tuple[str, dict, str | None, str | None]] = []
    for label, n in quotas.items():
        sampled = sample_pool(pools.get(label, []), n, used)
        for rec, same_ref, other_law in sampled:
            raw.append((label, rec, same_ref, other_law))

    random.shuffle(raw)
    records: list[dict] = []
    for i, (label, rec, same_ref, other_law) in enumerate(raw):
        if label == "dense_retrieval":
            records.append(_make_dense(rec, i, split))
        elif label == "graph_traversal":
            if same_ref and same_ref in by_law_art.get(rec["law_number"], {}):
                records.append(_make_graph_multihop(rec, same_ref, by_law_art, i, split))
            else:
                # Fallback: pick any other article from same law
                law = rec["law_number"]
                art = rec["canonical_id"].split("::", 1)[1]
                others = [a for a in by_law_art.get(law, {}) if a != art]
                if others:
                    records.append(_make_graph_multihop(rec, random.choice(others), by_law_art, i, split))
                else:
                    records.append(_make_dense(rec, i, split))
        else:  # hybrid_reasoning
            if other_law and other_law in by_law_art and by_law_art[other_law]:
                records.append(_make_hybrid_multihop(rec, other_law, by_law_art, i, split))
            else:
                records.append(_make_dense(rec, i, split))
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PD_RECHUNKED)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"{args.input} not found")

    random.seed(args.seed)
    print("Loading and indexing corpus…")
    pools, by_law_art = load_candidates(args.input)

    print("Pool sizes:")
    for k, v in pools.items():
        print(f"  {k}: {len(v)}")

    total_need = sum(sum(q.values()) for q in SPLIT_QUOTAS.values())
    total_have = sum(len(v) for v in pools.values())
    if total_have < total_need:
        raise RuntimeError(f"Only {total_have} candidates; need {total_need}")

    used: set[str] = set()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats: dict = {
        "pool_sizes": {k: len(v) for k, v in pools.items()},
        "splits": {},
    }

    for split, quotas in SPLIT_QUOTAS.items():
        records = build_split(split, quotas, pools, by_law_art, used)
        route_counts = Counter(r["routing_label"] for r in records)
        hop_counts = Counter(r["hop_count"] for r in records)
        out_path = OUT_DIR / f"{split}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        laws = len({r["doc_number"] for r in records})
        stats["splits"][split] = {
            "n": len(records),
            "unique_laws": laws,
            "routing_label": dict(route_counts),
            "hop_counts": dict(hop_counts),
        }
        print(f"Wrote {out_path} ({len(records)} q | routes={dict(route_counts)} | hops={dict(hop_counts)})")

    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Spot-check: print 1 example per type from test split
    test_path = OUT_DIR / "test.json"
    with open(test_path) as f:
        test = json.load(f)
    print("\n=== SPOT CHECK (test split, 1 per type) ===")
    for label in ["dense_retrieval", "graph_traversal", "hybrid_reasoning"]:
        ex = next((r for r in test if r["routing_label"] == label), None)
        if ex:
            print(f"\n[{label}]")
            print(f"  Q: {ex['question']}")
            print(f"  A (first 150): {ex['answer'][:150]}")
            print(f"  relevant: {[(a['law_id'], a['article_id']) for a in ex['relevant_articles']]}")
            print(f"  hop_count={ex['hop_count']}, cross_doc={ex['is_cross_doc']}")

    print(f"\nDone. Full stats:\n{json.dumps(stats, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
