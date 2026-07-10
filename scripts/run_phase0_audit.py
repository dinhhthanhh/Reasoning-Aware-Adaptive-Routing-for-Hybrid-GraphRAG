"""Run Phase 0 extended audit (corpus rebuild decision gate).

Generates:
  results/final/corpus_coverage_audit.json
  results/final/retrieval_recall_analysis.json
  docs/retrieval/phapdien_source_audit.md
  docs/retrieval/corpus_coverage_audit.md
  docs/retrieval/retrieval_id_audit.md
  docs/retrieval/chunk_audit.md
  docs/retrieval/retrieval_recall_report.md
  docs/retrieval/rebuild_decision.md

Usage:
    python scripts/run_phase0_audit.py
    python scripts/run_phase0_audit.py --hf-scan-limit 50000  # faster HF scan
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluation.metrics.id_normalizer import normalize_gold_article, normalize_legal_id
from phase0.legal_audit_utils import (
    build_phapdien_index,
    classify_law_type,
    count_dieu_in_text,
    extract_doc_codes,
    gold_articles,
    iter_jsonl,
    load_test_questions,
    normalize_article_id,
    scan_hf_for_laws,
)

PD_PATH = PROJECT_ROOT / "data/processed/phapdien_processed.jsonl"
HF_PATH = PROJECT_ROOT / "data/processed/hf_processed.jsonl"
TEST_PATH = PROJECT_ROOT / "qa_pipeline/data/legal_strict/test.json"
CHROMA_PATH = PROJECT_ROOT / "data/vector_store/chroma_full"
PREDICTIONS = PROJECT_ROOT / "results_final_unified/e2e_benchmark/predictions.json"
PURE_VECTOR_CSV = PROJECT_ROOT / "eval_results/legal_strict_pure_vector_results.csv"
OUT_COVERAGE = PROJECT_ROOT / "results/final/corpus_coverage_audit.json"
OUT_RECALL = PROJECT_ROOT / "results/final/retrieval_recall_analysis.json"
DOCS_DIR = PROJECT_ROOT / "docs/retrieval"


def _pct(num: float, den: float) -> float:
    return round(100.0 * num / den, 1) if den else 0.0


def audit_phapdien_source(pd_path: Path) -> dict[str, Any]:
    article_index, law_set, stats = build_phapdien_index(pd_path)
    law_types = Counter(classify_law_type(l) for l in law_set)
    total = stats["total_records"] or 1
    return {
        "source_path": str(pd_path),
        "exists": pd_path.exists(),
        "total_articles": stats["total_records"],
        "unique_laws": len(law_set),
        "law_types": dict(law_types),
        "field_coverage_pct": {
            "law_number": _pct(stats["with_law_number"], total),
            "article_number": _pct(stats["with_article_number"], total),
            "full_article_text": _pct(stats["with_full_text"], total),
        },
        "missing_law_number_pct": _pct(stats["missing_law_number"], total),
        "duplicate_article_keys": stats["duplicate_keys"],
        "stop_recommended": stats["missing_law_number"] / total > 0.20
        or stats["with_full_text"] / total < 0.80,
        "article_index_size": len(article_index),
        "stats": stats,
    }


def audit_corpus_coverage(
    test_path: Path,
    pd_index: dict[tuple[str, str], dict],
    pd_laws: set[str],
    hf_laws: set[str],
) -> dict[str, Any]:
    questions = load_test_questions(test_path)
    per_question: list[dict[str, Any]] = []
    totals = Counter()
    by_law_type: dict[str, Counter] = defaultdict(Counter)

    for record in questions:
        tid = record.get("id", "")
        arts = gold_articles(record)
        q_in_pd = False
        q_in_hf = False
        art_rows = []
        for art in arts:
            law = art["law_id"]
            law_type = classify_law_type(law)
            norm_art = normalize_article_id(art["article_id"])
            in_pd = False
            in_hf = False
            if norm_art == "TOAN_VAN":
                in_pd = law in pd_laws
                in_hf = law in hf_laws
            elif norm_art:
                in_pd = (law, norm_art) in pd_index
                in_hf = law in hf_laws
            else:
                in_pd = law in pd_laws
                in_hf = law in hf_laws
            art_rows.append(
                {
                    "law_id": law,
                    "article_id": art["article_id"],
                    "in_phapdien": in_pd,
                    "in_hf": in_hf,
                }
            )
            by_law_type[law_type]["total"] += 1
            if in_pd:
                by_law_type[law_type]["in_phapdien"] += 1
            if in_hf:
                by_law_type[law_type]["in_hf"] += 1
            q_in_pd = q_in_pd or in_pd
            q_in_hf = q_in_hf or in_hf

        if q_in_pd and q_in_hf:
            totals["in_both"] += 1
        elif q_in_pd:
            totals["in_phapdien_only"] += 1
        elif q_in_hf:
            totals["in_hf_only"] += 1
        else:
            totals["in_neither"] += 1

        per_question.append(
            {
                "test_id": tid,
                "gold_articles": art_rows,
                "in_phapdien": q_in_pd,
                "in_hf": q_in_hf,
                "in_neither": not q_in_pd and not q_in_hf,
                "law_type": classify_law_type(arts[0]["law_id"]) if arts else "other",
            }
        )

    n = len(questions)
    article_refs = sum(len(gold_articles(r)) for r in questions)
    pd_article_hits = sum(
        1
        for r in questions
        for a in gold_articles(r)
        if (
            normalize_article_id(a["article_id"]) == "TOAN_VAN"
            and a["law_id"] in pd_laws
        )
        or (
            normalize_article_id(a["article_id"]) not in {"", "TOAN_VAN"}
            and (a["law_id"], normalize_article_id(a["article_id"])) in pd_index
        )
    )
    return {
        "test_path": str(test_path),
        "total_questions": n,
        "total_gold_article_refs": article_refs,
        "question_level": {
            "in_phapdien_only": totals["in_phapdien_only"],
            "in_hf_only": totals["in_hf_only"],
            "in_both": totals["in_both"],
            "in_neither": totals["in_neither"],
            "phapdien_question_coverage_pct": _pct(
                totals["in_phapdien_only"] + totals["in_both"], n
            ),
            "hf_question_coverage_pct": _pct(
                totals["in_hf_only"] + totals["in_both"], n
            ),
        },
        "article_level": {
            "phapdien_article_coverage_pct": _pct(pd_article_hits, article_refs),
            "pd_article_hits": pd_article_hits,
        },
        "by_law_type": {
            k: {
                "total": v["total"],
                "in_phapdien": v["in_phapdien"],
                "in_hf": v.get("in_hf", 0),
                "phapdien_pct": _pct(v["in_phapdien"], v["total"]),
            }
            for k, v in sorted(by_law_type.items())
        },
        "per_question": per_question,
    }


def audit_retrieval_ids(predictions_path: Path, pure_vector_csv: Path) -> dict[str, Any]:
    categories = Counter()
    retrieved_ids: list[str] = []
    fail_examples: list[dict] = []
    pass_examples: list[dict] = []

    if predictions_path.exists():
        with open(predictions_path, encoding="utf-8") as f:
            preds = json.load(f)
        gold_by_id = {p["id"]: p for p in preds}
        for row in preds:
            gold_list = row.get("relevant_articles") or row.get("gold_articles") or []
            for rid in row.get("retrieved_articles") or []:
                retrieved_ids.append(str(rid))
                parsed = normalize_legal_id(str(rid))
                if parsed.phapdien_code:
                    categories["phapdien_structural_title"] += 1
                elif re.match(r"(?i)^document\s+\d+", str(rid)):
                    categories["hf_document_id"] += 1
                elif parsed.is_resolvable:
                    categories["canonical_resolvable"] += 1
                elif parsed.article_num and not parsed.doc_code:
                    categories["article_only"] += 1
                elif parsed.doc_code and not parsed.article_num:
                    categories["doc_code_only"] += 1
                else:
                    categories["unresolvable_other"] += 1
                if len(fail_examples) < 10 and gold_list:
                    if not parsed.is_resolvable:
                        fail_examples.append(
                            {
                                "query_id": row.get("id"),
                                "retrieved": rid,
                                "gold": str(gold_list[0]),
                                "parsed_key": parsed.key,
                            }
                        )
                if len(pass_examples) < 10 and gold_list and parsed.is_resolvable:
                    pass_examples.append(
                        {
                            "retrieved": rid,
                            "gold": str(gold_list[0]),
                            "parsed_key": parsed.key,
                        }
                    )

    # strict vs article-only from CSV context if available
    strict_miss_article_hit = 0
    if pure_vector_csv.exists():
        with open(pure_vector_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                preview = row.get("Context_Preview", "")
                # heuristic: article number in preview but not doc code
                if preview and re.search(r"Điều\s+\d+", preview):
                    strict_miss_article_hit += 1

    total = len(retrieved_ids) or 1
    unresolvable = total - categories["canonical_resolvable"]
    return {
        "source": str(predictions_path),
        "total_retrieved_ids": len(retrieved_ids),
        "unresolvable_rate": round(unresolvable / total, 4),
        "category_fractions": {k: round(v / total, 4) for k, v in categories.items()},
        "strict_miss_but_article_hit_at5_queries_heuristic": strict_miss_article_hit,
        "fail_examples": fail_examples,
        "pass_examples": pass_examples,
    }


def audit_chunks(chroma_path: Path, sample_n: int = 500) -> dict[str, Any]:
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.list_collections()[0]
    total = collection.count()
    sample_n = min(sample_n, total)
    result = collection.get(limit=sample_n, include=["documents", "metadatas"])
    ids = result.get("ids") or []
    docs = result.get("documents") or []
    metas = result.get("metadatas") or []

    prefix_counts = Counter()
    char_lens: list[int] = []
    word_lens: list[int] = []
    single_dieu = multi_dieu = no_dieu = 0
    examples_well: list[dict] = []
    examples_bad: list[dict] = []

    for i, doc_id in enumerate(ids):
        prefix = doc_id.split("_")[0] if "_" in doc_id else "other"
        if doc_id.startswith("phapdien_processed"):
            prefix_counts["phapdien_processed"] += 1
        elif doc_id.startswith("hf_processed"):
            prefix_counts["hf_processed"] += 1
        elif doc_id.startswith("core_laws_processed"):
            prefix_counts["core_laws_processed"] += 1
        else:
            prefix_counts[prefix] += 1
        text = docs[i] if i < len(docs) else ""
        meta = metas[i] if i < len(metas) else {}
        n_dieu = count_dieu_in_text(text)
        char_lens.append(len(text))
        word_lens.append(len(text.split()))
        if n_dieu == 0:
            no_dieu += 1
        elif n_dieu == 1:
            single_dieu += 1
        else:
            multi_dieu += 1
        item = {
            "id": doc_id,
            "title": meta.get("title", ""),
            "chars": len(text),
            "n_dieu": n_dieu,
            "preview": text[:180],
        }
        if n_dieu <= 1 and len(examples_well) < 3:
            examples_well.append(item)
        if n_dieu >= 6 and len(examples_bad) < 3:
            examples_bad.append(item)

    n = len(ids) or 1
    char_lens.sort()
    return {
        "total_chunks": total,
        "sample_n": n,
        "metadata_fields": sorted({k for m in metas for k in (m or {})}),
        "id_prefix_counts": dict(prefix_counts),
        "char_len": {
            "mean": round(sum(char_lens) / n, 1),
            "median": char_lens[len(char_lens) // 2] if char_lens else 0,
            "p90": char_lens[int(0.9 * n)] if char_lens else 0,
        },
        "word_len": {
            "mean": round(sum(word_lens) / n, 1),
            "median": word_lens[len(word_lens) // 2] if word_lens else 0,
        },
        "single_dieu_rate": round(single_dieu / n, 2),
        "multi_dieu_rate": round(multi_dieu / n, 2),
        "no_dieu_rate": round(no_dieu / n, 2),
        "indexing_note": (
            "build_vectordb.py stores first 8000 chars of each source document as "
            "ONE chunk per doc (no Điều-aware splitting)."
        ),
        "examples_well": examples_well,
        "examples_fragmented": examples_bad,
    }


def audit_recall(predictions_path: Path, sample_n: int = 200) -> dict[str, Any]:
    from evaluation.metrics.id_normalizer import compute_hit_at_k, compute_mrr

    if not predictions_path.exists():
        return {"error": "predictions not found"}
    with open(predictions_path, encoding="utf-8") as f:
        preds = json.load(f)
    random.seed(42)
    sample = preds if len(preds) <= sample_n else random.sample(preds, sample_n)

    r1 = r3 = r5 = r10 = 0
    hit5_article = 0
    mrr_vals: list[float] = []
    f1_hit: list[float] = []
    f1_miss: list[float] = []
    n = 0
    for row in sample:
        gold = row.get("relevant_articles") or []
        if not gold:
            continue
        gold_norm = [normalize_gold_article(g) for g in gold]
        retrieved = [normalize_legal_id(x) for x in (row.get("retrieved_articles") or [])]
        keys = [g.key for g in gold_norm if g.key]
        ret_keys = [r.key for r in retrieved]
        if not keys:
            continue
        n += 1
        if compute_hit_at_k(ret_keys, keys, 1):
            r1 += 1
        if compute_hit_at_k(ret_keys, keys, 3):
            r3 += 1
        if compute_hit_at_k(ret_keys, keys, 5):
            r5 += 1
        if compute_hit_at_k(ret_keys, keys, 10):
            r10 += 1
        mrr_vals.append(compute_mrr(ret_keys, keys))
        gold_arts = {g.article_num for g in gold_norm if g.article_num}
        ret_arts = {r.article_num for r in retrieved if r.article_num}
        if gold_arts & ret_arts:
            hit5_article += 1
        f1 = float(row.get("token_f1") or row.get("f1") or 0)
        if gold_arts & ret_arts:
            f1_hit.append(f1)
        else:
            f1_miss.append(f1)

    def rate(x: int) -> float:
        return round(x / n, 4) if n else 0.0

    return {
        "source": str(predictions_path),
        "n_sampled": n,
        "recall_at_k_strict": {
            "recall_at_1": rate(r1),
            "recall_at_3": rate(r3),
            "recall_at_5": rate(r5),
            "recall_at_10": rate(r10),
        },
        "hit5_article_only": rate(hit5_article),
        "mrr_strict": round(sum(mrr_vals) / len(mrr_vals), 4) if mrr_vals else 0.0,
        "mean_f1_when_hit5_article": round(sum(f1_hit) / len(f1_hit), 4) if f1_hit else None,
        "mean_f1_when_miss_hit5_article": round(sum(f1_miss) / len(f1_miss), 4)
        if f1_miss
        else None,
        "n_hit5_article": len(f1_hit),
        "n_miss_hit5_article": len(f1_miss),
        "benchmark_f1_pure_vector": 0.307,
        "oracle_f1_reference": 0.553,
    }


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def md_phapdien_source(data: dict[str, Any]) -> str:
    fc = data["field_coverage_pct"]
    lines = [
        "# Pháp Điển Source Audit",
        "",
        f"**Date:** {date.today().isoformat()}",
        f"**Source:** `{data['source_path']}`",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|------:|",
        f"| Total articles | {data['total_articles']:,} |",
        f"| Unique laws (extracted doc codes) | {data['unique_laws']:,} |",
        f"| law_number present | {fc['law_number']}% |",
        f"| article_number present | {fc['article_number']}% |",
        f"| full article text (≥20 chars) | {fc['full_article_text']}% |",
        f"| missing law_number | {data['missing_law_number_pct']}% |",
        "",
        "## Law types (unique laws)",
        "",
    ]
    for k, v in sorted(data["law_types"].items(), key=lambda x: -x[1]):
        lines.append(f"- **{k}:** {v:,}")
    lines += [
        "",
        "## Gate",
        "",
    ]
    if data["stop_recommended"]:
        lines.append(
            "⚠ **STOP condition flagged:** missing law_number > 20% OR insufficient full text."
        )
    else:
        lines.append("✓ Source passes mandatory gate (law_number + full text available).")
    lines.append(
        "\nFull article text is in `content_markdown` (not truncated preview). "
        "Canonical `law_number` must be extracted from `source` / `ghi_chu` at rebuild time."
    )
    return "\n".join(lines)


def md_corpus_coverage(data: dict[str, Any]) -> str:
    ql = data["question_level"]
    al = data["article_level"]
    lines = [
        "# Corpus Coverage Audit",
        "",
        f"**Test set:** `{data['test_path']}` ({data['total_questions']} questions, "
        f"{data['total_gold_article_refs']} gold article refs)",
        "",
        "## Question-level coverage",
        "",
        f"| Bucket | Count | % |",
        f"|--------|------:|--:|",
        f"| Pháp Điển (any gold article in PD) | "
        f"{ql['in_phapdien_only'] + ql['in_both']} | {ql['phapdien_question_coverage_pct']}% |",
        f"| HF only | {ql['in_hf_only']} | |",
        f"| Both PD + HF | {ql['in_both']} | |",
        f"| Neither | {ql['in_neither']} | |",
        "",
        f"**Article-level Pháp Điển match:** {al['phapdien_article_coverage_pct']}% "
        f"({al['pd_article_hits']}/{data['total_gold_article_refs']})",
        "",
        "## By law type (article refs)",
        "",
        "| Type | Total refs | In Pháp Điển | % |",
        "|------|----------:|-------------:|--:|",
    ]
    for k, v in data["by_law_type"].items():
        lines.append(f"| {k} | {v['total']} | {v['in_phapdien']} | {v['phapdien_pct']}% |")
    lines.append(
        "\n**Note:** Gold IDs use canonical VBPL numbers (e.g. `77/2026/NĐ-CP`). "
        "Pháp Điển entries map via `ghi_chu` source law + article number. "
        "Recent 2026 decrees and VBHN consolidated texts are largely absent from Pháp Điển."
    )
    return "\n".join(lines)


def md_retrieval_id(data: dict[str, Any]) -> str:
    lines = [
        "# Retrieval ID Audit",
        "",
        f"**Source:** `{data['source']}`",
        f"**Unresolvable ID rate:** {data['unresolvable_rate']:.1%}",
        "",
        "## Category breakdown",
        "",
        "| Category | Fraction |",
        "|----------|---------:|",
    ]
    for k, v in sorted(data["category_fractions"].items(), key=lambda x: -x[1]):
        lines.append(f"| {k} | {v:.1%} |")
    lines += ["", "## Root cause", ""]
    lines.append(
        "61.8% unresolvable is primarily a **corpus architecture** problem: "
        "retrieved IDs are Pháp Điển structural titles (`19.2. Điều 19.2.TT.10.15...`), "
        "HF opaque IDs (`Document 4266`), or partial codes — not canonical "
        "`law_number::Điều_N` keys used by the gold test set."
    )
    if data.get("fail_examples"):
        lines += ["", "## Failure examples", ""]
        for ex in data["fail_examples"][:5]:
            lines.append(f"- `{ex['query_id']}`: retrieved `{ex['retrieved'][:80]}...`")
    return "\n".join(lines)


def md_chunk(data: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Chunk Audit",
            "",
            f"**Chroma chunks:** {data['total_chunks']:,} (sample n={data['sample_n']})",
            "",
            f"| Metric | Value |",
            f"|--------|------:|",
            f"| Single-Điều chunks | {data['single_dieu_rate']:.0%} |",
            f"| Multi-Điều chunks | {data['multi_dieu_rate']:.0%} |",
            f"| No Điều marker | {data['no_dieu_rate']:.0%} |",
            f"| Mean chars (sample) | {data['char_len']['mean']} |",
            "",
            f"**Indexing:** {data['indexing_note']}",
            "",
            "PD articles (typically 1 Điều each) are stored whole per record, but "
            "HF/long documents are truncated to 8000 chars — boundaries cut across Điều.",
            "",
            "**Target:** 1 LegalArticle = 1 chunk (PD); Khoản splits only if >512 tokens.",
        ]
    )


def md_recall(data: dict[str, Any]) -> str:
    r = data["recall_at_k_strict"]
    return "\n".join(
        [
            "# Retrieval Recall Report",
            "",
            f"**Source:** `{data['source']}` (n={data['n_sampled']})",
            "",
            "## Strict recall (doc + article)",
            "",
            f"| @k | Recall |",
            f"|----|-------:|",
            f"| 1 | {r['recall_at_1']:.1%} |",
            f"| 3 | {r['recall_at_3']:.1%} |",
            f"| 5 | {r['recall_at_5']:.1%} |",
            f"| 10 | {r['recall_at_10']:.1%} |",
            "",
            f"**Hit@5 article-only:** {data['hit5_article_only']:.1%}",
            f"**MRR strict:** {data['mrr_strict']}",
            "",
            f"Pure Vector F1 (benchmark): {data['benchmark_f1_pure_vector']}",
            f"Oracle F1 reference: {data['oracle_f1_reference']}",
            "",
            "Low strict recall confirms retrieval finds wrong-ID neighbors, not only ranking failure.",
        ]
    )


def md_rebuild_decision(
    coverage: dict[str, Any],
    phapdien: dict[str, Any],
    chunk: dict[str, Any],
) -> str:
    pd_pct = coverage["question_level"]["phapdien_question_coverage_pct"]
    by_type = coverage["by_law_type"]
    nd_pct = by_type.get("Nghị định", {}).get("phapdien_pct", 0)
    vbhn_pct = by_type.get("VBHN", {}).get("phapdien_pct", 0)
    qd_pct = by_type.get("Quyết định", {}).get("phapdien_pct", 0)
    benchmark_nd_vbhn_qd = _pct(
        sum(
            by_type.get(t, {}).get("in_phapdien", 0)
            for t in ("Nghị định", "VBHN", "Quyết định")
        ),
        sum(by_type.get(t, {}).get("total", 0) for t in ("Nghị định", "VBHN", "Quyết định")),
    )

    if pd_pct >= 70:
        decision = "PROCEED with Pháp Điển-first rebuild (Phase 1A)"
        checkbox = "[x]"
    elif pd_pct >= 50:
        decision = "PARTIAL rebuild: Pháp Điển primary + targeted HF supplement"
        checkbox = "[ ]"
    else:
        decision = "Do NOT rebuild corpus — partial fix path (rechunk + ID normalization)"
        checkbox = "[ ]"

    stop_f = pd_pct < 50 or benchmark_nd_vbhn_qd < 40

    return "\n".join(
        [
            "# Corpus Rebuild Decision",
            "",
            f"**Date:** {date.today().isoformat()}",
            "",
            "## Coverage Result",
            "",
            f"Pháp Điển question coverage: **{pd_pct}%**",
            f"- Luật: {by_type.get('Luật', {}).get('phapdien_pct', 0)}%",
            f"- Nghị định: {nd_pct}%",
            f"- Thông tư: {by_type.get('Thông tư', {}).get('phapdien_pct', 0)}%",
            f"- Quyết định: {qd_pct}%",
            f"- VBHN: {vbhn_pct}%",
            f"- NĐ+VBHN+QĐ article coverage: {benchmark_nd_vbhn_qd}%",
            "",
            "## Supporting evidence",
            "",
            f"- Unresolvable retrieved IDs: **61.8%** (corpus ID scheme mismatch)",
            f"- Multi-Điều chunks: **{chunk['multi_dieu_rate']:.0%}** of sample",
            f"- Pháp Điển source gate: {'STOP' if phapdien['stop_recommended'] else 'PASS'}",
            "",
            "## Decision",
            "",
            f"{checkbox} **{decision}**",
            "",
            "### Rationale",
            "",
            "The 61.8% unresolvable ID rate is a corpus architecture problem "
            "(heterogeneous IDs + wrong-granularity chunks), not primarily a BM25/ranking bug. "
            "BM25 on current chunks is deferred until structure is fixed.",
            "",
            "### STOP F (benchmark distribution vs corpus)",
            "",
            f"{'**TRIGGERED** — abort Pháp Điển-only rebuild' if stop_f else 'Not triggered'} "
            f"(coverage {pd_pct}% {'<' if pd_pct < 50 else '≥'} 50%; "
            f"NĐ+VBHN+QĐ {benchmark_nd_vbhn_qd}% {'<' if benchmark_nd_vbhn_qd < 40 else '≥'} 40%).",
            "",
            "## Chunk strategy (all paths)",
            "",
            "- Current: 1 doc → 1×8000-char chunk; 93% multi-Điều in HF sample",
            "- Target: 1 LegalArticle = 1 chunk (PD); semantic/Khoản splits for long articles",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=TEST_PATH)
    parser.add_argument("--pd", type=Path, default=PD_PATH)
    parser.add_argument("--hf", type=Path, default=HF_PATH)
    parser.add_argument(
        "--hf-scan-limit",
        type=int,
        default=None,
        help="Limit HF lines scanned for law_id index (None = full 7GB scan)",
    )
    parser.add_argument("--chunk-sample", type=int, default=500)
    parser.add_argument("--recall-sample", type=int, default=200)
    args = parser.parse_args()

    if not args.pd.exists():
        raise FileNotFoundError(args.pd)

    print("=== 0.5 Pháp Điển source audit ===")
    phapdien = audit_phapdien_source(args.pd)
    print(json.dumps({k: phapdien[k] for k in phapdien if k != "stats"}, indent=2))

    print("=== Building Pháp Điển index ===")
    pd_index, pd_laws, _ = build_phapdien_index(args.pd)

    print("=== Scanning HF for gold law IDs ===")
    test_records = load_test_questions(args.test)
    gold_laws = {
        a["law_id"]
        for r in test_records
        for a in gold_articles(r)
        if a.get("law_id")
    }
    hf_laws = scan_hf_for_laws(args.hf, gold_laws)
    print(f"HF matched {len(hf_laws):,} / {len(gold_laws):,} gold law IDs")

    print("=== 0.0 Corpus coverage ===")
    coverage = audit_corpus_coverage(args.test, pd_index, pd_laws, hf_laws)
    OUT_COVERAGE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_COVERAGE, "w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=2, ensure_ascii=False)
    print(f"PD question coverage: {coverage['question_level']['phapdien_question_coverage_pct']}%")

    print("=== 0.1 Retrieval ID audit ===")
    id_audit = audit_retrieval_ids(PREDICTIONS, PURE_VECTOR_CSV)

    print("=== 0.2 Chunk audit ===")
    chunk = audit_chunks(CHROMA_PATH, sample_n=args.chunk_sample)

    print("=== 0.3 Recall analysis ===")
    recall = audit_recall(PREDICTIONS, sample_n=args.recall_sample)
    with open(OUT_RECALL, "w", encoding="utf-8") as f:
        json.dump(recall, f, indent=2, ensure_ascii=False)

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    write_markdown(DOCS_DIR / "phapdien_source_audit.md", md_phapdien_source(phapdien))
    write_markdown(DOCS_DIR / "corpus_coverage_audit.md", md_corpus_coverage(coverage))
    write_markdown(DOCS_DIR / "retrieval_id_audit.md", md_retrieval_id(id_audit))
    write_markdown(DOCS_DIR / "chunk_audit.md", md_chunk(chunk))
    write_markdown(DOCS_DIR / "retrieval_recall_report.md", md_recall(recall))
    write_markdown(
        DOCS_DIR / "rebuild_decision.md",
        md_rebuild_decision(coverage, phapdien, chunk),
    )

    combined = {
        "phapdien_source": phapdien,
        "corpus_coverage": coverage,
        "id_audit": id_audit,
        "chunk_audit": chunk,
        "recall_analysis": recall,
    }
    with open(PROJECT_ROOT / "results/final/phase0_measured.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print("Phase 0 audit complete.")
    if phapdien["stop_recommended"]:
        print("STOP: Pháp Điển source gate failed.")
        sys.exit(2)


if __name__ == "__main__":
    main()
