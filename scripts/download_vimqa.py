"""Download and convert ViMQA dataset for routing ablation study.

ViMQA (Vietnamese Multi-hop QA) is a HotpotQA-style dataset with ~10K
Vietnamese Wikipedia multi-hop questions. This script downloads it and
converts to our routing evaluation format with auto-labeled routing_label.

Usage:
  pip install datasets
  python -m scripts.download_vimqa
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from loguru import logger


def auto_label_route(question: str, answer: str, supporting_facts: list) -> dict:
    """Auto-label routing class based on question structure.

    Heuristic labeling (since ViMQA doesn't have routing labels):
    - All ViMQA questions are multi-hop by design → most should be graph_traversal
    - Comparison questions (so sánh, ai hơn, nào lớn hơn) → graph_traversal
    - Questions referencing 2+ Wikipedia titles → hybrid_reasoning
    - Simple single-fact → dense_retrieval (rare in ViMQA)

    Returns:
        Dict with routing_label, hop_count, is_cross_doc.
    """
    q_lower = question.lower()

    # Count unique supporting document titles
    unique_titles = set()
    for sf in supporting_facts:
        if isinstance(sf, (list, tuple)) and len(sf) >= 1:
            unique_titles.add(sf[0])

    is_cross_doc = len(unique_titles) >= 2
    hop_count = len(unique_titles)  # Approximate: 1 title = 1 hop

    # Comparison patterns (Vietnamese)
    comparison_patterns = re.compile(
        r"(?:so sánh|ai hơn|nào lớn hơn|nào cao hơn|nào nhiều hơn|"
        r"khác nhau|giống nhau|cả hai|hay là|đúng hơn|hơn hay)",
        re.IGNORECASE,
    )
    is_comparison = bool(comparison_patterns.search(q_lower))

    # Yes/No patterns
    yesno_patterns = re.compile(
        r"(?:có đúng|đúng không|phải không|có phải|đúng hay sai)",
        re.IGNORECASE,
    )
    is_yesno = bool(yesno_patterns.search(q_lower))

    # Route labeling
    if is_cross_doc and (is_comparison or hop_count >= 3):
        routing_label = "hybrid_reasoning"
    elif is_cross_doc or hop_count >= 2:
        routing_label = "graph_traversal"
    else:
        routing_label = "dense_retrieval"

    return {
        "routing_label": routing_label,
        "hop_count": hop_count,
        "is_cross_doc": is_cross_doc,
        "is_comparison": is_comparison,
        "is_yesno": is_yesno,
    }


def download_and_convert(output_dir: str = "data/vimqa", max_samples: int | None = None):
    """Download ViMQA from HuggingFace and convert to our format."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install datasets: pip install datasets")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading ViMQA from HuggingFace (SEACrowd/vimqa)...")

    try:
        # Try loading with SEACrowd schema
        dset = load_dataset("SEACrowd/vimqa", trust_remote_code=True)
    except Exception as e1:
        logger.warning("SEACrowd loader failed: {}. Trying alternative...", e1)
        try:
            # Alternative: direct JSON from GitHub
            dset = load_dataset("vimqa/vimqa", trust_remote_code=True)
        except Exception as e2:
            logger.error("All download methods failed: {}", e2)
            logger.info("Please download manually from: https://github.com/vimqa/vimqa")
            logger.info("Then place the JSON files in {}", output_path)
            return

    # Process each split
    stats = {"total": 0, "dense_retrieval": 0, "graph_traversal": 0, "hybrid_reasoning": 0}

    for split_name in ["train", "validation", "test"]:
        if split_name not in dset:
            logger.warning("Split '{}' not found, skipping", split_name)
            continue

        split = dset[split_name]
        converted = []

        for i, item in enumerate(split):
            if max_samples and i >= max_samples:
                break

            question = item.get("question", "")
            answer = item.get("answer", "")
            supporting_facts = item.get("supporting_facts", [])

            # Handle SEACrowd format vs raw format
            if isinstance(supporting_facts, dict):
                # SEACrowd format: {"title": [...], "sent_id": [...]}
                titles = supporting_facts.get("title", [])
                sent_ids = supporting_facts.get("sent_id", [])
                supporting_facts = list(zip(titles, sent_ids))

            # Build context from paragraphs
            context_paragraphs = item.get("context", [])
            context_text = ""
            if isinstance(context_paragraphs, dict):
                # SEACrowd format
                ctx_titles = context_paragraphs.get("title", [])
                ctx_sentences = context_paragraphs.get("sentences", [])
                for title, sents in zip(ctx_titles, ctx_sentences):
                    context_text += f"## {title}\n" + " ".join(sents) + "\n\n"
            elif isinstance(context_paragraphs, list):
                for ctx in context_paragraphs:
                    if isinstance(ctx, (list, tuple)) and len(ctx) >= 2:
                        title, sents = ctx[0], ctx[1]
                        if isinstance(sents, list):
                            context_text += f"## {title}\n" + " ".join(sents) + "\n\n"

            # Auto-label routing
            labels = auto_label_route(question, answer, supporting_facts)

            record = {
                "question": question,
                "answer": answer,
                "routing_label": labels["routing_label"],
                "hop_count": labels["hop_count"],
                "is_cross_doc": labels["is_cross_doc"],
                "context": context_text.strip(),
                "supporting_facts": [[sf[0], sf[1]] for sf in supporting_facts if len(sf) >= 2],
                "source": "vimqa",
            }

            converted.append(record)
            stats["total"] += 1
            stats[labels["routing_label"]] += 1

        # Save split
        out_file = output_path / f"{split_name}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)

        logger.info("Saved {} → {} samples", split_name, len(converted))

    # Print stats
    logger.info("=" * 50)
    logger.info("ViMQA Conversion Summary:")
    logger.info("  Total: {}", stats["total"])
    logger.info("  dense_retrieval: {} ({:.1f}%)",
                stats["dense_retrieval"],
                100 * stats["dense_retrieval"] / max(stats["total"], 1))
    logger.info("  graph_traversal: {} ({:.1f}%)",
                stats["graph_traversal"],
                100 * stats["graph_traversal"] / max(stats["total"], 1))
    logger.info("  hybrid_reasoning: {} ({:.1f}%)",
                stats["hybrid_reasoning"],
                100 * stats["hybrid_reasoning"] / max(stats["total"], 1))
    logger.info("Output directory: {}", output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and convert ViMQA dataset")
    parser.add_argument("--output", default="data/vimqa", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per split (for quick testing)")

    args = parser.parse_args()
    download_and_convert(output_dir=args.output, max_samples=args.max_samples)
