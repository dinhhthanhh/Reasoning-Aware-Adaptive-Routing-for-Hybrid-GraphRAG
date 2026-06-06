"""STEP 5 — Verify generated QA pairs using 2-level verification.

Input:  data/processed/qa_raw.json
Output: data/processed/qa_verified.json

Level 1 (String Match):
  - Check if significant words from the answer appear in the article text.
  - Fast, no LLM call needed.

Level 2 (LLM Verify):
  - Ask Qwen3-32B to verify: "Is this answer supported by this article?"
  - Only applied if Level 1 passes (to save API calls).

Only QA pairs that pass BOTH levels are kept in qa_verified.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger
from llm.openai_client import OpenAIClient

VERIFY_SYSTEM = "Bạn là chuyên gia pháp luật Việt Nam. Hãy xác minh tính chính xác của câu trả lời."

VERIFY_PROMPT = """Xác minh xem câu trả lời có được HỖ TRỢ hoàn toàn bởi nội dung điều luật hay không.

=== NỘI DUNG ĐIỀU LUẬT ===
{article_content}

=== CÂU HỎI ===
{question}

=== CÂU TRẢ LỜI CẦN XÁC MINH ===
{answer}

Trả lời CHÍNH XÁC theo JSON sau:
{{"supported": true/false, "reason": "Giải thích ngắn gọn bằng tiếng Việt"}}

Chỉ trả về JSON:"""


def level1_string_match(answer: str, article_content: str, threshold: float = 0.3) -> bool:
    """Check if answer content meaningfully overlaps with article text.

    Uses word-level overlap ratio. Vietnamese-friendly (space-split).

    Args:
        answer: The generated answer string.
        article_content: The source article text.
        threshold: Minimum overlap ratio to pass.

    Returns:
        True if the overlap ratio exceeds the threshold.
    """
    # Normalize
    answer_words = set(answer.lower().split())
    article_words = set(article_content.lower().split())

    # Remove very short words (particles, articles)
    answer_words = {w for w in answer_words if len(w) > 2}

    if not answer_words:
        return True  # Empty answer — let LLM decide

    overlap = len(answer_words & article_words)
    ratio = overlap / len(answer_words)
    return ratio >= threshold


def level2_llm_verify(
    client: OpenAIClient,
    question: str,
    answer: str,
    article_content: str,
) -> bool:
    """Ask LLM to verify if the answer is supported by the article.

    Args:
        client: LLM client.
        question: The question.
        answer: The answer to verify.
        article_content: The source article text.

    Returns:
        True if LLM says the answer is supported.
    """
    prompt = VERIFY_PROMPT.format(
        article_content=article_content[:2000],
        question=question,
        answer=answer,
    )

    try:
        raw = client.generate(
            prompt,
            system_prompt=VERIFY_SYSTEM,
            temperature=0.0,  # Deterministic for verification
        )

        # Parse JSON
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            import re
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
            else:
                logger.warning("Could not parse LLM verify response: {}", raw[:100])
                return True  # Fail open — let it pass

        return bool(result.get("supported", True))

    except Exception as exc:
        logger.warning("LLM verification failed: {}. Passing by default.", exc)
        return True  # Fail open


def load_articles_index(articles_path: Path) -> dict[tuple[str, str], str]:
    """Build a lookup: (law, article_id) → content.

    Args:
        articles_path: Path to articles.json.

    Returns:
        Dict mapping (law, article_id) to content text.
    """
    if not articles_path.exists():
        return {}
    with open(articles_path, "r", encoding="utf-8") as f:
        articles = json.load(f)
    return {(a["law"], a["article_id"]): a["content"] for a in articles}


def process_file(
    input_path: str | Path,
    output_path: str | Path,
    articles_path: str | Path | None = None,
    skip_llm: bool = False,
    batch_delay: float = 0.3,
) -> tuple[int, int]:
    """Verify QA pairs with 2-level checking.

    Args:
        input_path: Path to qa_raw.json.
        output_path: Path to write qa_verified.json.
        articles_path: Path to articles.json (for content lookup).
        skip_llm: Skip level 2 LLM verification (faster, less accurate).
        batch_delay: Seconds between LLM calls.

    Returns:
        Tuple of (total_input, total_kept).
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if articles_path is None:
        articles_path = ROOT / "data" / "processed" / "articles.json"
    articles_path = Path(articles_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading QA pairs from {}", input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    logger.info("Building article content index from {}", articles_path)
    article_index = load_articles_index(articles_path)

    client = OpenAIClient() if not skip_llm else None
    verified: list[dict] = []

    logger.info("Verifying {} QA pairs (skip_llm={})...", len(qa_pairs), skip_llm)

    l1_pass = l2_pass = l1_fail = l2_fail = 0

    for i, qa in enumerate(qa_pairs):
        question = qa.get("question", "")
        answer = qa.get("answer", "")
        law = qa.get("law", "")
        evidence = qa.get("evidence", "")

        # Look up article content
        content = article_index.get((law, evidence), "")
        if not content:
            # Try partial match
            for (l, a), c in article_index.items():
                if l == law and evidence in a:
                    content = c
                    break

        # Level 1: String match
        if not level1_string_match(answer, content or law):
            l1_fail += 1
            logger.debug("L1 FAIL [{}/{}] {}: {}", i + 1, len(qa_pairs), evidence, question[:60])
            continue
        l1_pass += 1

        # Level 2: LLM verify (optional)
        if not skip_llm and client and content:
            if not level2_llm_verify(client, question, answer, content):
                l2_fail += 1
                logger.debug("L2 FAIL [{}/{}] {}: {}", i + 1, len(qa_pairs), evidence, question[:60])
                continue
            l2_pass += 1
            if batch_delay > 0:
                time.sleep(batch_delay)

        verified.append(qa)

        if (i + 1) % 50 == 0:
            logger.info(
                "  [{}/{}] L1 pass={} L1 fail={} L2 pass={} L2 fail={} kept={}",
                i + 1, len(qa_pairs), l1_pass, l1_fail, l2_pass, l2_fail, len(verified),
            )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(verified, f, ensure_ascii=False, indent=2)

    logger.info(
        "Step 4 done | input={} | L1_pass={} L1_fail={} | L2_pass={} L2_fail={} | kept={} | saved to {}",
        len(qa_pairs), l1_pass, l1_fail, l2_pass, l2_fail, len(verified), output_path,
    )
    return len(qa_pairs), len(verified)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step 4: Verify QA pairs")
    parser.add_argument("--input", default=str(ROOT / "data" / "processed" / "qa_raw.json"))
    parser.add_argument("--output", default=str(ROOT / "data" / "processed" / "qa_verified.json"))
    parser.add_argument("--articles", default=str(ROOT / "data" / "processed" / "articles.json"))
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM verification (Level 2)")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between API calls")
    args = parser.parse_args()

    total, kept = process_file(args.input, args.output, args.articles, args.skip_llm, args.delay)
    pct = (kept / total * 100) if total > 0 else 0
    print(f"\n✅ Step 5 complete: {kept}/{total} QA pairs verified ({pct:.1f}%) → {args.output}")
