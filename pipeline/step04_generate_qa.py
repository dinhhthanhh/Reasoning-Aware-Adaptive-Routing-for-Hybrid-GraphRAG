"""STEP 4 — Generate QA pairs from legal articles using Qwen3-32B-AWQ.

Input:  data/processed/articles.json
Output: data/processed/qa_raw.json

Each output record:
  {
    "question": "...",
    "answer": "...",
    "evidence": "Điều X",
    "law": "...",
    "doc_number": "...",
    "url": "..."
  }

Generates 2-3 QA pairs per article. Batch-processes with progress tracking
and checkpoint support for resuming interrupted runs.
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


SYSTEM_PROMPT = """Bạn là chuyên gia pháp luật Việt Nam. Nhiệm vụ của bạn là tạo ra các cặp hỏi-đáp 
dựa HOÀN TOÀN vào nội dung điều luật được cung cấp.

Quy tắc BẮT BUỘC:
1. Câu hỏi phải thực tế, phù hợp với những gì công dân/doanh nghiệp thường hỏi.
2. Câu trả lời phải trích dẫn chính xác từ nội dung điều luật. KHÔNG tự ý bịa đặt.
3. Chỉ trả lời bằng JSON hợp lệ, không thêm bất kỳ văn bản nào khác.
4. Tất cả nội dung phải bằng tiếng Việt."""

QA_PROMPT_TEMPLATE = """Dựa vào nội dung điều luật sau đây, tạo 2-3 cặp hỏi-đáp pháp lý thực tế:

=== NỘI DUNG ĐIỀU LUẬT ===
Văn bản: {law}
{article_id}: {content}
=========================

Trả lời theo định dạng JSON sau (mảng các đối tượng):
[
  {{
    "question": "Câu hỏi thực tế bằng tiếng Việt",
    "answer": "Câu trả lời trích từ điều luật",
    "evidence": "{article_id}"
  }}
]

Chỉ trả về JSON, không giải thích thêm:"""


def generate_qa_for_article(
    client: OpenAIClient,
    article: dict,
    max_retries: int = 2,
) -> list[dict]:
    """Generate QA pairs for a single article.

    Args:
        client: Initialized OpenAI-compatible LLM client.
        article: Article dict with law, article_id, content fields.
        max_retries: Number of retry attempts on parse failure.

    Returns:
        List of QA pair dicts (may be empty on failure).
    """
    law = article.get("law", "")
    article_id = article.get("article_id", "Điều ?")
    content = article.get("content", "")
    doc_number = article.get("doc_number", "")
    url = article.get("url", "")
    article_key = article.get("article_key", f"{doc_number}::{article_id}")

    prompt = QA_PROMPT_TEMPLATE.format(
        law=law,
        article_id=article_id,
        content=content[:2000],  # Limit input size
    )

    for attempt in range(1, max_retries + 1):
        try:
            raw = client.generate(prompt, system_prompt=SYSTEM_PROMPT, temperature=0.3)

            # Parse JSON
            try:
                qa_list = json.loads(raw)
            except json.JSONDecodeError:
                # Try to extract JSON array from markdown or surrounding text
                import re
                match = re.search(r"\[.*\]", raw, re.DOTALL)
                if match:
                    qa_list = json.loads(match.group(0))
                else:
                    raise ValueError("No JSON array found in response")

            if not isinstance(qa_list, list):
                raise ValueError(f"Expected list, got {type(qa_list)}")

            # Enrich with source metadata
            result = []
            for qa in qa_list:
                if not isinstance(qa, dict):
                    continue
                question = qa.get("question", "").strip()
                answer = qa.get("answer", "").strip()
                if len(question) < 10 or len(answer) < 10:
                    continue
                result.append(
                    {
                        "question": question,
                        "answer": answer,
                        "evidence": qa.get("evidence", article_id),
                        "article_key": article.get("article_key", f"{doc_number}::{article_id}"),
                        "law": law,
                        "doc_number": doc_number,
                        "url": url,
                    }
                )
            return result

        except Exception as exc:
            logger.warning(
                "QA generation attempt {}/{} failed for '{}': {}",
                attempt,
                max_retries,
                article_id,
                exc,
            )
            if attempt < max_retries:
                time.sleep(2)

    return []


def process_file(
    input_path: str | Path,
    output_path: str | Path,
    max_articles: int | None = None,
    batch_delay: float = 0.5,
) -> int:
    """Generate QA pairs for all articles.

    Args:
        input_path: Path to articles.json.
        output_path: Path to write qa_raw.json.
        max_articles: Limit number of articles (for testing).
        batch_delay: Seconds to wait between API calls.

    Returns:
        Total number of QA pairs generated.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Streaming articles from {}", input_path)
    
    # Load existing output for checkpoint support
    existing: list[dict] = []
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                logger.warning("Existing output file is malformed, starting fresh")
        logger.info("Resuming: {} QA pairs already generated", len(existing))

    processed_keys: set[str] = set()
    for qa in existing:
        if "article_key" in qa:
            processed_keys.add(qa["article_key"])
        else:
            processed_keys.add(f"{qa.get('doc_number', '')}::{qa.get('evidence', '')}")

    client = OpenAIClient()
    all_qa = list(existing)
    
    # We use a count to track progress since we are streaming
    item_count = 0
    generated_count = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            item = json.loads(line)
            item_count += 1
            
            if max_articles and item_count > max_articles:
                logger.info("Reached limit of {} articles", max_articles)
                break
                
            # Map fields from hf_processed format to the internal article format
            article = {
                "law": item.get("title", ""),
                "article_id": item.get("doc_id", "Điều ?"),
                "content": item.get("content_markdown", "")[:2000],
                "doc_number": item.get("doc_id", ""),
                "url": item.get("source", "HuggingFace"),
                "article_key": str(item.get("doc_id", "")),
            }
            
            key = article["article_key"]
            if key in processed_keys:
                continue

            qa_pairs = generate_qa_for_article(client, article)
            all_qa.extend(qa_pairs)
            generated_count += len(qa_pairs)

            if item_count % 10 == 0 or qa_pairs:
                logger.info(
                    "  [{}] {} → +{} QA pairs (total: {})",
                    item_count,
                    article.get("article_id", "?"),
                    len(qa_pairs),
                    len(all_qa),
                )

            # Save checkpoint every 20 articles
            if item_count % 20 == 0:
                with open(output_path, "w", encoding="utf-8") as out_f:
                    json.dump(all_qa, out_f, ensure_ascii=False, indent=2)
                logger.info("  Checkpoint saved: {} QA pairs", len(all_qa))

            if batch_delay > 0:
                time.sleep(batch_delay)

    # Final save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_qa, f, ensure_ascii=False, indent=2)

    logger.info(
        "Step 3 done | articles={} | qa_pairs={} | saved to {}",
        item_count,
        len(all_qa),
        output_path,
    )
    return len(all_qa)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step 3: Generate QA pairs from legal articles")
    parser.add_argument("--input", default=str(ROOT / "data" / "processed" / "articles.json"))
    parser.add_argument("--output", default=str(ROOT / "data" / "processed" / "qa_raw.json"))
    parser.add_argument("--max-articles", type=int, default=None, help="Limit articles (for testing)")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls (seconds)")
    args = parser.parse_args()

    count = process_file(args.input, args.output, args.max_articles, args.delay)
    print(f"\n✅ Step 4 complete: {count} QA pairs generated → {args.output}")
