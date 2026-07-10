"""
build_benchmark_v2.py
=====================
Augments the existing test.json benchmark with a `concise_answer` field.

ROOT CAUSE OF LOW F1:
    answer == gold_context (100% of records).
    gold_context = full legal article text (~431 words average, max 1870 words).
    When your system generates a concise answer (~100 words), token F1
    collapses because Recall = 100_matching / 431_total_gold ≈ 0.23.

SOLUTION:
    Extract a `concise_answer` from gold_context using Gemini API.
    The prompt is EXTRACTIVE: the model picks 1-4 relevant sentences/clauses
    from the article. It does NOT paraphrase or add external knowledge.

OUTPUT:
    test_benchmark_v2.json — original fields + concise_answer, concise_words
"""

import json
import argparse
import time
import re
import sys
import os
from pathlib import Path
from openai import OpenAI
import yaml

SYSTEM_PROMPT = """Bạn là chuyên gia pháp luật Việt Nam có nhiệm vụ trích xuất câu trả lời ngắn gọn và chính xác từ văn bản pháp luật.

Nguyên tắc bắt buộc:
1. Chỉ được dùng các câu, cụm từ, mệnh đề có trong văn bản pháp luật được cung cấp. KHÔNG được thêm thông tin bên ngoài.
2. Câu trả lời phải trực tiếp trả lời câu hỏi, không lan man.
3. Độ dài: 1–4 câu hoặc 1–3 điểm gạch đầu dòng (nếu câu hỏi yêu cầu liệt kê).
4. Với câu hỏi Yes/No: bắt đầu bằng "Có" hoặc "Không", sau đó trích câu luật liên quan.
5. KHÔNG trích dẫn toàn bộ điều luật. Chỉ lấy phần liên quan trực tiếp đến câu hỏi.
6. KHÔNG giải thích, KHÔNG thêm "theo Điều X" trừ khi cần để phân biệt nguồn.
7. Nếu văn bản không đủ thông tin để trả lời, ghi: "[KHÔNG ĐỦ THÔNG TIN]"

Định dạng đầu ra: Chỉ viết câu trả lời, không có tiêu đề, không có giải thích."""

USER_TEMPLATE = """Câu hỏi: {question}

[VĂN BẢN PHÁP LUẬT]:
{gold_context}

Câu trả lời ngắn gọn (chỉ dùng từ ngữ trong văn bản trên):"""

def build_prompt(record: dict) -> str:
    ctx = record.get("gold_context", record.get("evidence", ""))
    if len(ctx) > 3000:
        ctx = ctx[:3000] + "\n[...văn bản còn tiếp...]"
    return USER_TEMPLATE.format(
        question=record["question"],
        gold_context=ctx,
    )

def extract_concise_answer(
    client: OpenAI,
    model: str,
    record: dict,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    user_msg = build_prompt(record)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()

def post_process(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"^(Câu trả lời|Trả lời)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()

def word_count(text: str) -> int:
    return len(text.split()) if text else 0

def run(args):
    input_path = Path(args.input)
    assert input_path.exists(), f"Input file not found: {input_path}"
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.dry_run:
        data = data[: args.n]
        print(f"[DRY RUN] Processing first {len(data)} records only.")

    output_path = Path(args.output)
    processed_ids: set[str] = set()
    results: list[dict] = []

    if output_path.exists() and not args.dry_run:
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        processed_ids = {r["id"] for r in results if "concise_answer" in r}
        print(f"[RESUME] Found {len(processed_ids)} already processed records.")

    # Tự động lấy config từ config.yaml
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    api_key = os.environ.get("OPENAI_API_KEY") or config.get("openai", {}).get("api_key")
    base_url = config.get("openai", {}).get("base_url")
    model_name = args.model or config.get("openai", {}).get("model", "gemini-3.1-flash-lite")

    if not api_key:
        raise ValueError("API Key is missing. Set it in config.yaml or OPENAI_API_KEY env var.")

    client = OpenAI(base_url=base_url, api_key=api_key)
    
    print(f"Using Model: {model_name} on {base_url}")

    total = len(data)
    errors = 0

    for i, record in enumerate(data):
        record_id = record.get("id", f"idx_{i}")

        if record_id in processed_ids:
            continue

        print(f"[{i+1}/{total}] {record_id} | route={record.get('routing_label', 'N/A')}")

        try:
            raw_answer = extract_concise_answer(
                client=client,
                model=model_name,
                record=record,
                max_tokens=args.max_tokens,
                temperature=0.0,
            )
            concise = post_process(raw_answer)

            wc = word_count(concise)
            gold_wc = word_count(record.get("gold_context", record.get("evidence", "")))
            ratio = wc / gold_wc if gold_wc > 0 else 0

            if ratio > 0.7:
                print(f"  ⚠ WARNING: concise_answer is {wc} words ({ratio:.0%} of gold). May not be extractive.")

            augmented = {**record, "concise_answer": concise, "concise_words": wc}

            if args.dry_run:
                print(f"  Q: {record['question']}")
                print(f"  Gold ({gold_wc}w): {record.get('gold_context', record.get('evidence', ''))[:200]}...")
                print(f"  Concise ({wc}w): {concise}")
                print()
            else:
                results.append(augmented)
                if (i + 1) % args.batch_size == 0:
                    _save(results, output_path)
                    print(f"  [SAVED] {len(results)} records → {output_path}")

        except Exception as e:
            errors += 1
            print(f"  ✗ ERROR for {record_id}: {e}")
            if not args.dry_run:
                results.append({**record, "concise_answer": None, "concise_words": None})

        if args.sleep > 0:
            time.sleep(args.sleep)

    if not args.dry_run:
        _save(results, output_path)

    if not args.dry_run:
        valid = [r for r in results if r.get("concise_answer")]
        concise_words = [r["concise_words"] for r in valid if r["concise_words"]]
        gold_words = [word_count(r.get("gold_context", r.get("evidence", ""))) for r in valid]

        print("\n" + "=" * 60)
        print(f"DONE. Total: {len(results)} | Errors: {errors}")
        if concise_words:
            import statistics
            print(f"concise_answer words: mean={statistics.mean(concise_words):.0f}, "
                  f"median={statistics.median(concise_words):.0f}, "
                  f"max={max(concise_words)}")
            if gold_words:
                print(f"gold_context words:   mean={statistics.mean(gold_words):.0f}, "
                      f"median={statistics.median(gold_words):.0f}")
                reduction = 1 - statistics.mean(concise_words) / statistics.mean(gold_words)
                print(f"Length reduction:     {reduction:.0%}")
        print(f"Output: {output_path}")

def _save(data: list, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="qa_pipeline/data/legal_strict/test.json")
    p.add_argument("--output", default="qa_pipeline/data/legal_strict/test_benchmark_v2.json")
    p.add_argument("--model", default=None, help="Model name as served by API")
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--sleep", type=float, default=0.2)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--n", type=int, default=10)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args)
