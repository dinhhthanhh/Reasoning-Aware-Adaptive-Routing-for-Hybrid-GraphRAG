"""Paraphrase the synthetic QA benchmark using the local LLM.

The original benchmark uses rigid templates which causes data leakage for the
XGBoost router. This script rewrites the questions to introduce natural language
diversity while preserving legal entities.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from llm.openai_client import OpenAIClient
from loguru import logger

DATA_DIR = ROOT / "qa_pipeline/data/phapdien_strict"

SYSTEM_PROMPT = """Bạn là một chuyên gia về luật pháp Việt Nam. Nhiệm vụ của bạn là viết lại câu hỏi pháp lý dưới đây sao cho tự nhiên, đa dạng, và giống cách người dân bình thường hoặc luật sư thường hỏi.
QUY TẮC BẮT BUỘC:
1. KHÔNG được trả lời câu hỏi. Chỉ in ra DUY NHẤT câu hỏi được viết lại.
2. PHẢI giữ nguyên chính xác CÁC SỐ HIỆU ĐIỀU LUẬT, KHOẢN, MỤC, VÀ TÊN VĂN BẢN QUY PHẠM PHÁP LUẬT nếu có trong câu gốc.
3. Thay đổi cách hành văn, cấu trúc câu sao cho đa dạng, tránh khuôn mẫu.
4. KHÔNG giải thích thêm, KHÔNG dùng markdown bọc câu hỏi."""

def rewrite_dataset(client: OpenAIClient, file_path: Path):
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return

    logger.info(f"Loading {file_path.name}...")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Paraphrasing {len(data)} samples in {file_path.name}...")
    for idx, sample in enumerate(tqdm(data)):
        original_q = sample["question"]
        try:
            prompt = f"Câu hỏi gốc: {original_q}\nCâu hỏi viết lại:"
            response = client.generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.7,
                max_tokens=200
            )
            response = client._strip_thinking(response)
            # Cleanup common prefixes LLM might output
            if response.lower().startswith("câu hỏi viết lại:"):
                response = response[len("câu hỏi viết lại:"):].strip()
            if response.startswith('"') and response.endswith('"'):
                response = response[1:-1]
            
            sample["question"] = response.strip()
        except Exception as e:
            logger.error(f"Error generating for sample {idx}: {e}")
            # Keep original question on failure
    
    # Backup original
    backup_path = file_path.with_name(f"{file_path.stem}_original.json")
    if not backup_path.exists():
        import shutil
        shutil.copy2(file_path, backup_path)
        logger.info(f"Backed up original to {backup_path.name}")

    # Save
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {file_path.name}")


def main():
    logger.info("Initializing OpenAI client...")
    client = OpenAIClient()
    if not client.health_check():
        logger.error("LLM Server is not healthy. Exiting.")
        return

    # Rewrite all splits
    for split in ["train.json", "dev.json", "test.json"]:
        rewrite_dataset(client, DATA_DIR / split)
    
    logger.info("All datasets paraphrased successfully.")

if __name__ == "__main__":
    main()
