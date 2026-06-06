import requests
import json
import logging
import time
import io
import os
import tempfile
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

HF_API_BASE = "https://datasets-server.huggingface.co"
DATASET_ID  = "th1nhng0/vietnamese-legal-documents"
ALL_CONFIGS = ["relationships", "metadata", "content"]


def fetch_parquet_files(config: str) -> list[dict]:
    url = f"{HF_API_BASE}/parquet?dataset={DATASET_ID}&config={config}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json().get("parquet_files", [])


def _crawl_config_parquet(
    config: str,
    output_dir: Path,
    max_rows: Optional[int] = None,
    delay: float = 0.5,
    batch_size: int = 5000,
) -> dict:
    """
    Streaming Parquet to JSONL with disk-based temporary storage.
    Returns: {"count": int, "sample": list[dict]}
    """
    config_dir = output_dir / config
    config_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{config}] Fetching Parquet file list...")
    try:
        files = fetch_parquet_files(config)
    except Exception as e:
        logger.error(f"[{config}] Could not fetch parquet files: {e}")
        return {"count": 0, "sample": []}

    logger.info(f"[{config}] Found {len(files)} parquet file(s). Downloading...")

    total_fetched = 0
    sample_records = []
    
    # Final merged file
    merged_file = config_dir / f"{config}_all.jsonl"
    
    # Ensure current data is cleared or we append
    if merged_file.exists():
        merged_file.unlink()

    with open(merged_file, "a", encoding="utf-8") as out_f:
        for i, f in enumerate(files):
            split = f.get("split", "unknown")
            url   = f.get("url", "")
            hf_name = url.split("/")[-1].replace(".parquet", "")
            logger.info(f"[{config}/{split}] Streaming File {i+1}/{len(files)}: {hf_name}")

            tmp_path = None
            try:
                # Use a temporary file to save RAM
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
                tmp_path = tmp.name
                try:
                    with requests.get(url, stream=True, timeout=300) as r:
                        r.raise_for_status()
                        for chunk in r.iter_content(chunk_size=8192):
                            tmp.write(chunk)
                finally:
                    tmp.close() # CRITICAL: Close for Windows 
                
                # Stream batches from disk to JSONL
                pf = pq.ParquetFile(tmp_path)
                for batch in pf.iter_batches(batch_size=batch_size):
                    df_batch = batch.to_pandas()
                    
                    # Track sample for first batch
                    if len(sample_records) < 5:
                        sample_records.extend(df_batch.head(5).to_dict(orient="records"))
                    
                    current_len = len(df_batch)
                    if max_rows and (total_fetched + current_len) > max_rows:
                        # Slice the last batch to fit max_rows
                        remaining = max_rows - total_fetched
                        df_batch = df_batch.head(remaining)
                        current_len = len(df_batch)

                    if current_len > 0:
                        # Write to merged file
                        df_batch.to_json(out_f, orient="records", lines=True, force_ascii=False)
                        out_f.write("\n")
                    
                    total_fetched += current_len
                    if max_rows and total_fetched >= max_rows:
                        break

            except Exception as e:
                logger.error(f"[{config}] Error on file {hf_name}: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except:
                        pass

            if max_rows and total_fetched >= max_rows:
                logger.info(f"[{config}] Reached max_rows={max_rows}, stopping.")
                break

            if i < len(files) - 1:
                time.sleep(delay)

    logger.info(f"[{config}] {total_fetched:,} rows saved → {merged_file}")
    return {"count": total_fetched, "sample": sample_records[:5]}


def crawl_huggingface(
    output_dir: Path,
    configs: Optional[list[str]] = None,
    max_rows: Optional[int] = None,
    delay: float = 0.5,
) -> dict[str, dict]:
    """
    Returns: {config: {"count": int, "sample": list[dict]}}
    """
    if configs is None:
        configs = ALL_CONFIGS

    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    for config in configs:
        logger.info("=" * 60)
        logger.info(f"HuggingFace Config: [{config}]")
        logger.info("=" * 60)
        results[config] = _crawl_config_parquet(config, output_dir, max_rows, delay)

    # Combined summary file
    combined_summary = {
        "dataset": DATASET_ID,
        "configs": {c: {"count": res["count"]} for c, res in results.items()},
        "total_rows": sum(res["count"] for res in results.values())
    }
    
    summary_file = output_dir / "hf_info.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(combined_summary, f, ensure_ascii=False, indent=2)

    logger.info(f"HuggingFace crawl complete. Total: {combined_summary['total_rows']:,} rows.")
    return results
