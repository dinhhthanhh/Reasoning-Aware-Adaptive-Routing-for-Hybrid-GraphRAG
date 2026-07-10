"""Index Pháp Điển canonical chunks into ChromaDB (PD-only).

Collection ``phapdien_full`` at ``data/vector_store/chroma_full`` — only Pháp Điển
articles from ``pd_rechunked.jsonl``. Legacy collections (e.g. ``chroma_full_v2``,
``legal_docs``) are left untouched but not used by the active pipeline.

Usage:
    python scripts/phapdien_pipeline/build_chroma.py
    python scripts/phapdien_pipeline/build_chroma.py --config configs/build_kg_no_ner.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

import chromadb
import numpy as np
import tqdm
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from vector_store.safe_embedding import SafeEmbeddingFunction

PD_RECHUNKED = ROOT / "data/processed/pd_rechunked.jsonl"


def get_embedding_function(config: dict) -> SafeEmbeddingFunction:
    import torch

    emb = config.get("embedding", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SafeEmbeddingFunction(
        model_name=emb.get("model_name", "microsoft/Harrier-OSS-v1-0.6B"),
        device=device,
        max_seq_length=emb.get("max_length", 512),
    )


def flush_batch(collection, emb_fn, documents, metadatas, ids):
    embeddings = emb_fn(documents)
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PD_RECHUNKED)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/build_kg_no_ner.yaml")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    chroma_path = ROOT / config["chroma"]["path"]
    collection_name = config["chroma"]["collection_name"]
    batch_size = int(config.get("embedding", {}).get("batch_size", 96))

    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    for coll in client.list_collections():
        if coll.name == collection_name:
            client.delete_collection(collection_name)
            print(f"Deleted existing collection {collection_name}")

    emb_fn = get_embedding_function(config)
    collection = client.create_collection(
        name=collection_name,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"},
    )

    documents, metadatas, ids = [], [], []
    seen: set[str] = set()
    lengths: list[int] = []
    total = 0
    dups = 0
    t0 = time.perf_counter()

    with open(args.input, encoding="utf-8") as f:
        for line in tqdm.tqdm(f, desc="Indexing Pháp Điển"):
            if not line.strip():
                continue
            item = json.loads(line)
            if not item.get("has_canonical_id"):
                continue
            content = (item.get("content") or "").strip()
            if len(content) < 50:
                continue

            cid = item["canonical_id"]
            final_id = cid
            if final_id in seen:
                dups += 1
                n = 2
                while f"{cid}_{n}" in seen:
                    n += 1
                final_id = f"{cid}_{n}"
            seen.add(final_id)

            text = unicodedata.normalize("NFC", content[:8000])
            text = " ".join(text.split())
            lengths.append(len(text))

            documents.append(text)
            ids.append(final_id)
            metadatas.append(
                {
                    "canonical_id": cid,
                    "law_number": str(item.get("law_number", "")),
                    "article_number": str(item.get("article_number", "")),
                    "source": "phapdien",
                    "has_canonical_id": "true",
                    "title": str(item.get("title", ""))[:500],
                }
            )

            if len(documents) >= batch_size:
                flush_batch(collection, emb_fn, documents, metadatas, ids)
                total += len(documents)
                documents, metadatas, ids = [], [], []

    if documents:
        flush_batch(collection, emb_fn, documents, metadatas, ids)
        total += len(documents)

    arr = np.array(lengths) if lengths else np.array([0])
    print(f"Indexed {collection.count()} vectors in {time.perf_counter()-t0:.0f}s")
    print(f"Duplicates resolved: {dups}")
    print(
        f"Chunk lengths — mean={arr.mean():.0f} p50={np.percentile(arr,50):.0f} "
        f"p90={np.percentile(arr,90):.0f} max={arr.max():.0f}"
    )


if __name__ == "__main__":
    main()
