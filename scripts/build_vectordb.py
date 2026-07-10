import json
import logging
import sys
import time
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import tqdm
import unicodedata
import chromadb
from chromadb.utils import embedding_functions
from vector_store.safe_embedding import SafeEmbeddingFunction

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def get_embedding_function(config):
    import torch
    emb_config = config.get("embedding", {})
    model_name = emb_config.get("model_name", "microsoft/Harrier-OSS-v1-0.6B")
    max_length = emb_config.get("max_length", 512)
    
    # Target GPU for RTX 4060 Ti
    device = emb_config.get("device", "cuda")
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    
    logger.info(f"Loading embedding model: {model_name} on {device} (max_length={max_length})")
    return SafeEmbeddingFunction(
        model_name=model_name,
        device=device,
        max_seq_length=max_length
    )

def clear_memory():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def flush_batch(
    collection,
    emb_fn,
    documents,
    metadatas,
    ids,
    batch_id,
    start_index,
    total_indexed,
    started_at,
):
    end_index = start_index + len(documents) - 1
    logger.info(
        "Batch %s start | start_index=%s | end_index=%s | batch_size=%s | total_indexed=%s | elapsed=%.1fs",
        batch_id,
        start_index,
        end_index,
        len(documents),
        total_indexed,
        time.perf_counter() - started_at,
    )

    try:
        embeddings = emb_fn(documents)
    except Exception as e:
        logger.error(
            "Embedding encode failed | batch_id=%s | start_index=%s | end_index=%s | error=%s\n%s",
            batch_id,
            start_index,
            end_index,
            e,
            traceback.format_exc(),
        )
        raise

    try:
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
    except Exception as e:
        logger.error(
            "Chroma write failed | batch_id=%s | start_index=%s | end_index=%s | error=%s\n%s",
            batch_id,
            start_index,
            end_index,
            e,
            traceback.format_exc(),
        )
        raise

    total_indexed += len(documents)
    logger.info(
        "Batch %s done | start_index=%s | end_index=%s | batch_size=%s | total_indexed=%s | elapsed=%.1fs",
        batch_id,
        start_index,
        end_index,
        len(documents),
        total_indexed,
        time.perf_counter() - started_at,
    )
    clear_memory()
    return total_indexed


def index_jsonl(
    file_path: Path,
    collection,
    emb_fn,
    batch_size=500,
    limit=None,
    log_every=1000,
    initial_indexed=0,
    initial_batch_id=0,
    seen_ids=None,
):
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return initial_indexed, initial_batch_id

    documents = []
    metadatas = []
    ids = []
    source_prefix = file_path.stem
    total_indexed = initial_indexed
    batch_id = initial_batch_id
    started_at = time.perf_counter()
    next_log_at = None
    if seen_ids is None:
        seen_ids = set()
    if log_every:
        next_log_at = ((initial_indexed // log_every) + 1) * log_every
    
    # We use streaming to avoid OOM on large 6GB files
    with open(file_path, "r", encoding="utf-8") as f:
        # User requested to avoid double open for counting. 
        # We'll use tqdm without total or with manual estimation if needed.
        for line in tqdm.tqdm(f, desc=f"Indexing {file_path.name}", unit="line"):
            try:
                if limit is not None and total_indexed + len(documents) >= limit:
                    break

                line = line.strip()
                if not line: continue
                
                item = json.loads(line)
                doc_id = item.get("doc_id")
                content = item.get("content_markdown", "").strip()
                
                if doc_id is None:
                    continue
                
                if not content:
                    continue
                
                # Chunking: take first 8000 chars and normalize
                text_to_index = unicodedata.normalize("NFC", content[:8000])
                text_to_index = " ".join(text_to_index.split())  # remove control chars and extra whitespace
                
                documents.append(text_to_index)
                # Prefixed ID to avoid collisions, but skip prefix if it's already a canonical ID (contains ::)
                base_doc_id = str(doc_id) if "::" in str(doc_id) else f"{source_prefix}_{doc_id}"
                final_id = base_doc_id
                counter = 1
                while final_id in seen_ids:
                    counter += 1
                    final_id = f"{base_doc_id}_{counter}"
                seen_ids.add(final_id)
                ids.append(final_id)
                
                metadatas.append({
                    "title": str(item.get("title", "")),
                    "type": str(item.get("type", "")),
                    "source": str(item.get("source", "Unknown")),
                    "authority": str(item.get("authority", "") or "Unknown")
                })
                
                if len(documents) >= batch_size:
                    batch_id += 1
                    start_index = total_indexed + 1
                    total_indexed = flush_batch(
                        collection=collection,
                        emb_fn=emb_fn,
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids,
                        batch_id=batch_id,
                        start_index=start_index,
                        total_indexed=total_indexed,
                        started_at=started_at,
                    )
                    documents, metadatas, ids = [], [], []
                    while next_log_at is not None and total_indexed >= next_log_at:
                        logger.info(
                            "Progress | file=%s | total_indexed=%s | elapsed=%.1fs",
                            file_path.name,
                            total_indexed,
                            time.perf_counter() - started_at,
                        )
                        next_log_at += log_every
            except json.JSONDecodeError as e:
                logger.warning(
                    "Malformed JSON in %s | error=%s\n%s",
                    file_path.name,
                    e,
                    traceback.format_exc(),
                )
            except Exception as e:
                logger.error(
                    "Data parsing/indexing failed in %s | total_indexed=%s | pending_batch=%s | error=%s\n%s",
                    file_path.name,
                    total_indexed,
                    len(documents),
                    e,
                    traceback.format_exc(),
                )
                raise
        
        # Final batch
        if documents:
            batch_id += 1
            start_index = total_indexed + 1
            total_indexed = flush_batch(
                collection=collection,
                emb_fn=emb_fn,
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                batch_id=batch_id,
                start_index=start_index,
                total_indexed=total_indexed,
                started_at=started_at,
            )

    return total_indexed, batch_id

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--batch-size", type=int, default=None, help="Override embedding.batch_size from config.")
    parser.add_argument("--limit", type=int, default=None, help="Index only the first N valid documents.")
    parser.add_argument("--persist-dir", default=None, help="Override Chroma persist directory.")
    parser.add_argument("--collection-name", default=None, help="Override Chroma collection name.")
    parser.add_argument("--log-every", type=int, default=1000, help="Log progress every N indexed documents.")
    args = parser.parse_args()
    
    try:
        config_path = Path(args.config)
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
        is_en = "config_en" in args.config
        data_dir = config.get("data", {}).get("processed_dir", "data/processed")

        chroma_path = Path(args.persist_dir or config["chroma"]["path"])
        # Fix: ensure chroma_path itself (as a dir) is created
        chroma_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Opening ChromaDB at {chroma_path}...")
        client = chromadb.PersistentClient(path=str(chroma_path))
        
        collection_name = args.collection_name or config["chroma"]["collection_name"]
        batch_size = args.batch_size or config["embedding"].get("batch_size", 500)
        logger.info(
            "Build settings | persist_dir=%s | collection=%s | batch_size=%s | limit=%s | log_every=%s",
            chroma_path,
            collection_name,
            batch_size,
            args.limit,
            args.log_every,
        )
        
        try:
            client.delete_collection(collection_name)
            logger.warning(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass # Collection probably doesn't exist
        
        emb_fn = get_embedding_function(config)
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=emb_fn,
            metadata={"hnsw:space": "cosine"}
        )

        total_indexed = 0
        batch_id = 0
        global_seen_ids = set()
        if is_en:
            total_indexed, batch_id = index_jsonl(
                Path(data_dir) / "hotpot_full.jsonl",
                collection,
                emb_fn,
                batch_size=batch_size,
                limit=args.limit,
                log_every=args.log_every,
                initial_indexed=total_indexed,
                initial_batch_id=batch_id,
                seen_ids=global_seen_ids,
            )
        else:
            for jsonl_path in (
                Path(data_dir) / "core_laws_rechunked.jsonl",
                Path(data_dir) / "hf_rechunked.jsonl",
                Path(data_dir) / "phapdien_processed.jsonl",
            ):
                if args.limit is not None and total_indexed >= args.limit:
                    break
                total_indexed, batch_id = index_jsonl(
                    jsonl_path,
                    collection,
                    emb_fn,
                    batch_size=batch_size,
                    limit=args.limit,
                    log_every=args.log_every,
                    initial_indexed=total_indexed,
                    initial_batch_id=batch_id,
                    seen_ids=global_seen_ids,
                )

        logger.info(f"Indexing complete. Total docs: {collection.count():,}")
    except Exception as e:
        logger.error("Critical error: %s\n%s", e, traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
