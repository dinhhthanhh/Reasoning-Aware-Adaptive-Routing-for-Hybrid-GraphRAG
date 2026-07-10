"""Quick retrieval audit for phapdien_strict benchmark."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.metrics.id_normalizer import (
    compute_hit_at_k,
    normalize_gold_article,
    normalize_legal_id,
)
from vector_store.vector_retriever import VectorRetriever


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs/config.yaml").read_text(encoding="utf-8"))
    vr = VectorRetriever(cfg)
    test = json.loads((ROOT / "qa_pipeline/data/phapdien_strict/test.json").read_text(encoding="utf-8"))

    # First item debug
    item = test[0]
    print("Q:", item["question"])
    print("Gold canonical_id:", item["canonical_id"])
    results = vr.retrieve(item["question"], top_k=8)
    for i, r in enumerate(results):
        meta = getattr(r, "metadata", None) or {}
        cid = meta.get("canonical_id") or getattr(r, "id", "")
        norm = normalize_legal_id(str(cid))
        print(f"  [{i}] canonical={cid!r} norm_key={norm.key!r} law={meta.get('law_number')}")

    hits = {1: 0, 3: 0, 5: 0, 8: 0}
    canon_hits = {1: 0, 8: 0}
    n = 0
    for item in test[:100]:
        gold = item.get("relevant_articles") or []
        gold_norm = [normalize_gold_article(g) for g in gold]
        keys = [g.key for g in gold_norm if g.key]
        gold_cid = item.get("canonical_id", "")
        if not keys:
            continue
        n += 1
        results = vr.retrieve(item["question"], top_k=8)
        ret_keys = []
        ret_cids = []
        for r in results:
            meta = getattr(r, "metadata", None) or {}
            cid = meta.get("canonical_id") or getattr(r, "id", "")
            ret_keys.append(normalize_legal_id(str(cid)).key)
            ret_cids.append(str(cid))
        for k in hits:
            if compute_hit_at_k(ret_keys, keys, k):
                hits[k] += 1
        if gold_cid in ret_cids[:1]:
            canon_hits[1] += 1
        if gold_cid in ret_cids[:8]:
            canon_hits[8] += 1

    print(f"\nAudited n={n}")
    for k, v in hits.items():
        print(f"  norm Hit@{k}: {v/n:.3f}")
    for k, v in canon_hits.items():
        print(f"  exact canonical_id Hit@{k}: {v/n:.3f}")


if __name__ == "__main__":
    main()
