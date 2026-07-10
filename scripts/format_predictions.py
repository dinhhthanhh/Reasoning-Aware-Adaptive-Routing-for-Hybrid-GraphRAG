import json
import argparse
import re
from pathlib import Path


def chunk_id_to_article_id(chunk_id: str, sep: str = "::") -> str:
    """
    Convert a chunk_id like "77/2026/NĐ-CP::Điều 18::chunk_0"
    to an article_id like "77/2026/NĐ-CP::Điều 18".
    """
    chunk_id = re.sub(r"[_:\-]+chunk[_\-]?\d+$", "", chunk_id, flags=re.IGNORECASE)
    chunk_id = re.sub(r"[_:\-]+\d+$", "", chunk_id)

    parts = chunk_id.split(sep)
    if len(parts) >= 2:
        return sep.join(parts[:2])

    parts = chunk_id.split("/", 1)
    if len(parts) == 2:
        return f"{parts[0]}::{parts[1]}"

    return chunk_id

import re

def extract_standard_ids(text: str) -> list[str]:
    """
    Extract law_id (e.g. 10/2015/TT-BKHCN) and article_id (e.g. Điều 15) from text
    and format them as law_id::article_id.
    """
    if not text:
        return []
    
    # 1. Extract law numbers
    law_pattern = re.compile(r"(\d+\s*/\s*\d{4}\s*/\s*[A-Za-zĐđÀ-ỹ][A-Za-zĐđÀ-ỹ\-]*)")
    laws = law_pattern.findall(text)
    laws = [re.sub(r"\s+", "", l) for l in laws] # Normalize spacing
    
    # 2. Extract article names
    art_pattern = re.compile(r"((?:Điều|Khoản|Điểm)\s+\d+[\w\.]*)", re.IGNORECASE)
    arts = art_pattern.findall(text)
    
    # Simple pairing: if we find laws and arts, pair them up. 
    # If there is only 1 law, pair it with all arts.
    results = set()
    if laws and arts:
        # If multiple laws, pair art with the nearest law or just pair all to all to maximize Hit@K
        for law in laws:
            for art in arts:
                art = " ".join(art.split()) # Normalize spacing
                art = art.capitalize()
                results.add(f"{law}::{art}")
    
    return list(results)

def format_predictions(
    system_output: list[dict],
    id_field: str = "query_id",
    route_field: str = "final_route",
    answer_field: str = "answer",
    chunk_id_field: str = "chunk_id",
    chunk_sep: str = "::",
) -> list[dict]:
    predictions = []

    for item in system_output:
        record_id = item.get(id_field) or item.get("id") or item.get("query_id", "")
        predicted_route = (
            item.get(route_field)
            or item.get("final_route")
            or item.get("predicted_route")
            or item.get("route")
            or item.get("Actual_Route") # support run_benchmark_eval.py output
            or "dense_retrieval"
        )
        
        # In run_benchmark_eval.py output, Actual_Route can be hybrid_reasoning:neo4j, so we trim it
        predicted_route = predicted_route.split(":")[0]

        retrieved_articles = []
        seen = set()

        for chunk_field in ["retrieved_chunks", "retrieved_docs", "retrieved", "chunks", "Sources"]:
            chunks = item.get(chunk_field, [])
            if isinstance(chunks, str):
                chunks = chunks.split(";")
            if chunks:
                for chunk in chunks:
                    if isinstance(chunk, dict):
                        cid = chunk.get(chunk_id_field) or chunk.get("id") or chunk.get("chunk_id", "")
                    elif isinstance(chunk, str):
                        cid = chunk
                    else:
                        continue
                    if not cid: continue
                    art_id = chunk_id_to_article_id(cid, sep=chunk_sep)
                    if art_id and art_id not in seen:
                        seen.add(art_id)
                        retrieved_articles.append(art_id)
                break

        generated_answer = (
            item.get(answer_field)
            or item.get("generated_answer")
            or item.get("answer")
            or item.get("Generated_Answer")
            or ""
        )

        extracted_ids = extract_standard_ids(";".join([str(c) for c in retrieved_articles]) + " " + generated_answer)
        for eid in extracted_ids:
            if eid not in seen:
                seen.add(eid)
                retrieved_articles.append(eid)

        predictions.append({
            "id": record_id,
            "predicted_route": predicted_route,
            "retrieved_articles": retrieved_articles,
            "generated_answer": generated_answer,
        })

    return predictions


def run(args):
    if args.system_output.endswith('.csv'):
        import csv
        with open(args.system_output, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            system_output = list(reader)
    else:
        with open(args.system_output, "r", encoding="utf-8") as f:
            system_output = json.load(f)

    benchmark = None
    if args.benchmark:
        with open(args.benchmark, "r", encoding="utf-8") as f:
            benchmark = json.load(f)

    predictions = format_predictions(
        system_output=system_output,
        id_field=args.id_field,
        route_field=args.route_field,
        answer_field=args.answer_field,
        chunk_id_field=args.chunk_id_field,
        chunk_sep=args.chunk_sep,
    )

    if benchmark:
        bench_ids = {r["id"]: i for i, r in enumerate(benchmark)}
        pred_ids = [p["id"] for p in predictions]
        missing = [pid for pid in pred_ids if pid not in bench_ids]
        if missing:
            print(f"⚠ {len(missing)} prediction IDs not found in benchmark:")
            for mid in missing[:5]:
                print(f"  {mid}")

        pred_map = {p["id"]: p for p in predictions}
        ordered = []
        for r in benchmark:
            if r["id"] in pred_map:
                ordered.append(pred_map[r["id"]])
            else:
                ordered.append({
                    "id": r["id"],
                    "predicted_route": "MISSING",
                    "retrieved_articles": [],
                    "generated_answer": "",
                })
                print(f"  ⚠ Missing prediction for: {r['id']}")

        predictions = ordered
        print(f"Aligned {len(predictions)} predictions to benchmark order.")

    from collections import Counter
    route_dist = Counter(p["predicted_route"] for p in predictions)
    print(f"Predictions: {len(predictions)}")
    print("Predicted route distribution:")
    for route, count in sorted(route_dist.items(), key=lambda x: -x[1]):
        print(f"  {route}: {count} ({count/len(predictions)*100:.1f}%)")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"Saved to: {args.output}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--system_output", required=True)
    p.add_argument("--benchmark", default=None)
    p.add_argument("--output", default="predictions.json")
    p.add_argument("--id_field", default="query_id")
    p.add_argument("--route_field", default="final_route")
    p.add_argument("--answer_field", default="answer")
    p.add_argument("--chunk_id_field", default="chunk_id")
    p.add_argument("--chunk_sep", default="::")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args)
