"""Run benchmark evaluation and generate CSV/JSON/Markdown reports.

Processes evaluation queries, measures pipeline performance (latency, routing steps),
calculates EM/F1/Accuracy against ground truth, and compares:
  - pure_vector: VectorRAG only
  - pure_graph: GraphRAG only
  - single_stage_router: HybridPipeline with Stage 2 disabled
  - two_stage_hybrid: full Reasoning-Aware Adaptive Router
"""

import json
import logging
import csv
import copy
from pathlib import Path
import argparse
import sys
import tqdm
import gc
import time
from dataclasses import dataclass
from typing import Any
try:
    import torch
except ImportError:
    torch = None


# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.hybrid_pipeline import HybridPipeline
from rag.vector_rag import VectorRAG
from rag.graph_rag_adapter import GraphRAGAdapter
try:
    from evaluation.metrics import evaluate_prediction
except ImportError:
    def evaluate_prediction(ground_truth: str, answer: str) -> tuple[float, float, float]:
        from evaluation.metrics import compute_answer_f1
        keywords = [k.strip() for k in ground_truth.split() if k.strip()]
        f1 = compute_answer_f1(answer, keywords)
        ans_lower = answer.lower()
        gt_lower = ground_truth.lower()
        em = 1.0 if gt_lower == ans_lower else 0.0
        acc = 1.0 if gt_lower in ans_lower else 0.0
        return em, f1, acc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEMS = ("pure_vector", "pure_graph", "single_stage_router", "two_stage_hybrid")


@dataclass
class EvalResponse:
    answer: str
    route: str
    actual_route: str
    steps: int
    latency_ms: float
    stage2_invoked: bool = False
    stage2_override: bool = False
    sources: list[str] | None = None
    context: str = ""
    kg_source: str = ""


class BenchmarkSystem:
    """Small adapter layer so every baseline exposes the same evaluate() API."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name = name
        self.config = copy.deepcopy(config)
        self._instance: Any | None = None

    def _load(self) -> Any:
        if self._instance is not None:
            return self._instance

        if self.name == "pure_vector":
            vector = VectorRAG(self.config)
            vector.load_index()
            self._instance = vector
        elif self.name == "pure_graph":
            self._instance = GraphRAGAdapter(self.config)
        elif self.name == "single_stage_router":
            cfg = copy.deepcopy(self.config)
            cfg.setdefault("router", {}).setdefault("stage2", {})["enabled"] = False
            self._instance = HybridPipeline.from_config(cfg)
        elif self.name == "two_stage_hybrid":
            self._instance = HybridPipeline.from_config(self.config)
        else:
            raise ValueError(f"Unknown benchmark system: {self.name}")

        return self._instance

    def evaluate(self, query: str, qid: str) -> EvalResponse:
        start = time.perf_counter()
        instance = self._load()

        if self.name == "pure_vector":
            result = instance.answer(query, history=None)
            return EvalResponse(
                answer=result.answer,
                route="dense_retrieval",
                actual_route="dense_retrieval",
                steps=1,
                latency_ms=result.latency_ms,
                stage2_invoked=False,
                stage2_override=False,
                sources=result.sources,
                context=result.context,
            )

        if self.name == "pure_graph":
            result = instance.answer(query, history=None)
            return EvalResponse(
                answer=result.get("answer", ""),
                route="graph_traversal",
                actual_route="graph_traversal",
                steps=2,
                latency_ms=result.get("latency_ms", (time.perf_counter() - start) * 1000),
                stage2_invoked=False,
                stage2_override=False,
                sources=result.get("sources", []),
                context=result.get("context", ""),
                kg_source=result.get("kg_source", ""),
            )

        response = instance.query(query, session_id=f"{self.name}_{qid}", verbose=False)
        steps = _steps_for_route(response.route_used)
        return EvalResponse(
            answer=response.answer,
            route=response.route_used,
            actual_route=response.actual_pipeline_used,
            steps=steps,
            latency_ms=response.latency_ms,
            stage2_invoked=response.stage2_invoked,
            stage2_override=response.stage2_override,
            sources=response.sources,
            context=response.context,
            kg_source=response.kg_source,
        )


def _steps_for_route(route: str) -> int:
    if route == "dense_retrieval":
        return 1
    if route == "graph_traversal":
        return 2
    if route == "hybrid_reasoning":
        return 3
    if route == "clarify":
        return 1
    return 0


def _normalize_eval_item(item: dict[str, Any], idx: int) -> dict[str, Any]:
    """Normalize legal QA, ViMQA, and Hotpot-style records to one schema."""
    qid = item.get("id") or item.get("_id") or item.get("qid") or f"item_{idx}"
    query = item.get("query") or item.get("question") or ""
    ground_truth = item.get("ground_truth") or item.get("answer") or item.get("reference_answer") or ""
    expected_route = item.get("expected_route") or item.get("routing_label") or item.get("route")
    return {
        "id": str(qid),
        "query": str(query),
        "ground_truth": str(ground_truth),
        "expected_route": expected_route,
        "hop_count": item.get("hop_count"),
        "is_cross_doc": item.get("is_cross_doc"),
    }


def _load_eval_items(eval_file: Path, limit: int | None) -> list[dict[str, Any]]:
    if not eval_file.exists():
        logger.error(f"Evaluation file not found: {eval_file}")
        return []

    items: list[dict[str, Any]] = []
    if eval_file.suffix.lower() == ".json":
        with open(eval_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("data", data.get("items", []))
        for idx, item in enumerate(data):
            if limit and len(items) >= limit:
                break
            items.append(_normalize_eval_item(item, idx))
    else:
        with open(eval_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if limit and len(items) >= limit:
                    break
                line = line.strip()
                if line:
                    items.append(_normalize_eval_item(json.loads(line), idx))
    return items


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "total_queries": 0,
            "exact_match": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
            "avg_latency_ms": 0.0,
            "route_distribution": {},
            "routing_accuracy": None,
            "stage2_trigger_rate": 0.0,
            "stage2_override_rate": 0.0,
            "kg_source_distribution": {},
        }

    route_counts: dict[str, int] = {}
    route_correct = 0
    route_total = 0
    stage2_total = 0
    stage2_override_total = 0
    kg_source_counts: dict[str, int] = {}
    for row in rows:
        route = row["Route"]
        route_counts[route] = route_counts.get(route, 0) + 1
        expected_route = row.get("Expected_Route", "")
        if expected_route:
            route_total += 1
            if route == expected_route:
                route_correct += 1
        if row.get("Stage2"):
            stage2_total += 1
        if row.get("Stage2_Override"):
            stage2_override_total += 1
        kg_source = row.get("KG_Source") or ""
        if kg_source:
            kg_source_counts[kg_source] = kg_source_counts.get(kg_source, 0) + 1

    total = len(rows)
    return {
        "total_queries": total,
        "exact_match": sum(row["EM"] for row in rows) / total,
        "f1": sum(row["F1"] for row in rows) / total,
        "accuracy": sum(row["Acc"] for row in rows) / total,
        "avg_latency_ms": sum(row["Time_ms"] for row in rows) / total,
        "route_distribution": route_counts,
        "routing_accuracy": route_correct / route_total if route_total else None,
        "stage2_trigger_rate": stage2_total / total,
        "stage2_override_rate": stage2_override_total / stage2_total if stage2_total else 0.0,
        "kg_source_distribution": kg_source_counts,
    }


def _write_summary(summary_path: Path, md_path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    lines = [
        "# Full Evaluation Summary",
        "",
        "| System | Queries | EM | F1 | Answer Acc | Routing Acc | Stage2 | Stage2 Override | Avg Latency (ms) | KG Source | Route Distribution |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for system, metrics in summaries.items():
        route_dist = ", ".join(
            f"{route}: {count}" for route, count in sorted(metrics["route_distribution"].items())
        )
        kg_dist = ", ".join(
            f"{source}: {count}" for source, count in sorted(metrics.get("kg_source_distribution", {}).items())
        ) or "n/a"
        routing_acc = (
            f"{metrics['routing_accuracy']:.4f}"
            if metrics["routing_accuracy"] is not None
            else "n/a"
        )
        lines.append(
            f"| {system} | {metrics['total_queries']} | "
            f"{metrics['exact_match']:.4f} | {metrics['f1']:.4f} | "
            f"{metrics['accuracy']:.4f} | {routing_acc} | "
            f"{metrics['stage2_trigger_rate']:.4f} | "
            f"{metrics['stage2_override_rate']:.4f} | "
            f"{metrics['avg_latency_ms']:.2f} | {kg_dist} | {route_dist} |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def evaluate_dataset(
    config_path: str,
    eval_file: Path,
    output_dir: Path,
    dataset: str,
    systems: list[str],
    limit: int | None = None,
    eval_answer_style: bool = False,
    router_model_path: str | None = None,
) -> dict[str, dict[str, Any]]:
    items = _load_eval_items(eval_file, limit)
    if not items:
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        import yaml
        config = yaml.safe_load(f)
    if router_model_path:
        config.setdefault("router", {}).setdefault("stage1", {})["model_path"] = router_model_path
        logger.info("Router model path overridden for benchmark: {}", router_model_path)

    if eval_answer_style:
        config["concise_answer"] = True
        if config.get("task_type") == "legal_citation":
            config["task_type"] = "legal_eval"
        config.setdefault("openai", {})["max_tokens"] = min(
            int(config.get("openai", {}).get("max_tokens", 1024)),
            384,
        )
        for store_key in ("faiss", "chroma"):
            if store_key in config:
                config[store_key]["top_k"] = max(int(config[store_key].get("top_k", 8)), 12)
        config.setdefault("rag", {})["max_context_chars"] = max(
            int(config.get("rag", {}).get("max_context_chars", 12000)),
            18000,
        )
        config["rag"]["max_chunk_chars"] = max(
            int(config["rag"].get("max_chunk_chars", 1500)),
            2200,
        )
        config["rag"]["hybrid_vector_top_k"] = max(
            int(config["rag"].get("hybrid_vector_top_k", 5)),
            5,
        )
        config["rag"]["hybrid_vector_candidate_k"] = max(
            int(config["rag"].get("hybrid_vector_candidate_k", 15)),
            15,
        )
        config["rag"]["retrieval_candidate_multiplier"] = max(
            int(config["rag"].get("retrieval_candidate_multiplier", 3)),
            3,
        )
        config["rag"]["hybrid_vector_chunk_chars"] = max(
            int(config["rag"].get("hybrid_vector_chunk_chars", 1000)),
            1200,
        )
        config["rag"]["hybrid_graph_top_k"] = max(
            int(config["rag"].get("hybrid_graph_top_k", 3)),
            4,
        )
        config["rag"]["hybrid_graph_context_chars"] = max(
            int(config["rag"].get("hybrid_graph_context_chars", 2500)),
            3500,
        )
        logger.info("Eval answer style enabled: concise legal answers, top_k>=12, wider context.")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting evaluation of {len(items)} queries across {len(systems)} systems...")

    all_summaries: dict[str, dict[str, Any]] = {}

    for system_name in systems:
        logger.info(f"=== Evaluating system: {system_name} ===")
        system = BenchmarkSystem(system_name, config)
        output_csv = output_dir / f"{dataset}_{system_name}_results.csv"

        rows: list[dict[str, Any]] = []
        with open(output_csv, "w", encoding="utf-8", newline="") as f_csv:
            writer = csv.DictWriter(
                f_csv,
                fieldnames=[
                    "ID",
                    "System",
                    "Query",
                    "Ground_Truth",
                    "Generated_Answer",
                    "Expected_Route",
                    "Route",
                    "Actual_Route",
                    "KG_Source",
                    "Sources",
                    "Context_Chars",
                    "Context_Preview",
                    "Steps",
                    "Stage2",
                    "Stage2_Override",
                    "Time_ms",
                    "EM",
                    "F1",
                    "Acc",
                ],
            )
            writer.writeheader()

            for item in tqdm.tqdm(items, desc=f"Evaluating {system_name}"):
                qid = item["id"]
                query = item["query"]
                ground_truth = item["ground_truth"]
                expected_route = item.get("expected_route") or ""

                try:
                    response = system.evaluate(query, qid)
                    em, f1, acc = evaluate_prediction(ground_truth, response.answer)

                    row = {
                        "ID": qid,
                        "System": system_name,
                        "Query": query,
                        "Ground_Truth": ground_truth,
                        "Generated_Answer": response.answer,
                        "Expected_Route": expected_route,
                        "Route": response.route,
                        "Actual_Route": response.actual_route,
                        "KG_Source": response.kg_source,
                        "Sources": ";".join(response.sources or []),
                        "Context_Chars": len(response.context or ""),
                        "Context_Preview": " ".join((response.context or "").split())[:500],
                        "Steps": response.steps,
                        "Stage2": response.stage2_invoked,
                        "Stage2_Override": response.stage2_override,
                        "Time_ms": round(response.latency_ms, 2),
                        "EM": em,
                        "F1": round(f1, 4),
                        "Acc": acc,
                    }
                    writer.writerow(row)
                    f_csv.flush()
                    rows.append(row)

                    gc.collect()
                    if torch and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"Error evaluating query {qid} on {system_name}: {e}")

        summary = _summarize(rows)
        all_summaries[system_name] = summary
        logger.info(
            f"{system_name}: EM={summary['exact_match']:.4f} "
            f"F1={summary['f1']:.4f} Acc={summary['accuracy']:.4f} "
            f"Latency={summary['avg_latency_ms']:.2f}ms"
        )

    suffix = f"{dataset}_{'full' if limit is None else f'limit{limit}'}"
    _write_summary(
        output_dir / f"{suffix}_summary.json",
        output_dir / f"{suffix}_summary.md",
        all_summaries,
    )
    return all_summaries


def evaluate_dataset_legacy(config_path: str, eval_file: Path, output_csv: Path, limit: int = None):
    logger.info("Legacy single-system evaluation requested.")
    summaries = evaluate_dataset(
        config_path=config_path,
        eval_file=eval_file,
        output_dir=output_csv.parent,
        dataset=output_csv.stem.replace("_results", ""),
        systems=["two_stage_hybrid"],
        limit=limit,
        eval_answer_style=False,
    )
    return summaries.get("two_stage_hybrid")

def _old_evaluate_dataset(config_path: str, eval_file: Path, output_csv: Path, limit: int = None):
    results = []
    total_em = 0
    total_f1 = 0.0
    total_acc = 0
    total_time = 0.0
    count = 0

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(eval_file, "r", encoding="utf-8") as f:
        # Count lines for tqdm
        lines = f.readlines()
        if limit:
            lines = lines[:limit]
        
    logger.info(f"Starting evaluation of {len(lines)} queries...")
    
    with open(output_csv, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.writer(f_csv)
        # Write CSV header
        writer.writerow(["ID", "Query", "Ground_Truth", "Generated_Answer", "Route", "Actual_Route", "Steps", "Time_ms", "EM", "F1", "Acc"])

        for line in tqdm.tqdm(lines, desc="Evaluating"):
            item = json.loads(line.strip())
            qid = item["id"]
            query = item["query"]
            ground_truth = item["ground_truth"]

            try:
                # Query the pipeline with a unique session_id per query
                response = pipeline.query(query, session_id=qid, verbose=False)
                
                # Metrics
                em, f1, acc = evaluate_prediction(ground_truth, response.answer)
                
                # Steps logic based on route
                if response.route_used == "dense_retrieval":
                    steps = 1
                elif response.route_used == "graph_traversal":
                    steps = 2
                elif response.route_used == "hybrid_reasoning":
                    steps = 3
                elif response.route_used == "clarify":
                    steps = 1
                else:
                    steps = 0

                writer.writerow([
                    qid, 
                    query, 
                    ground_truth, 
                    response.answer, 
                    response.route_used, 
                    response.actual_pipeline_used,
                    steps, 
                    round(response.latency_ms, 2), 
                    em, 
                    round(f1, 4), 
                    acc
                ])
                f_csv.flush()

                total_em += em
                total_f1 += f1
                total_acc += acc
                total_time += response.latency_ms
                count += 1
                
                # Aggressively clean up memory to prevent OOM on large datasets
                gc.collect()
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error evaluating query {qid}: {e}")

    if count > 0:
        avg_em = total_em / count
        avg_f1 = total_f1 / count
        avg_acc = total_acc / count
        avg_time = total_time / count
        
        logger.info("=== EVALUATION RESULTS ===")
        logger.info(f"Dataset: {eval_file.name}")
        logger.info(f"Total Queries: {count}")
        logger.info(f"Exact Match (EM): {avg_em:.4f}")
        logger.info(f"F1 Score: {avg_f1:.4f}")
        logger.info(f"Accuracy (Inclusion): {avg_acc:.4f}")
        logger.info(f"Avg Latency: {avg_time:.2f} ms")
        logger.info(f"Results saved to: {output_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_hotpot.yaml")
    parser.add_argument(
        "--dataset",
        default="hotpot",
        help="Dataset name: legal, legal_research, hotpot, vimqa, or a processed prefix",
    )
    parser.add_argument("--eval-file", default=None, help="Explicit evaluation file path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries to evaluate")
    parser.add_argument(
        "--systems",
        default="two_stage_hybrid",
        help="Comma-separated systems or 'all'. Options: pure_vector,pure_graph,single_stage_router,two_stage_hybrid",
    )
    parser.add_argument(
        "--eval-answer-style",
        action="store_true",
        help="Use concise legal-answer prompting and wider retrieval for automatic QA metrics.",
    )
    parser.add_argument(
        "--router-model-path",
        default=None,
        help="Override router.stage1.model_path, useful for strict no-leakage experiments.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.router_model_path:
        config.setdefault("router", {}).setdefault("stage1", {})["model_path"] = args.router_model_path

    if args.eval_file:
        eval_file = Path(args.eval_file)
    elif args.dataset == "legal":
        eval_file = Path("qa_pipeline/data/final/test.json")
    elif args.dataset == "legal_research":
        eval_file = Path("qa_pipeline/data/final/legal_research_eval_600.json")
    elif args.dataset == "legal_strict":
        eval_file = Path("qa_pipeline/data/legal_strict/test.json")
    elif args.dataset == "vimqa":
        eval_file = Path(config.get("data", {}).get("processed_dir", "data/vimqa")) / "test.json"
    else:
        data_dir = Path(config.get("data", {}).get("processed_dir", "data/en_benchmark/processed"))
        eval_file = data_dir / f"{args.dataset}_eval.jsonl"
    
    if args.systems.strip().lower() == "all":
        systems = list(SYSTEMS)
    else:
        systems = [s.strip() for s in args.systems.split(",") if s.strip()]
        unknown = sorted(set(systems) - set(SYSTEMS))
        if unknown:
            raise ValueError(f"Unknown systems: {unknown}. Valid systems: {SYSTEMS}")

    out_dir = Path("eval_results")
    evaluate_dataset(
        args.config,
        eval_file,
        out_dir,
        args.dataset,
        systems,
        args.limit,
        eval_answer_style=args.eval_answer_style,
        router_model_path=args.router_model_path,
    )

if __name__ == "__main__":
    main()
