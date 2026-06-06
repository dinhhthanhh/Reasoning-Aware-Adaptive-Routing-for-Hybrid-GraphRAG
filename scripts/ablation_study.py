"""Ablation Study for Router Threshold Optimization.

Runs routing-only evaluation (no LLM answer generation) across
different threshold configurations to find optimal values.

This is ~100x faster than full pipeline evaluation because:
  1. Only extracts features + routes (no retrieval, no LLM generation)
  2. Measures routing accuracy against gold labels
  3. Parallelizes across threshold configs

Usage:
  # Vietnamese Legal QA (default)
  python -m scripts.ablation_study --config configs/config.yaml

  # ViMQA (Vietnamese multi-hop)
  python -m scripts.ablation_study --config configs/config.yaml --dataset vimqa

  # HotpotQA (English multi-hop)
  python -m scripts.ablation_study --config configs/config_en.yaml --dataset hotpotqa

  # All datasets at once (cross-dataset comparison)
  python -m scripts.ablation_study --dataset all
"""

from __future__ import annotations

import copy
import json
import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AblationResult:
    """Result of a single ablation run."""
    param_name: str
    param_value: Any
    routing_accuracy: float = 0.0
    per_class_accuracy: dict[str, float] = field(default_factory=dict)
    stage2_rate: float = 0.0
    misroute_rate: float = 0.0
    avg_latency_ms: float = 0.0
    total_queries: int = 0


@dataclass
class ThresholdConfig:
    """A single threshold parameter to sweep."""
    name: str
    values: list[Any]
    # Path in config dict (dot-separated) or "code" if it's a code-level param
    config_path: str | None = None
    # If "code", which attribute to modify on the router object
    code_attr: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Threshold Definitions — organized by theory vs empirical
# ─────────────────────────────────────────────────────────────────────────────

# GROUP A: Has theoretical basis — likely don't need sweeping
# (but included for completeness / validation)
THEORY_BASED = [
    ThresholdConfig(
        name="entity_count_threshold",
        values=[0, 1, 2, 3, 5, 10],
        code_attr="entity_count_threshold",
    ),
    ThresholdConfig(
        name="complexity_level_3_law_names",
        values=[1, 2, 3, 5],
        code_attr="complexity_l3_law_names",
    ),
]

# GROUP B: Needs experimental validation — sweep these
EMPIRICAL = [
    ThresholdConfig(
        name="confidence_threshold",
        values=[0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99],
        config_path="router.stage1.confidence_threshold",
    ),
    ThresholdConfig(
        name="multi_hop_normalization",
        values=[1.0, 2.0, 3.0, 5.0, 10.0],
        code_attr="multi_hop_norm_divisor",
    ),
    ThresholdConfig(
        name="multi_hop_score_override",
        values=[0.05, 0.1, 0.3, 0.5, 0.7, 0.9],
        code_attr="multi_hop_score_threshold",
    ),
    ThresholdConfig(
        name="comparison_boost",
        values=[0.3, 0.5, 0.7, 0.9, 1.0],
        code_attr="comparison_boost",
    ),
    ThresholdConfig(
        name="cross_doc_boost",
        values=[0.5, 0.7, 0.9, 1.0],
        code_attr="cross_doc_boost",
    ),
    ThresholdConfig(
        name="ambiguity_clarify_threshold",
        values=[0.5, 0.7, 0.8, 0.9, 0.95, 0.99],
        code_attr="ambiguity_clarify_threshold",
    ),
    ThresholdConfig(
        name="ambiguity_force_stage2_threshold",
        values=[0.2, 0.4, 0.6, 0.8],
        code_attr="ambiguity_force_stage2_threshold",
    ),
    ThresholdConfig(
        name="high_confidence_dense_skip_threshold",
        values=[0.8, 0.85, 0.9, 0.95],
        code_attr="high_confidence_dense_skip_threshold",
    ),
    ThresholdConfig(
        name="dense_skip_max_ambiguity",
        values=[0.2, 0.4, 0.6],
        code_attr="dense_skip_max_ambiguity",
    ),
    ThresholdConfig(
        name="reasoning_force_stage2_enabled",
        values=[False, True],
        code_attr="reasoning_force_stage2_enabled",
    ),
    ThresholdConfig(
        name="reasoning_force_confidence_ceiling",
        values=[0.6, 0.7, 0.8],
        code_attr="reasoning_force_confidence_ceiling",
    ),
]

ALL_THRESHOLDS = THEORY_BASED + EMPIRICAL


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Definitions
# ─────────────────────────────────────────────────────────────────────────────

DATASET_PATHS = {
    "legal": {
        "test": "qa_pipeline/data/final/test.json",
        "fallback": "qa_pipeline/data/final/test.json",
        "config": "configs/config.yaml",
        "description": "Vietnamese Legal QA (custom)",
    },
    "vimqa": {
        "test": "data/vimqa/validation.json",
        "fallback": "data/vimqa/train.json",
        "config": "configs/config.yaml",  # Vietnamese
        "description": "ViMQA - Vietnamese Multi-hop QA (Wikipedia)",
    },
    "hotpotqa": {
        "test": "data/en_benchmark/processed/hotpot_eval.jsonl",
        "fallback": "qa_pipeline/data/final/test.json",
        "config": "configs/config_en.yaml",
        "description": "HotpotQA - English Multi-hop QA",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Core: Routing-Only Evaluation (Fast Path)
# ─────────────────────────────────────────────────────────────────────────────

def load_test_data(
    config: dict,
    test_path: str | None = None,
    dataset_name: str = "legal",
    max_samples: int | None = None,
) -> list[dict]:
    """Load test data with gold routing labels.

    Supports multiple dataset formats:
    - legal: qa_pipeline format (question, routing_label, hop_count)
    - vimqa: ViMQA format (question, answer, routing_label)
    - hotpotqa: HotpotQA format (question, expected_route)
    """
    if test_path:
        p = Path(test_path)
    else:
        ds_info = DATASET_PATHS.get(dataset_name, DATASET_PATHS["legal"])
        p = Path(ds_info["test"])
        if not p.exists():
            p = Path(ds_info["fallback"])

    if not p.exists():
        raise FileNotFoundError(
            f"Test data not found: {p}\n"
            f"For ViMQA, run first: python -m scripts.download_vimqa\n"
            f"For HotpotQA, run first: python -m scripts.download_en_benchmark"
        )

    if p.suffix == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    else:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

    # Normalize format
    items = []
    for d in data:
        q = d.get("question", d.get("query", ""))
        route = d.get("routing_label", d.get("expected_route", "dense_retrieval"))
        hop = d.get("hop_count", 1)
        items.append({"query": q, "expected_route": route, "hop_count": hop})

    if max_samples:
        items = items[:max_samples]

    logger.info("Loaded {} queries from {} ({})", len(items), p.name, dataset_name)
    return items


def evaluate_routing_only(
    config: dict,
    test_data: list[dict],
    overrides: dict[str, Any] | None = None,
) -> AblationResult:
    """Run routing-only evaluation — NO retrieval, NO LLM generation.

    This is the fast path: extract features → route → compare to gold.
    ~100x faster than full pipeline evaluation.

    Args:
        config: Full config dict.
        test_data: List of test items with query and expected_route.
        overrides: Dict of threshold overrides to apply.

    Returns:
        AblationResult with routing accuracy metrics.
    """
    overrides = overrides or {}
    cfg = copy.deepcopy(config)

    # Apply config-level overrides
    for key, value in overrides.items():
        if "." in key:
            _set_nested(cfg, key, value)

    # Import here to avoid circular imports at module level
    from router.features import FeatureExtractor
    from router.ambiguity_detector import AmbiguityDetector
    from router.router_model import RouterModel

    # Initialize components
    feature_extractor = FeatureExtractor(config=cfg)
    ambiguity_detector = AmbiguityDetector(cfg.get("ambiguity"))
    router_model = RouterModel(cfg)

    # Extract code-level overrides
    entity_threshold = overrides.get("entity_count_threshold", 1)
    confidence_threshold = cfg["router"]["stage1"].get("confidence_threshold", 0.85)
    mh_norm = overrides.get("multi_hop_norm_divisor", 3.0)
    mh_score_threshold = overrides.get("multi_hop_score_threshold", 0.3)
    comp_boost = overrides.get("comparison_boost", 0.7)
    cross_boost = overrides.get("cross_doc_boost", 0.9)
    amb_clarify = overrides.get("ambiguity_clarify_threshold", 0.8)
    amb_force_s2 = overrides.get("ambiguity_force_stage2_threshold", 0.6)
    high_conf_dense_skip = overrides.get("high_confidence_dense_skip_threshold", confidence_threshold)
    dense_skip_max_ambiguity = overrides.get("dense_skip_max_ambiguity", 0.4)
    reasoning_force_enabled = overrides.get("reasoning_force_stage2_enabled", False)
    reasoning_force_threshold = overrides.get("reasoning_force_stage2_threshold", 0.6)
    reasoning_conf_ceiling = overrides.get("reasoning_force_confidence_ceiling", 0.7)

    correct = 0
    total = 0
    class_correct: dict[str, int] = {}
    class_total: dict[str, int] = {}
    stage2_count = 0
    misroute_count = 0
    latencies: list[float] = []

    for item in test_data:
        query = item["query"]
        expected = item["expected_route"]

        start = time.perf_counter()

        # Step 1: Ambiguity detection
        amb_report = ambiguity_detector.detect(query)

        # Step 2: Feature extraction
        features = feature_extractor.extract(
            query=query,
            ambiguity_score=amb_report.score,
            has_pronoun="pronoun" in amb_report.ambiguity_types,
            missing_entity_type=amb_report.missing_entity_type,
        )

        # Step 3: Stage 1 prediction
        s1_output = router_model.predict(features)
        final_route = s1_output.route
        final_confidence = s1_output.confidence

        # Step 4: Ambiguity override
        # Dense-forcing hard overrides were removed from the production router
        # because they suppress valid multi-hop questions with sparse surface cues.
        if amb_report.is_ambiguous and amb_report.score >= amb_clarify:
            final_route = "clarify"
            final_confidence = amb_report.score

        # Step 5: Would Stage 2 trigger?
        dense_fast_path = (
            final_route == "dense_retrieval"
            and final_confidence >= high_conf_dense_skip
            and amb_report.score <= dense_skip_max_ambiguity
        )

        if dense_fast_path:
            is_uncertain = False
        else:
            is_uncertain = final_confidence < confidence_threshold
            if amb_report.is_ambiguous and amb_report.score >= amb_force_s2:
                is_uncertain = True

            reasoning_signal = (
                features.multi_hop_score >= reasoning_force_threshold
                or features.cross_doc_signals
                or features.graph_keyword_count >= 2
                or features.complexity_level >= 3
            )
            if (
                reasoning_force_enabled
                and reasoning_signal
                and final_confidence <= reasoning_conf_ceiling
            ):
                is_uncertain = True

            if final_route == "clarify" and amb_report.score < amb_force_s2:
                is_uncertain = True
        if is_uncertain:
            stage2_count += 1

        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

        # Score
        total += 1
        class_total[expected] = class_total.get(expected, 0) + 1

        if final_route == expected:
            correct += 1
            class_correct[expected] = class_correct.get(expected, 0) + 1
        else:
            misroute_count += 1

    # Build result
    per_class_acc = {}
    for cls in class_total:
        per_class_acc[cls] = class_correct.get(cls, 0) / class_total[cls] if class_total[cls] > 0 else 0.0

    return AblationResult(
        param_name="",
        param_value=None,
        routing_accuracy=correct / total if total > 0 else 0.0,
        per_class_accuracy=per_class_acc,
        stage2_rate=stage2_count / total if total > 0 else 0.0,
        misroute_rate=misroute_count / total if total > 0 else 0.0,
        avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        total_queries=total,
    )


def _set_nested(d: dict, key: str, value: Any) -> None:
    """Set a nested dict value using dot notation."""
    parts = key.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


# ─────────────────────────────────────────────────────────────────────────────
# Ablation Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_single_param_sweep(
    config: dict,
    test_data: list[dict],
    threshold: ThresholdConfig,
) -> list[AblationResult]:
    """Sweep a single threshold parameter across its values."""
    results = []

    for value in threshold.values:
        overrides = {}

        if threshold.config_path:
            overrides[threshold.config_path] = value
        if threshold.code_attr:
            overrides[threshold.code_attr] = value

        result = evaluate_routing_only(config, test_data, overrides)
        result.param_name = threshold.name
        result.param_value = value

        logger.info(
            "  {} = {} → accuracy={:.4f} | stage2_rate={:.4f} | misroute={:.4f}",
            threshold.name, value,
            result.routing_accuracy, result.stage2_rate, result.misroute_rate,
        )
        results.append(result)

    return results


def run_ablation_study(
    config_path: str = "configs/config.yaml",
    test_path: str | None = None,
    sweep_group: str = "all",
    dataset_name: str = "legal",
    max_samples: int | None = None,
) -> dict[str, list[AblationResult]]:
    """Run complete ablation study.

    Args:
        config_path: Path to config file.
        test_path: Path to test data (optional).
        sweep_group: "all", "theory", or "empirical".
        dataset_name: "legal", "vimqa", "hotpotqa".
        max_samples: Limit samples per dataset (for quick testing).

    Returns:
        Dict mapping param_name to list of AblationResult.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    test_data = load_test_data(config, test_path, dataset_name, max_samples)
    logger.info("Dataset: {} | {} queries", dataset_name, len(test_data))

    # Select which thresholds to sweep
    if sweep_group == "theory":
        thresholds = THEORY_BASED
    elif sweep_group == "empirical":
        thresholds = EMPIRICAL
    else:
        thresholds = ALL_THRESHOLDS

    all_results: dict[str, list[AblationResult]] = {}

    # First run baseline (default config)
    logger.info("=" * 60)
    logger.info("Running BASELINE (default config)...")
    baseline = evaluate_routing_only(config, test_data)
    baseline.param_name = "baseline"
    baseline.param_value = "default"
    all_results["baseline"] = [baseline]
    logger.info(
        "BASELINE: accuracy={:.4f} | stage2_rate={:.4f}",
        baseline.routing_accuracy, baseline.stage2_rate,
    )

    # Sweep each parameter
    for threshold in thresholds:
        logger.info("=" * 60)
        logger.info("Sweeping: {} (values={})", threshold.name, threshold.values)
        results = run_single_param_sweep(config, test_data, threshold)
        all_results[threshold.name] = results

    return all_results


def run_cross_dataset_ablation(
    sweep_group: str = "empirical",
    max_samples: int | None = 500,
) -> dict[str, dict[str, list[AblationResult]]]:
    """Run ablation across ALL available datasets for cross-dataset comparison.

    Returns:
        Nested dict: {dataset_name: {param_name: [AblationResult]}}
    """
    cross_results = {}

    for ds_name, ds_info in DATASET_PATHS.items():
        test_path = Path(ds_info["test"])
        fallback_path = Path(ds_info["fallback"])

        if not test_path.exists() and not fallback_path.exists():
            logger.warning("Skipping {} — data not found ({})", ds_name, test_path)
            continue

        config_path = ds_info["config"]
        logger.info("\n" + "#" * 70)
        logger.info("# DATASET: {} ({})", ds_name.upper(), ds_info["description"])
        logger.info("#" * 70)

        try:
            results = run_ablation_study(
                config_path=config_path,
                dataset_name=ds_name,
                sweep_group=sweep_group,
                max_samples=max_samples,
            )
            cross_results[ds_name] = results
        except Exception as e:
            logger.error("Failed on {}: {}", ds_name, e)

    return cross_results


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_ablation_report(all_results: dict[str, list[AblationResult]]) -> None:
    """Print ablation results as formatted tables."""
    from tabulate import tabulate

    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)

    # Baseline
    if "baseline" in all_results:
        bl = all_results["baseline"][0]
        print(f"\n📊 BASELINE: accuracy={bl.routing_accuracy:.4f} | "
              f"stage2_rate={bl.stage2_rate:.4f} | queries={bl.total_queries}")

    # Per-parameter results
    for param_name, results in all_results.items():
        if param_name == "baseline":
            continue

        print(f"\n{'─' * 60}")
        print(f"📌 Parameter: {param_name}")
        print(f"{'─' * 60}")

        table = []
        best_acc = max(r.routing_accuracy for r in results)

        for r in results:
            marker = " ⭐" if r.routing_accuracy == best_acc else ""
            table.append([
                r.param_value,
                f"{r.routing_accuracy:.4f}{marker}",
                f"{r.stage2_rate:.4f}",
                f"{r.misroute_rate:.4f}",
                f"{r.avg_latency_ms:.1f}",
            ])

        print(tabulate(
            table,
            headers=["Value", "Routing Acc", "Stage2 Rate", "Misroute Rate", "Latency(ms)"],
            tablefmt="grid",
        ))

        # Per-class breakdown for best config
        best_result = max(results, key=lambda r: r.routing_accuracy)
        if best_result.per_class_accuracy:
            print(f"\n  Best value: {best_result.param_value}")
            for cls, acc in sorted(best_result.per_class_accuracy.items()):
                print(f"    {cls}: {acc:.4f}")


def save_ablation_results(
    all_results: dict[str, list[AblationResult]],
    output_path: str = "eval_results/ablation_results.json",
) -> None:
    """Save ablation results to JSON."""
    output = {}
    for param_name, results in all_results.items():
        output[param_name] = [
            {
                "param_name": r.param_name,
                "param_value": r.param_value,
                "routing_accuracy": r.routing_accuracy,
                "per_class_accuracy": r.per_class_accuracy,
                "stage2_rate": r.stage2_rate,
                "misroute_rate": r.misroute_rate,
                "avg_latency_ms": r.avg_latency_ms,
                "total_queries": r.total_queries,
            }
            for r in results
        ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Ablation results saved to {}", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Router Threshold Ablation Study")
    parser.add_argument("--config", default=None, help="Config path (auto-detected per dataset if omitted)")
    parser.add_argument("--test", default=None, help="Test data path (overrides --dataset)")
    parser.add_argument(
        "--dataset", default="legal",
        choices=["legal", "vimqa", "hotpotqa", "all"],
        help="Dataset to evaluate on (default: legal)",
    )
    parser.add_argument(
        "--group", default="empirical",
        choices=["all", "theory", "empirical"],
        help="Which thresholds to sweep (default: empirical)",
    )
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per dataset (for quick testing)")
    parser.add_argument("--output", default="eval_results/ablation_results.json")

    args = parser.parse_args()

    if args.dataset == "all":
        # Cross-dataset comparison mode
        cross_results = run_cross_dataset_ablation(
            sweep_group=args.group,
            max_samples=args.max_samples,
        )
        # Print and save each dataset's results
        for ds_name, results in cross_results.items():
            print(f"\n{'█' * 70}")
            print(f"█ DATASET: {ds_name.upper()}")
            print(f"{'█' * 70}")
            print_ablation_report(results)
            out_path = args.output.replace(".json", f"_{ds_name}.json")
            save_ablation_results(results, out_path)

        # Print cross-dataset summary
        print(f"\n{'═' * 70}")
        print("CROSS-DATASET SUMMARY — Best values per parameter")
        print(f"{'═' * 70}")
        all_params = set()
        for results in cross_results.values():
            all_params.update(k for k in results.keys() if k != "baseline")

        for param in sorted(all_params):
            print(f"\n📌 {param}:")
            for ds_name, results in cross_results.items():
                if param in results:
                    best = max(results[param], key=lambda r: r.routing_accuracy)
                    print(f"    {ds_name:12s}: best={best.param_value} (acc={best.routing_accuracy:.4f})")
    else:
        # Single dataset mode
        config_path = args.config
        if config_path is None:
            ds_info = DATASET_PATHS.get(args.dataset, DATASET_PATHS["legal"])
            config_path = ds_info["config"]

        results = run_ablation_study(
            config_path=config_path,
            test_path=args.test,
            sweep_group=args.group,
            dataset_name=args.dataset,
            max_samples=args.max_samples,
        )

        print_ablation_report(results)
        
        # Luôn tạo file riêng theo tên dataset để tránh ghi đè
        final_output = args.output
        if args.dataset != "all" and f"_{args.dataset}" not in final_output:
             final_output = final_output.replace(".json", f"_{args.dataset}.json")
             
        save_ablation_results(results, final_output)
