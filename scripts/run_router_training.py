#!/usr/bin/env python3
"""End-to-end router training + evaluation script.

Usage:
    python run_router_training.py                    # Train on train.json, eval on dev+test
    python run_router_training.py --skip_eval        # Train only
    python run_router_training.py --eval_only        # Load saved model, run eval on test only
    python run_router_training.py --clarify_n 80     # Override clarify augmentation count

This script:
  1. Loads the verified QA train split (qa_pipeline/data/final/train.json)
  2. Adds synthetic 'clarify' samples for augmentation
  3. Extracts features with FeatureExtractor
  4. Trains XGBoost with balanced class weights + 5-fold CV
  5. Saves model to configs/router_model_path (from config.yaml)
  6. Evaluates on dev split (quick sanity check)
  7. Evaluates on test split (final numbers for thesis)
  8. Saves all results to eval_results/router_results.json
  9. Prints full classification report + confusion matrix
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router.features import FeatureExtractor
from router.router_model import RouterModel
from router.train_router import (
    load_qa_dataset_for_training,
    train_and_evaluate,
    _print_training_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Reasoning-Aware Router (Stage 1 XGBoost)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="qa_pipeline/data/final/train.json",
        help="Path to QA training split",
    )
    parser.add_argument(
        "--dev_path",
        type=str,
        default="qa_pipeline/data/final/dev.json",
        help="Path to QA dev split",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="qa_pipeline/data/final/test.json",
        help="Path to QA test split",
    )
    parser.add_argument(
        "--clarify_n",
        type=int,
        default=60,
        help="Number of synthetic 'clarify' samples to add (default: 60)",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation after training",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training, load existing model and evaluate on test split only",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results/router_results.json",
        help="Path to save evaluation results JSON",
    )
    parser.add_argument(
        "--router-model-path",
        type=str,
        default=None,
        help="Override router.stage1.model_path for strict-split experiments",
    )
    return parser.parse_args()


def evaluate_split(
    model: RouterModel,
    extractor: FeatureExtractor,
    split_path: str | Path,
    split_name: str,
    class_labels: list[str],
) -> dict:
    """Evaluate the trained router on a data split.

    Args:
        model: Trained RouterModel.
        extractor: FeatureExtractor instance.
        split_path: Path to the QA split JSON.
        split_name: Human-readable name for this split.
        class_labels: List of class label strings.

    Returns:
        Dict with accuracy, per-class F1, confusion matrix.
    """
    split_path = Path(split_path)
    if not split_path.exists():
        logger.warning("Split not found, skipping: {}", split_path)
        return {}

    logger.info("Evaluating on {} split: {}", split_name, split_path)

    with open(split_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions: list[str] = []
    ground_truth: list[str] = []

    for item in tqdm(data, desc=f"Eval {split_name}"):
        question = item.get("question", "").strip()
        label = item.get("routing_label", "").strip()

        if not question or not label:
            continue

        features = extractor.extract(question)
        output = model.predict(features)

        predictions.append(output.route)
        ground_truth.append(label)

    if not predictions:
        logger.warning("No valid samples in {} split", split_name)
        return {}

    # Build label list (only those present in this split)
    present_labels = sorted(set(ground_truth) | set(predictions))
    present_labels = [l for l in present_labels if l in RouterModel.LABEL_TO_IDX or l == "clarify"]

    pred_idx = [RouterModel.LABEL_TO_IDX.get(p, len(class_labels) - 1) for p in predictions]
    gt_idx = [RouterModel.LABEL_TO_IDX.get(g, len(class_labels) - 1) for g in ground_truth]

    # Classification report
    cls_report_str = classification_report(
        ground_truth,
        predictions,
        labels=present_labels,
        zero_division=0,
    )
    cls_report_dict = classification_report(
        ground_truth,
        predictions,
        labels=present_labels,
        output_dict=True,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(ground_truth, predictions, labels=present_labels)

    # Overall accuracy
    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    accuracy = correct / len(predictions)

    print(f"\n{'=' * 60}")
    print(f"EVALUATION RESULTS — {split_name.upper()} SPLIT ({len(data)} samples)")
    print(f"{'=' * 60}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(cls_report_str)

    # Print confusion matrix
    print("Confusion Matrix:")
    header = "              " + "  ".join(f"{l[:12]:>12s}" for l in present_labels)
    print(header)
    for i, row in enumerate(cm.tolist()):
        lbl = present_labels[i] if i < len(present_labels) else str(i)
        row_str = f"  {lbl[:12]:12s}" + "  ".join(f"{v:>12d}" for v in row)
        print(row_str)

    return {
        "split": split_name,
        "accuracy": accuracy,
        "num_samples": len(predictions),
        "per_class": {
            label: {
                "precision": cls_report_dict[label]["precision"],
                "recall": cls_report_dict[label]["recall"],
                "f1": cls_report_dict[label]["f1-score"],
                "support": cls_report_dict[label]["support"],
            }
            for label in present_labels
            if label in cls_report_dict
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": present_labels,
        "macro_f1": cls_report_dict.get("macro avg", {}).get("f1-score", 0.0),
        "weighted_f1": cls_report_dict.get("weighted avg", {}).get("f1-score", 0.0),
    }


def main() -> None:
    args = parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config not found: {}", config_path)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.router_model_path:
        model_path = Path(args.router_model_path)
        config.setdefault("router", {}).setdefault("stage1", {})["model_path"] = str(model_path)
        config.setdefault("data", {})["router_training_dir"] = str(model_path.parent)
        logger.info("Router model path overridden: {}", model_path)

    logger.info("=" * 70)
    logger.info("Reasoning-Aware Router Training & Evaluation")
    logger.info("=" * 70)
    logger.info("Class labels: {}", RouterModel.CLASS_LABELS)

    extractor = FeatureExtractor(config=config)
    model = RouterModel(config)

    # ── TRAINING ──────────────────────────────────────────────────────────
    if not args.eval_only:
        logger.info("Phase 1: Loading training data from {}", args.train_path)

        train_df = load_qa_dataset_for_training(
            split_path=args.train_path,
            include_clarify_augment=True,
            clarify_samples=args.clarify_n,
        )

        dev_df = load_qa_dataset_for_training(
            split_path=args.dev_path,
            include_clarify_augment=False,
            clarify_samples=0,
        ) if Path(args.dev_path).exists() else None

        logger.info("Phase 2: Extracting features and training XGBoost...")

        # Feature extraction
        feature_vectors: list[list[float]] = []
        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Features"):
            feat = extractor.extract(row["query"])
            feature_vectors.append(feat.to_vector())

        X = np.array(feature_vectors, dtype=np.float32)
        y = np.array(train_df["label_idx"].values, dtype=np.int32)

        logger.info("Feature matrix shape: {} | Label distribution: {}", X.shape, np.bincount(y))

        # Train
        report = model.train(X, y)
        model.save()

        # Print training report
        _print_training_report(report, RouterModel.CLASS_LABELS)

    else:
        # Load existing model
        logger.info("eval_only mode: loading existing model...")
        if not model.load():
            logger.error("No saved model found at {}", model.model_path)
            sys.exit(1)

    # ── EVALUATION ────────────────────────────────────────────────────────
    if not args.skip_eval:
        logger.info("Phase 3: Evaluating on dev and test splits...")

        all_results: list[dict] = []

        dev_result = evaluate_split(
            model, extractor,
            args.dev_path, "dev",
            RouterModel.CLASS_LABELS,
        )
        if dev_result:
            all_results.append(dev_result)

        test_result = evaluate_split(
            model, extractor,
            args.test_path, "test",
            RouterModel.CLASS_LABELS,
        )
        if test_result:
            all_results.append(test_result)

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        logger.info("Results saved to {}", output_path)

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for r in all_results:
            print(
                f"  {r['split']:6s} | accuracy={r['accuracy']:.4f} | "
                f"macro_f1={r.get('macro_f1', 0):.4f} | "
                f"weighted_f1={r.get('weighted_f1', 0):.4f} | "
                f"n={r['num_samples']}"
            )
        print("=" * 70)

    logger.info("Done.")


if __name__ == "__main__":
    main()
