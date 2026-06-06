"""Router training pipeline — loads real QA dataset and trains XGBoost.

Primary data source: qa_pipeline/data/final/train.json (940-sample QA dataset
with verified routing_label annotations: dense_retrieval / graph_traversal /
hybrid_reasoning).

Weak-label templates are used only as supplementary augmentation for the
'clarify' class (not present in the QA dataset).
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from tqdm import tqdm

from router.features import FeatureExtractor, QueryFeatures
from router.router_model import RouterModel, TrainingReport


# ──────────────────────────────────────────────────────────────────────────────
# Clarify-class templates (used only to augment the 'clarify' class)
# These represent ambiguous queries that cannot be answered without more context.
# ──────────────────────────────────────────────────────────────────────────────
CLARIFY_TEMPLATES: list[str] = [
    "Ông ấy có quyền gì?",
    "Luật đó quy định như thế nào?",
    "Điều đó có đúng không?",
    "Người đó có bị xử phạt không?",
    "Cơ quan đó có thẩm quyền gì?",
    "Quy định có áp dụng không?",
    "Họ phải làm gì?",
    "Vấn đề đó giải quyết ra sao?",
    "Bên đó có trách nhiệm gì?",
    "Điều khoản đó nói gì?",
    "Luật này áp dụng cho ai?",
    "Nó có còn hiệu lực không?",
    "Trường hợp đó được xử lý như thế nào?",
    "Bà ấy có được miễn không?",
    "Quyết định đó có hợp lệ không?",
]


def load_qa_dataset_for_training(
    split_path: str | Path,
    include_clarify_augment: bool = True,
    clarify_samples: int = 60,
) -> pd.DataFrame:
    """Load the verified QA dataset as training data for the router.

    Reads routing_label from each sample and converts to an integer index
    for XGBoost training. Optionally augments with synthetic 'clarify'
    samples (since that class is not present in the QA dataset).

    Args:
        split_path: Path to a QA split JSON file (train/dev/test).
        include_clarify_augment: Whether to add synthetic 'clarify' samples.
        clarify_samples: Number of synthetic 'clarify' samples to add.

    Returns:
        DataFrame with columns: query, label, label_idx.
    """
    split_path = Path(split_path)
    logger.info("Loading QA dataset from {}", split_path)

    with open(split_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    label_map = RouterModel.LABEL_TO_IDX

    rows: list[dict[str, Any]] = []
    skipped = 0

    for item in data:
        question = item.get("question", "").strip()
        label = item.get("routing_label", "").strip()

        if not question or not label:
            skipped += 1
            continue

        if label not in label_map:
            logger.warning("Unknown routing_label '{}', skipping", label)
            skipped += 1
            continue

        rows.append({
            "query": question,
            "label": label,
            "label_idx": label_map[label],
            "hop_count": item.get("hop_count", 1),
            "is_cross_doc": item.get("is_cross_doc", False),
            "difficulty": item.get("difficulty", 0.0),
            "source": "qa_dataset",
        })

    logger.info(
        "Loaded {} samples from QA dataset ({} skipped)",
        len(rows), skipped,
    )

    # Log label distribution
    from collections import Counter
    dist = Counter(r["label"] for r in rows)
    logger.info("Label distribution: {}", dict(dist))

    # Augment with 'clarify' samples (not in QA dataset)
    if include_clarify_augment and clarify_samples > 0:
        clarify_idx = label_map.get("clarify")
        if clarify_idx is not None:
            random.seed(42)
            for _ in range(clarify_samples):
                rows.append({
                    "query": random.choice(CLARIFY_TEMPLATES),
                    "label": "clarify",
                    "label_idx": clarify_idx,
                    "hop_count": 0,
                    "is_cross_doc": False,
                    "difficulty": 0.5,
                    "source": "synthetic_clarify",
                })
            logger.info("Added {} synthetic 'clarify' samples", clarify_samples)

    df = pd.DataFrame(rows)
    random.seed(42)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(
        "Final training DataFrame: {} rows | label distribution: {}",
        len(df),
        dict(Counter(df["label"])),
    )

    return df


def train_and_evaluate(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    eval_df: pd.DataFrame | None = None,
) -> TrainingReport:
    """Train the router model and evaluate performance.

    Pipeline:
    1. Extract features for all queries.
    2. Build feature matrix X and label vector y.
    3. Train XGBoost with 5-fold cross-validation + balanced class weights.
    4. If eval_df provided, also evaluate on that split.
    5. Print accuracy, F1 per class, confusion matrix.
    6. Save model.

    Args:
        df: Training DataFrame with columns: query, label, label_idx.
        config: Full config dict.
        eval_df: Optional held-out evaluation DataFrame (e.g. dev split).

    Returns:
        TrainingReport with all metrics.
    """
    if config is None:
        config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    # Initialize feature extractor (NER is optional — speeds up training when disabled)
    extractor = FeatureExtractor(config=config)

    # ── Extract training features ──────────────────────────────────────────
    logger.info("Extracting features for {} training queries...", len(df))
    feature_vectors: list[list[float]] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Train features"):
        features = extractor.extract(row["query"])
        feature_vectors.append(features.to_vector())

    X = np.array(feature_vectors, dtype=np.float32)
    y = np.array(df["label_idx"].values, dtype=np.int32)

    logger.info("Feature matrix: {} | Label counts: {}", X.shape, np.bincount(y))

    # ── Train ──────────────────────────────────────────────────────────────
    model = RouterModel(config)
    report = model.train(X, y)

    # ── Save model ─────────────────────────────────────────────────────────
    save_dir = Path(config["data"]["router_training_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save()

    # Save training data summary
    training_data_path = save_dir / "training_data.csv"
    df[["query", "label", "label_idx", "source"]].to_csv(
        training_data_path, index=False, encoding="utf-8"
    )
    logger.info("Training data saved to {}", training_data_path)

    # ── Evaluate on held-out split if provided ─────────────────────────────
    if eval_df is not None and not eval_df.empty:
        logger.info("Evaluating on held-out split ({} samples)...", len(eval_df))
        eval_features: list[list[float]] = []
        for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Eval features"):
            feat = extractor.extract(row["query"])
            eval_features.append(feat.to_vector())

        X_eval = np.array(eval_features, dtype=np.float32)
        y_eval = np.array(eval_df["label_idx"].values, dtype=np.int32)

        eval_preds = [model.predict(
            # re-create QueryFeatures from vector — simpler: use direct predict
            type("F", (), {"to_vector": lambda self: eval_features[i]})()
        ).route for i in range(len(eval_features))]

        # Convert route strings → idx for comparison
        from router.router_model import RouterModel as RM
        pred_idx = [RM.LABEL_TO_IDX.get(r, 0) for r in eval_preds]
        from sklearn.metrics import accuracy_score, classification_report as cr
        eval_acc = accuracy_score(y_eval, pred_idx)
        logger.info("Held-out eval accuracy: {:.4f}", eval_acc)

    # ── Print results ──────────────────────────────────────────────────────
    _print_training_report(report, RouterModel.CLASS_LABELS)

    return report


def _print_training_report(report: TrainingReport, class_labels: list[str]) -> None:
    """Print a rich training report to stdout.

    Args:
        report: TrainingReport from RouterModel.train().
        class_labels: List of class label strings.
    """
    print("\n" + "=" * 70)
    print("ROUTER TRAINING RESULTS")
    print("=" * 70)
    print(f"Validation Accuracy : {report.accuracy:.4f}")

    if report.cv_scores:
        arr = np.array(report.cv_scores)
        print(f"CV Accuracy (5-fold): {arr.mean():.4f} ± {arr.std():.4f}")

    print("\n--- Classification Report (val split) ---")
    if report.classification_report_str:
        print(report.classification_report_str)
    elif report.f1_per_class:
        print("F1 per class:")
        for cls, f1 in report.f1_per_class.items():
            print(f"  {cls:25s}: {f1:.4f}")

    if report.confusion_mat:
        present = list(range(len(report.confusion_mat)))
        present_labels = [class_labels[i] for i in present if i < len(class_labels)]
        print("--- Confusion Matrix ---")
        header = "              " + "  ".join(f"{l[:12]:>12s}" for l in present_labels)
        print(header)
        for i, row in enumerate(report.confusion_mat):
            lbl = present_labels[i] if i < len(present_labels) else str(i)
            row_str = f"  {lbl[:12]:12s}" + "  ".join(f"{v:>12d}" for v in row)
            print(row_str)

    if report.feature_importances:
        print("\nTop 5 Feature Importances:")
        sorted_feats = sorted(
            report.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for name, imp in sorted_feats[:5]:
            print(f"  {name:30s}: {imp:.4f}")

    print("=" * 70)
