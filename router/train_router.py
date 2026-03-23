"""Router training pipeline with weak label generation.

Auto-generates training data from processed documents using heuristics,
augments with template paraphrasing, and trains the XGBoost classifier.
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


# Query templates for data augmentation
VECTOR_TEMPLATES: list[str] = [
    "{entity} được quy định như thế nào?",
    "Nội dung {entity} là gì?",
    "{entity} quy định về vấn đề gì?",
    "Cho tôi biết về {entity}",
    "Quy định tại {entity} nói gì?",
    "{entity} có hiệu lực khi nào?",
    "Ai chịu trách nhiệm theo {entity}?",
]

GRAPH_TEMPLATES: list[str] = [
    "So sánh {entity1} và {entity2}",
    "Mối quan hệ giữa {entity1} với {entity2} là gì?",
    "Sự khác nhau giữa {entity1} và {entity2}?",
    "{entity1} tham chiếu đến {entity2} như thế nào?",
    "Theo {entity1} và {entity2}, quy định nào áp dụng?",
    "{entity1} liên quan đến {entity2} ra sao?",
    "Căn cứ vào {entity1} và {entity2}, kết luận gì?",
]

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
]


def generate_weak_labels(
    processed_docs_dir: str | Path,
    config: dict[str, Any] | None = None,
    max_queries_per_class: int = 200,
) -> pd.DataFrame:
    """Generate weakly-labeled training data from processed documents.

    Uses heuristics to classify synthetic queries:
    - Queries with comparisons / multi-entity references → 'graph'
    - Queries with ambiguous pronouns → 'clarify'
    - Simple single-entity queries → 'vector'

    Augments data with template paraphrasing and ensures class balance.

    Args:
        processed_docs_dir: Directory with processed JSON docs.
        config: Full config dict.
        max_queries_per_class: Maximum samples per class.

    Returns:
        DataFrame with columns: query, label, label_idx.
    """
    docs_path = Path(processed_docs_dir)
    if config is None:
        config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    # Collect legal entities from documents
    entities: list[str] = []
    json_files = sorted(docs_path.glob("*.json"))

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                doc = json.load(f)
            content = doc.get("content", "")

            # Extract article references
            article_refs = re.findall(r"Điều\s+\d+[a-zđ]?", content, re.IGNORECASE)
            entities.extend(article_refs[:5])  # Limit per doc

            # Extract law names
            law_refs = re.findall(
                r"(?:Bộ\s+luật|Luật)\s+[^\n,.]{3,40}",
                content,
                re.IGNORECASE,
            )
            entities.extend(law_refs[:3])
        except (json.JSONDecodeError, IOError):
            continue

    # Deduplicate
    entities = list(set(entities))
    if not entities:
        # Fallback: generate some default legal entities
        entities = [
            "Điều 32",
            "Điều 45",
            "Điều 128",
            "Luật Lao động 2019",
            "Luật Doanh nghiệp 2020",
            "Bộ luật Dân sự 2015",
            "Luật Đất đai 2013",
            "Nghị định 132/2020",
        ]

    logger.info("Found {} unique legal entities for training data", len(entities))

    queries: list[dict[str, Any]] = []
    random.seed(42)

    # Generate vector queries
    for _ in range(max_queries_per_class):
        entity = random.choice(entities)
        template = random.choice(VECTOR_TEMPLATES)
        query = template.format(entity=entity)
        queries.append({"query": query, "label": "vector", "label_idx": 0})

    # Generate graph queries
    for _ in range(max_queries_per_class):
        if len(entities) >= 2:
            ent1, ent2 = random.sample(entities, 2)
        else:
            ent1 = entities[0]
            ent2 = "Điều 1"
        template = random.choice(GRAPH_TEMPLATES)
        query = template.format(entity1=ent1, entity2=ent2)
        queries.append({"query": query, "label": "graph", "label_idx": 1})

    # Generate clarify queries
    for _ in range(max_queries_per_class):
        template = random.choice(CLARIFY_TEMPLATES)
        queries.append({"query": template, "label": "clarify", "label_idx": 2})

    # Shuffle
    random.shuffle(queries)

    df = pd.DataFrame(queries)
    logger.info(
        "Generated {} training samples | vector={} | graph={} | clarify={}",
        len(df),
        len(df[df["label"] == "vector"]),
        len(df[df["label"] == "graph"]),
        len(df[df["label"] == "clarify"]),
    )

    return df


def train_and_evaluate(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> TrainingReport:
    """Train the router model and evaluate performance.

    Pipeline:
    1. Extract features for all queries
    2. Build feature matrix X and label vector y
    3. Train XGBoost with 5-fold cross-validation
    4. Print accuracy, F1 per class, confusion matrix
    5. Save model

    Args:
        df: DataFrame with columns: query, label, label_idx.
        config: Full config dict.

    Returns:
        TrainingReport with all metrics.
    """
    if config is None:
        config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    # Initialize feature extractor (without NER to speed up training)
    extractor = FeatureExtractor(config=config)

    # Extract features
    logger.info("Extracting features for {} queries...", len(df))
    feature_vectors: list[list[float]] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Feature extraction"):
        features = extractor.extract(row["query"])
        feature_vectors.append(features.to_vector())

    X = np.array(feature_vectors, dtype=np.float32)
    y = np.array(df["label_idx"].values, dtype=np.int32)

    logger.info("Feature matrix: {} | Labels: {}", X.shape, np.bincount(y))

    # Train
    model = RouterModel(config)
    report = model.train(X, y)

    # Save model
    save_dir = Path(config["data"]["router_training_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save()

    # Save training data
    training_data_path = save_dir / "training_data.csv"
    df.to_csv(training_data_path, index=False, encoding="utf-8")
    logger.info("Training data saved to {}", training_data_path)

    # Print results
    print("\n" + "=" * 60)
    print("ROUTER TRAINING RESULTS")
    print("=" * 60)
    print(f"Validation Accuracy: {report.accuracy:.4f}")
    if report.cv_scores:
        print(f"CV Accuracy (5-fold): {np.mean(report.cv_scores):.4f} ± {np.std(report.cv_scores):.4f}")
    if report.f1_per_class:
        print("\nF1 per class:")
        for cls, f1 in report.f1_per_class.items():
            print(f"  {cls:10s}: {f1:.4f}")
    if report.confusion_mat:
        print("\nConfusion Matrix:")
        labels = RouterModel.CLASS_LABELS
        header = "          " + "  ".join(f"{l:>8s}" for l in labels)
        print(header)
        for i, row in enumerate(report.confusion_mat):
            row_str = f"  {labels[i]:8s}" + "  ".join(f"{v:>8d}" for v in row)
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
    print("=" * 60)

    return report
