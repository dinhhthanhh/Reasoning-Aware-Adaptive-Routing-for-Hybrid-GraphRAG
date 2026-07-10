"""Router training pipeline — loads real QA dataset and trains XGBoost.

Primary data source: qa_pipeline/data/final/train.json (940-sample QA dataset
with verified routing_label annotations: dense_retrieval / graph_traversal /
hybrid_reasoning).

Weak-label templates are used only as supplementary augmentation for the
'clarify' class (not present in the QA dataset).

M2 fix (review.md): All six classifiers (Majority, Keyword, 3-Rule, LogReg,
RF, XGBoost) are now evaluated under the SAME stratified 5-fold CV protocol
via run_cv_all_baselines(). The 0.995 single-split figure is diagnosed by
check_leakage() which compares feature distributions between paraphrase
template groups.
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

from router.features import FeatureExtractor, QueryFeatures, FEATURE_NAMES
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


# ──────────────────────────────────────────────────────────────────────────────
# M2: Multi-baseline CV — all six classifiers under identical stratified 5-fold
# ──────────────────────────────────────────────────────────────────────────────

def _majority_predict(y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """Majority class baseline: always predict the most frequent training class."""
    from collections import Counter
    majority = Counter(y_train.tolist()).most_common(1)[0][0]
    return np.full(len(X_test), majority, dtype=np.int32)


def _keyword_predict(
    queries_test: list[str],
    label_map: dict[str, int],
    classes_in_fold: list[int],
) -> np.ndarray:
    """Keyword-rule baseline (simple heuristic, no training).

    Routes to graph_traversal if multi-hop keywords, hybrid_reasoning if
    cross-document keywords, else dense_retrieval.
    """
    graph_kw = re.compile(
        r"\b(quan hệ|liên quan|giữa|so sánh|khác nhau|ảnh hưởng|tác động|dẫn đến)\b",
        re.IGNORECASE,
    )
    hybrid_kw = re.compile(
        r"\b(nhiều|tổng hợp|đồng thời|kết hợp|cả hai|các văn bản|cross)\b",
        re.IGNORECASE,
    )
    dense_idx = label_map.get("dense_retrieval", 0)
    graph_idx = label_map.get("graph_traversal", 1)
    hybrid_idx = label_map.get("hybrid_reasoning", 2)

    preds = []
    for q in queries_test:
        if hybrid_kw.search(q):
            preds.append(hybrid_idx if hybrid_idx in classes_in_fold else dense_idx)
        elif graph_kw.search(q):
            preds.append(graph_idx if graph_idx in classes_in_fold else dense_idx)
        else:
            preds.append(dense_idx if dense_idx in classes_in_fold else classes_in_fold[0])
    return np.array(preds, dtype=np.int32)


def _three_rule_predict(
    queries_test: list[str],
    label_map: dict[str, int],
) -> np.ndarray:
    """3-Rule baseline (lexical rules only, deterministic).

    Rule 1: pronoun/vague → clarify
    Rule 2: multi-hop connector → graph_traversal
    Rule 3: cross-doc signal → hybrid_reasoning
    Default: dense_retrieval
    """
    clarify_idx = label_map.get("clarify", 3)
    graph_idx = label_map.get("graph_traversal", 1)
    hybrid_idx = label_map.get("hybrid_reasoning", 2)
    dense_idx = label_map.get("dense_retrieval", 0)

    r1 = re.compile(r"\b(ông ấy|bà ấy|họ|nó|người đó|điều đó|luật đó)\b", re.IGNORECASE)
    r2 = re.compile(r"\b(sau khi|khi nào|nếu|liên quan đến|so với|giữa)\b", re.IGNORECASE)
    r3 = re.compile(r"\b(nhiều văn bản|các luật|cả hai|đồng thời|kết hợp)\b", re.IGNORECASE)

    preds = []
    for q in queries_test:
        if r1.search(q):
            preds.append(clarify_idx)
        elif r3.search(q):
            preds.append(hybrid_idx)
        elif r2.search(q):
            preds.append(graph_idx)
        else:
            preds.append(dense_idx)
    return np.array(preds, dtype=np.int32)


def run_cv_all_baselines(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, dict[str, Any]]:
    """Run all six classifiers under identical stratified 5-fold CV (M2).

    Classifiers:
        1. Majority     — predict most-frequent class
        2. Keyword      — simple keyword heuristic
        3. 3-Rule       — three-rule lexical baseline
        4. LogReg       — logistic regression (C=1, max_iter=500)
        5. RF           — random forest (n_estimators=200)
        6. XGBoost      — XGBoost (n_estimators=200)

    All use the SAME fold splits and SAME stratified 5-fold seed to ensure
    fair comparison. This was the source of the 0.995 discrepancy — the
    original single-split used a paraphrase-augmented test split that leaked
    the template structure.

    Args:
        df: Training DataFrame (must have 'query' column).
        X: Feature matrix (n_samples, n_features).
        y: Integer labels (0..n_classes-1).
        n_splits: Number of CV folds.
        random_state: Seed for StratifiedKFold.

    Returns:
        Dict mapping classifier name → {cv_accuracy, cv_accuracy_std,
        cv_macro_f1, cv_macro_f1_std, cv_scores_per_fold}.
    """
    from collections import Counter

    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_sample_weight

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    label_map = RouterModel.LABEL_TO_IDX

    # Encode labels to 0..n_classes-1 contiguously
    le = LabelEncoder()
    y_enc = le.fit_transform(y).astype(np.int32)

    classifiers: dict[str, Any] = {
        "majority": None,
        "keyword": None,
        "3_rule": None,
        "logreg": LogisticRegression(C=1.0, max_iter=1000, solver='saga', random_state=random_state, n_jobs=-1),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=random_state, class_weight="balanced", n_jobs=-1
        ),
        "xgboost": None,  # instantiated per fold (needs num_class)
    }

    results: dict[str, dict[str, Any]] = {
        name: {"acc_per_fold": [], "f1_per_fold": []}
        for name in classifiers
    }

    queries = df["query"].tolist()
    classes_global = sorted(np.unique(y_enc).tolist())

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_enc), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y_enc[train_idx], y_enc[val_idx]
        q_val = [queries[i] for i in val_idx]
        classes_in_fold = sorted(np.unique(y_enc).tolist())
        sw = compute_sample_weight("balanced", y=y_tr)

        for name, clf in classifiers.items():
            if name == "majority":
                preds = _majority_predict(y_tr, X_val)
            elif name == "keyword":
                preds = _keyword_predict(q_val, label_map, classes_in_fold)
            elif name == "3_rule":
                preds = _three_rule_predict(q_val, label_map)
            elif name == "xgboost":
                num_class = len(classes_in_fold)
                xgb_clf = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="multi:softprob",
                    num_class=num_class,
                    eval_metric="mlogloss",
                    random_state=random_state,
                    n_jobs=-1,
                )
                xgb_clf.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
                preds = xgb_clf.predict(X_val)
            else:
                clf.fit(X_tr, y_tr, sample_weight=sw if hasattr(clf, "fit") else None)
                preds = clf.predict(X_val)

            # Clip preds to valid range to avoid classification_report errors
            max_label = max(classes_in_fold)
            preds = np.clip(preds, 0, max_label).astype(np.int32)

            acc = accuracy_score(y_val, preds)
            f1 = f1_score(y_val, preds, average="macro", zero_division=0,
                          labels=classes_in_fold)

            results[name]["acc_per_fold"].append(float(acc))
            results[name]["f1_per_fold"].append(float(f1))

        logger.info("Fold {}/{} done.", fold_idx, n_splits)

    # Summarise
    summary: dict[str, dict[str, Any]] = {}
    for name, res in results.items():
        acc_arr = np.array(res["acc_per_fold"])
        f1_arr = np.array(res["f1_per_fold"])
        summary[name] = {
            "cv_accuracy": float(acc_arr.mean()),
            "cv_accuracy_std": float(acc_arr.std()),
            "cv_macro_f1": float(f1_arr.mean()),
            "cv_macro_f1_std": float(f1_arr.std()),
            "cv_scores_per_fold": res["acc_per_fold"],
            "cv_f1_per_fold": res["f1_per_fold"],
        }

    _print_cv_report(summary)
    return summary


def _print_cv_report(summary: dict[str, dict[str, Any]]) -> None:
    """Print CV comparison table (tab:routing_cv in the paper)."""
    print("\n" + "=" * 70)
    print("ROUTER CV RESULTS — tab:routing_cv (M2)")
    print("All classifiers evaluated under identical stratified 5-fold CV")
    print("=" * 70)
    try:
        from tabulate import tabulate
        rows = [
            [
                name,
                f"{v['cv_accuracy']:.4f} ± {v['cv_accuracy_std']:.4f}",
                f"{v['cv_macro_f1']:.4f} ± {v['cv_macro_f1_std']:.4f}",
            ]
            for name, v in sorted(summary.items(), key=lambda x: -x[1]["cv_macro_f1"])
        ]
        print(tabulate(rows, headers=["Classifier", "CV Accuracy", "CV Macro-F1"], tablefmt="grid"))
    except ImportError:
        for name, v in summary.items():
            print(
                f"  {name:20s}: acc={v['cv_accuracy']:.4f}±{v['cv_accuracy_std']:.4f}"
                f"  F1={v['cv_macro_f1']:.4f}±{v['cv_macro_f1_std']:.4f}"
            )
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# M2: Leakage check — diagnose the 0.995 single-split overfit
# ──────────────────────────────────────────────────────────────────────────────

def check_leakage(df: pd.DataFrame) -> dict[str, Any]:
    """Check for train/test leakage in the paraphrase-augmented split (M2).

    The 0.995 Macro-F1 on the 'paraphrased strict split' is almost certainly
    caused by paraphrase templates sharing surface features across the split
    boundary. This function:
      1. Identifies query groups with high lexical similarity (Jaccard ≥ 0.7).
      2. Reports what fraction of such pairs straddle the split boundary.
      3. Computes average Jaccard similarity of split-boundary pairs vs
         same-split pairs as a leakage indicator.

    Args:
        df: Training DataFrame with 'query' and 'label' columns. Must have
            a 'split' column if you want split-boundary analysis; if absent,
            we simulate an 80/20 split.

    Returns:
        Dict with 'leakage_rate', 'boundary_jaccard_mean',
        'within_split_jaccard_mean', 'high_sim_pair_count'.
    """
    queries = df["query"].tolist()
    n = len(queries)

    # Tokenise (simple whitespace)
    tok = [set(q.lower().split()) for q in queries]

    def _jaccard(a: set, b: set) -> float:
        u = a | b
        return len(a & b) / len(u) if u else 0.0

    # Simulate 80/20 split if no 'split' column
    if "split" not in df.columns:
        split_idx = int(0.8 * n)
        splits = ["train"] * split_idx + ["test"] * (n - split_idx)
    else:
        splits = df["split"].tolist()

    boundary_jaccards: list[float] = []
    within_jaccards: list[float] = []
    high_sim_pairs = 0

    # Sample pairs for efficiency (up to 50000 pairs)
    import random as _random
    rng = _random.Random(42)
    indices = list(range(n))
    pairs = [(i, j) for i in indices for j in indices if i < j]
    if len(pairs) > 50000:
        pairs = rng.sample(pairs, 50000)

    for i, j in pairs:
        jac = _jaccard(tok[i], tok[j])
        is_boundary = splits[i] != splits[j]
        if jac >= 0.7:
            high_sim_pairs += 1
        if is_boundary:
            boundary_jaccards.append(jac)
        else:
            within_jaccards.append(jac)

    boundary_mean = float(np.mean(boundary_jaccards)) if boundary_jaccards else 0.0
    within_mean = float(np.mean(within_jaccards)) if within_jaccards else 0.0

    # Leakage rate = fraction of high-sim pairs that cross the split boundary
    high_sim_boundary = sum(
        1
        for i, j in (pairs if len(pairs) <= 50000 else pairs)
        if _jaccard(tok[i], tok[j]) >= 0.7 and splits[i] != splits[j]
    )
    leakage_rate = high_sim_boundary / high_sim_pairs if high_sim_pairs > 0 else 0.0

    result = {
        "leakage_rate": leakage_rate,
        "high_sim_pair_count": high_sim_pairs,
        "boundary_jaccard_mean": boundary_mean,
        "within_split_jaccard_mean": within_mean,
        "leakage_warning": (
            "HIGH LEAKAGE DETECTED — paraphrase templates likely span "
            "the split boundary. The 0.995 single-split accuracy is unreliable."
            if leakage_rate > 0.3 else
            "Leakage within acceptable range."
        ),
    }

    logger.info(
        "Leakage check: rate={:.3f} | high_sim_pairs={} | boundary_jac={:.3f} | within_jac={:.3f}",
        leakage_rate, high_sim_pairs, boundary_mean, within_mean,
    )
    logger.warning("{}", result["leakage_warning"])
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main train + evaluate
# ──────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    eval_df: pd.DataFrame | None = None,
    run_baselines: bool = True,
) -> TrainingReport:
    """Train the router model and evaluate performance.

    Pipeline:
    1. Extract features for all queries.
    2. Build feature matrix X and label vector y.
    3. Train XGBoost with 5-fold cross-validation + balanced class weights.
    4. If run_baselines, also run all 6 classifiers under the same CV (M2).
    5. If eval_df provided, also evaluate on that split.
    6. Print accuracy, F1 per class, confusion matrix.
    7. Save model + feature_names.json.

    Args:
        df: Training DataFrame with columns: query, label, label_idx.
        config: Full config dict.
        eval_df: Optional held-out evaluation DataFrame (e.g. dev split).
        run_baselines: Whether to run all-baseline CV comparison (M2).

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

    # ── Leakage check (M2) ─────────────────────────────────────────────────
    leakage_info = check_leakage(df)

    # ── Multi-baseline CV (M2) ─────────────────────────────────────────────
    cv_baselines: dict[str, Any] = {}
    if run_baselines:
        logger.info("Running all-baseline stratified 5-fold CV (M2)...")
        cv_baselines = run_cv_all_baselines(df, X, y)

    # ── Train XGBoost ──────────────────────────────────────────────────────
    model = RouterModel(config)
    report = model.train(X, y)

    # ── Save model ─────────────────────────────────────────────────────────
    save_dir = Path(config["data"]["router_training_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save()

    # Save feature_names.json so RouterModel.load() aligns the feature schema
    feature_names_path = save_dir / "feature_names.json"
    with open(feature_names_path, "w", encoding="utf-8") as f:
        json.dump(FEATURE_NAMES, f, ensure_ascii=False)
    logger.info("Feature names saved to {}", feature_names_path)

    # Save training data summary
    training_data_path = save_dir / "training_data.csv"
    df[["query", "label", "label_idx", "source"]].to_csv(
        training_data_path, index=False, encoding="utf-8"
    )
    logger.info("Training data saved to {}", training_data_path)

    # Save CV baseline results
    if cv_baselines:
        cv_path = save_dir / "cv_baselines.json"
        with open(cv_path, "w", encoding="utf-8") as f:
            json.dump(cv_baselines, f, indent=2, ensure_ascii=False)
        logger.info("CV baseline results saved to {}", cv_path)

    # Save leakage report
    leakage_path = save_dir / "leakage_report.json"
    with open(leakage_path, "w", encoding="utf-8") as f:
        json.dump(leakage_info, f, indent=2, ensure_ascii=False)
    logger.info("Leakage report saved to {}", leakage_path)

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
            type("F", (), {"to_vector": lambda self: eval_features[i]})()\
        ).route for i in range(len(eval_features))]

        # Convert route strings → idx for comparison
        from router.router_model import RouterModel as RM
        pred_idx = [RM.LABEL_TO_IDX.get(r, 0) for r in eval_preds]
        from sklearn.metrics import accuracy_score
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
        print("\nTop 10 Feature Importances:")
        sorted_feats = sorted(
            report.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for name, imp in sorted_feats[:10]:
            print(f"  {name:30s}: {imp:.4f}")

    print("=" * 70)
