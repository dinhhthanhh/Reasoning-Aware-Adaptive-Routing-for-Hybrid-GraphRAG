#!/usr/bin/env python3
"""Run all routing baselines on the strict legal QA split.

Baselines:
  1. MajorityRoute         — always predicts most frequent train class
  2. KeywordRuleRouter     — regex-based heuristic (no gold evidence)
  3. MetadataRuleRouter    — oracle-style (uses hop_count, is_cross_doc)
  4. LogisticRegression    — sklearn, same feature vector as Stage1Model
  5. RandomForest          — sklearn, same feature vector as Stage1Model
  6. PhoBERTClassifier     — vinai/phobert-base-v2 sequence classification
  7. Stage1Model           — existing XGBoost router (re-verified)

Anti-leakage:  LR, RF, PhoBERT and Stage1 use only question text.
               MetadataRuleRouter is explicitly oracle-style.
               Test set is NEVER used for tuning.

Output: artifacts/routing_baselines/
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "qa_pipeline" / "data" / "legal_strict"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "routing_baselines"
LABEL_ORDER = ["dense_retrieval", "graph_traversal", "hybrid_reasoning"]
LABEL2IDX = {l: i for i, l in enumerate(LABEL_ORDER)}

# Expected split sizes
EXPECTED_TRAIN = {"dense_retrieval": 356, "graph_traversal": 54, "hybrid_reasoning": 36}
EXPECTED_DEV = {"dense_retrieval": 50, "graph_traversal": 20, "hybrid_reasoning": 10}
EXPECTED_TEST = {"dense_retrieval": 300, "graph_traversal": 150, "hybrid_reasoning": 150}


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_split(split: str) -> list[dict]:
    path = DATA_DIR / f"{split}.json"
    if not path.exists():
        print(f"FATAL: {path} not found", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def verify_splits() -> tuple[list[dict], list[dict], list[dict]]:
    """Load and verify strict split counts."""
    train = load_split("train")
    dev = load_split("dev")
    test = load_split("test")

    for name, data, expected in [
        ("train", train, EXPECTED_TRAIN),
        ("dev", dev, EXPECTED_DEV),
        ("test", test, EXPECTED_TEST),
    ]:
        actual = dict(Counter(d["routing_label"] for d in data))
        if actual != expected:
            print(f"FATAL: {name} split mismatch!", file=sys.stderr)
            print(f"  Expected: {expected}", file=sys.stderr)
            print(f"  Actual:   {actual}", file=sys.stderr)
            sys.exit(1)
        print(f"  ✓ {name}: {len(data)} samples — {actual}")

    return train, dev, test


# ═══════════════════════════════════════════════════════════════════════════
# Feature extraction (reuse Stage1Model pipeline)
# ═══════════════════════════════════════════════════════════════════════════

def extract_features(questions: list[str]) -> np.ndarray:
    """Extract 28-dim feature vectors using the project's FeatureExtractor.

    Uses ONLY question text — no gold evidence, no answer, no metadata.
    """
    import yaml
    from router.features import FeatureExtractor

    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    extractor = FeatureExtractor(config=config)
    vectors = []
    for q in questions:
        feat = extractor.extract(q)
        vectors.append(feat.to_vector())
    return np.array(vectors, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Metric computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    baseline_name: str,
) -> dict:
    """Compute all required metrics with fixed label order."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=LABEL_ORDER, average="weighted", zero_division=0)

    report = classification_report(
        y_true, y_pred,
        labels=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER).tolist()

    per_class = {}
    for label in LABEL_ORDER:
        per_class[label] = {
            "precision": round(report[label]["precision"], 4),
            "recall": round(report[label]["recall"], 4),
            "f1": round(report[label]["f1-score"], 4),
            "support": int(report[label]["support"]),
        }

    pred_counts = dict(Counter(y_pred))

    return {
        "baseline": baseline_name,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "confusion_matrix_labels": LABEL_ORDER,
        "prediction_counts": pred_counts,
        "num_samples": len(y_true),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Baseline implementations
# ═══════════════════════════════════════════════════════════════════════════

def run_majority_route(train: list[dict], test: list[dict]) -> tuple[list[str], dict]:
    """Always predict the most common class in training set."""
    train_labels = [d["routing_label"] for d in train]
    majority_class = Counter(train_labels).most_common(1)[0][0]
    print(f"  Majority class: {majority_class} ({Counter(train_labels)[majority_class]}/{len(train_labels)})")

    test_labels = [d["routing_label"] for d in test]
    preds = [majority_class] * len(test)
    metrics = compute_metrics(test_labels, preds, "MajorityRoute")
    return preds, metrics


def run_keyword_rule_router(test: list[dict]) -> tuple[list[str], dict]:
    """Rule-based routing using regex features from question text only.

    Mirrors RouterModel._rule_based_fallback() WITHOUT clarify path.
    Rules designed without looking at test set.
    """
    import yaml
    from router.features import FeatureExtractor

    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    extractor = FeatureExtractor(config=config)
    test_labels = [d["routing_label"] for d in test]
    preds = []

    for d in test:
        q = d["question"].strip()
        feat = extractor.extract(q)

        if feat.cross_doc_signals:
            preds.append("hybrid_reasoning")
        elif (feat.multi_hop_score > 0.5
              or feat.has_comparison
              or feat.legal_reference_count > 1
              or feat.graph_keyword_count > 1):
            preds.append("graph_traversal")
        else:
            preds.append("dense_retrieval")

    metrics = compute_metrics(test_labels, preds, "KeywordRuleRouter")
    return preds, metrics


def run_metadata_rule_router(test: list[dict]) -> tuple[list[str], dict]:
    """Oracle-style router using hop_count and is_cross_doc from gold metadata.

    NOT a deployable baseline — marked as oracle diagnostic.
    """
    test_labels = [d["routing_label"] for d in test]
    preds = []
    for d in test:
        if d.get("is_cross_doc", False):
            preds.append("hybrid_reasoning")
        elif d.get("hop_count", 1) >= 2:
            preds.append("graph_traversal")
        else:
            preds.append("dense_retrieval")

    metrics = compute_metrics(test_labels, preds, "MetadataRuleRouter (oracle)")
    metrics["is_oracle"] = True
    return preds, metrics


def run_logistic_regression(
    train: list[dict],
    dev: list[dict],
    test: list[dict],
    seeds: list[int] = [42],
) -> tuple[list[str], dict, dict]:
    """Multinomial logistic regression with feature vector from FeatureExtractor.

    Tuning: C selected by dev Macro-F1.
    Returns: (test_preds_seed42, test_metrics, info_dict)
    """
    print("  Extracting features (train)...")
    train_q = [d["question"].strip() for d in train]
    X_train = extract_features(train_q)
    y_train = [d["routing_label"] for d in train]

    print("  Extracting features (dev)...")
    dev_q = [d["question"].strip() for d in dev]
    X_dev = extract_features(dev_q)
    y_dev = [d["routing_label"] for d in dev]

    print("  Extracting features (test)...")
    test_q = [d["question"].strip() for d in test]
    X_test = extract_features(test_q)
    y_test = [d["routing_label"] for d in test]

    C_values = [0.01, 0.1, 1.0, 10.0]
    best_C = None
    best_dev_f1 = -1.0
    dev_results = {}

    for C in C_values:
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=C, multi_class="multinomial", solver="lbfgs",
                max_iter=1000, random_state=42,
            )),
        ])
        pipe.fit(X_train, y_train)
        dev_preds = pipe.predict(X_dev)
        dev_f1 = f1_score(y_dev, dev_preds, labels=LABEL_ORDER, average="macro", zero_division=0)
        dev_results[C] = round(dev_f1, 4)
        print(f"    C={C:6.2f}  dev_macro_f1={dev_f1:.4f}")
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_C = C

    print(f"  → Best C={best_C} (dev Macro-F1={best_dev_f1:.4f})")

    # Final training with best C, evaluate on test
    all_seed_results = []
    final_preds_42 = None

    for seed in seeds:
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=best_C, multi_class="multinomial", solver="lbfgs",
                max_iter=1000, random_state=seed,
            )),
        ])
        pipe.fit(X_train, y_train)
        test_preds = pipe.predict(X_test).tolist()
        metrics = compute_metrics(y_test, test_preds, f"LogisticRegression (seed={seed})")
        all_seed_results.append(metrics)
        print(f"    seed={seed}  acc={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}")
        if seed == 42:
            final_preds_42 = test_preds

    # Primary result: seed 42
    primary = all_seed_results[0]
    primary["baseline"] = "LogisticRegression"

    info = {
        "best_C": best_C,
        "dev_results": dev_results,
        "seeds_run": seeds,
        "feature_count": int(X_train.shape[1]),
        "feature_names": _get_feature_names(),
    }

    if len(all_seed_results) > 1:
        accs = [r["accuracy"] for r in all_seed_results]
        f1s = [r["macro_f1"] for r in all_seed_results]
        wf1s = [r["weighted_f1"] for r in all_seed_results]
        info["multi_seed"] = {
            "accuracy_mean": round(float(np.mean(accs)), 4),
            "accuracy_std": round(float(np.std(accs, ddof=1)), 4),
            "macro_f1_mean": round(float(np.mean(f1s)), 4),
            "macro_f1_std": round(float(np.std(f1s, ddof=1)), 4),
            "weighted_f1_mean": round(float(np.mean(wf1s)), 4),
            "weighted_f1_std": round(float(np.std(wf1s, ddof=1)), 4),
        }

    return final_preds_42, primary, info


def run_random_forest(
    train: list[dict],
    dev: list[dict],
    test: list[dict],
    seeds: list[int] = [42],
) -> tuple[list[str], dict, dict]:
    """Random Forest with hyperparameter selection by dev Macro-F1."""
    print("  Extracting features (train)...")
    train_q = [d["question"].strip() for d in train]
    X_train = extract_features(train_q)
    y_train = [d["routing_label"] for d in train]

    print("  Extracting features (dev)...")
    dev_q = [d["question"].strip() for d in dev]
    X_dev = extract_features(dev_q)
    y_dev = [d["routing_label"] for d in dev]

    print("  Extracting features (test)...")
    test_q = [d["question"].strip() for d in test]
    X_test = extract_features(test_q)
    y_test = [d["routing_label"] for d in test]

    # Impute + scale (fit on train only)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_proc = scaler.fit_transform(imputer.fit_transform(X_train))
    X_dev_proc = scaler.transform(imputer.transform(X_dev))
    X_test_proc = scaler.transform(imputer.transform(X_test))

    # Grid search on dev
    param_grid = {
        "n_estimators": [200, 500],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 5],
        "class_weight": [None, "balanced"],
    }

    best_params = None
    best_dev_f1 = -1.0
    dev_results = []
    total_configs = (len(param_grid["n_estimators"]) *
                     len(param_grid["max_depth"]) *
                     len(param_grid["min_samples_leaf"]) *
                     len(param_grid["class_weight"]))
    print(f"  Searching {total_configs} configurations...")

    config_idx = 0
    for n_est in param_grid["n_estimators"]:
        for md in param_grid["max_depth"]:
            for msl in param_grid["min_samples_leaf"]:
                for cw in param_grid["class_weight"]:
                    config_idx += 1
                    clf = RandomForestClassifier(
                        n_estimators=n_est,
                        max_depth=md,
                        min_samples_leaf=msl,
                        class_weight=cw,
                        random_state=42,
                        n_jobs=-1,
                    )
                    clf.fit(X_train_proc, y_train)
                    dev_preds = clf.predict(X_dev_proc)
                    dev_f1 = f1_score(y_dev, dev_preds, labels=LABEL_ORDER, average="macro", zero_division=0)

                    params = {
                        "n_estimators": n_est,
                        "max_depth": md,
                        "min_samples_leaf": msl,
                        "class_weight": cw,
                    }
                    dev_results.append({**params, "dev_macro_f1": round(dev_f1, 4)})

                    if dev_f1 > best_dev_f1:
                        best_dev_f1 = dev_f1
                        best_params = params

                    if config_idx % 12 == 0 or config_idx == total_configs:
                        print(f"    [{config_idx}/{total_configs}] current best dev_f1={best_dev_f1:.4f}")

    print(f"  → Best params: {best_params} (dev Macro-F1={best_dev_f1:.4f})")

    # Final evaluation with best params across seeds
    all_seed_results = []
    final_preds_42 = None

    for seed in seeds:
        clf = RandomForestClassifier(
            **best_params,
            random_state=seed,
            n_jobs=-1,
        )
        clf.fit(X_train_proc, y_train)
        test_preds = clf.predict(X_test_proc).tolist()
        metrics = compute_metrics(y_test, test_preds, f"RandomForest (seed={seed})")
        all_seed_results.append(metrics)
        print(f"    seed={seed}  acc={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}")
        if seed == 42:
            final_preds_42 = test_preds

    primary = all_seed_results[0]
    primary["baseline"] = "RandomForest"

    info = {
        "best_params": best_params,
        "dev_results_top5": sorted(dev_results, key=lambda x: x["dev_macro_f1"], reverse=True)[:5],
        "seeds_run": seeds,
        "feature_count": int(X_train.shape[1]),
        "feature_names": _get_feature_names(),
    }

    if len(all_seed_results) > 1:
        accs = [r["accuracy"] for r in all_seed_results]
        f1s = [r["macro_f1"] for r in all_seed_results]
        wf1s = [r["weighted_f1"] for r in all_seed_results]
        info["multi_seed"] = {
            "accuracy_mean": round(float(np.mean(accs)), 4),
            "accuracy_std": round(float(np.std(accs, ddof=1)), 4),
            "macro_f1_mean": round(float(np.mean(f1s)), 4),
            "macro_f1_std": round(float(np.std(f1s, ddof=1)), 4),
            "weighted_f1_mean": round(float(np.mean(wf1s)), 4),
            "weighted_f1_std": round(float(np.std(wf1s, ddof=1)), 4),
        }

    return final_preds_42, primary, info


def run_stage1_model(test: list[dict]) -> tuple[list[str], dict]:
    """Re-verify existing Stage1Model (XGBoost) on test with 3-class labels."""
    import pickle
    import yaml
    from router.features import FeatureExtractor, QueryFeatures

    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_path = PROJECT_ROOT / config["router"]["stage1"]["model_path"]
    if not model_path.exists():
        print(f"  WARNING: Stage1Model not found at {model_path}", file=sys.stderr)
        return [], {}

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    extractor = FeatureExtractor(config=config)

    # Label mapping from Stage1Model
    STAGE1_LABELS = ["dense_retrieval", "graph_traversal", "hybrid_reasoning", "clarify"]

    test_labels = [d["routing_label"] for d in test]
    preds = []

    for d in test:
        q = d["question"].strip()
        feat = extractor.extract(q)
        feature_vec = np.array([feat.to_vector()], dtype=np.float32)
        probas = model.predict_proba(feature_vec)[0]

        # Get model classes
        model_classes = getattr(model, "classes_", None)
        predicted_idx = int(np.argmax(probas))

        if model_classes is not None and len(model_classes) == len(probas):
            route = STAGE1_LABELS[int(model_classes[predicted_idx])]
        else:
            route = STAGE1_LABELS[min(predicted_idx, len(STAGE1_LABELS) - 1)]

        # Map clarify → dense_retrieval for evaluation (no clarify in test)
        if route == "clarify":
            route = "dense_retrieval"

        preds.append(route)

    metrics = compute_metrics(test_labels, preds, "Stage1Model (XGBoost)")
    return preds, metrics


def run_phobert_classifier(
    train: list[dict],
    dev: list[dict],
    test: list[dict],
    seed: int = 42,
) -> tuple[list[str], dict, dict]:
    """Train PhoBERT sequence classification on question text only."""
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            get_linear_schedule_with_warmup,
        )
    except ImportError as e:
        print(f"  SKIP: PhoBERT not available — {e}", file=sys.stderr)
        return [], {}, {"error": str(e)}

    try:
        if not torch.cuda.is_available():
            print("  WARNING: No GPU detected. PhoBERT training will be slow on CPU.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {device}")

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        MODEL_NAME = "vinai/phobert-base-v2"

        # --- Tokenizer ---
        print(f"  Loading tokenizer: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # --- Determine max_length from train P95 ---
        train_questions = [d["question"].strip() for d in train]
        train_lengths = [len(tokenizer.encode(q, add_special_tokens=True)) for q in train_questions]
        p95_length = int(np.percentile(train_lengths, 95))
        max_length = min(p95_length, 256)
        print(f"  Token lengths P95={p95_length}, using max_length={max_length}")

        # --- Dataset ---
        class RouteDataset(Dataset):
            def __init__(self, items: list[dict]):
                self.questions = [d["question"].strip() for d in items]
                self.labels = [LABEL2IDX[d["routing_label"]] for d in items]

            def __len__(self):
                return len(self.questions)

            def __getitem__(self, idx):
                enc = tokenizer(
                    self.questions[idx],
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                return {
                    "input_ids": enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0),
                    "label": torch.tensor(self.labels[idx], dtype=torch.long),
                }

        train_ds = RouteDataset(train)
        dev_ds = RouteDataset(dev)
        test_ds = RouteDataset(test)

        batch_size = 16
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # --- Model ---
        print(f"  Loading model: {MODEL_NAME}")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=3
        ).to(device)

        lr = 2e-5
        max_epochs = 20
        patience = 3
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * max_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
        )

        # --- Training with early stopping on dev Macro-F1 ---
        best_dev_f1 = -1.0
        best_epoch = 0
        patience_counter = 0
        best_state = None

        print(f"  Training: lr={lr}, batch_size={batch_size}, max_epochs={max_epochs}, patience={patience}")

        for epoch in range(1, max_epochs + 1):
            model.train()
            total_loss = 0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # Evaluate on dev
            model.eval()
            dev_preds_all = []
            dev_labels_all = []
            with torch.no_grad():
                for batch in dev_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                    preds = torch.argmax(logits, dim=-1).cpu().tolist()
                    dev_preds_all.extend(preds)
                    dev_labels_all.extend(batch["label"].tolist())

            dev_pred_labels = [LABEL_ORDER[i] for i in dev_preds_all]
            dev_true_labels = [LABEL_ORDER[i] for i in dev_labels_all]
            dev_f1 = f1_score(dev_true_labels, dev_pred_labels, labels=LABEL_ORDER, average="macro", zero_division=0)
            dev_acc = accuracy_score(dev_true_labels, dev_pred_labels)

            print(f"    Epoch {epoch:2d}: loss={avg_loss:.4f}  dev_acc={dev_acc:.4f}  dev_macro_f1={dev_f1:.4f}")

            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_epoch = epoch
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        # Load best checkpoint
        if best_state:
            model.load_state_dict(best_state)
            model.to(device)
        print(f"  Best epoch: {best_epoch} (dev Macro-F1={best_dev_f1:.4f})")

        actual_epochs = epoch

        # --- Test evaluation ---
        model.eval()
        test_preds_all = []
        test_labels_all = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                preds = torch.argmax(logits, dim=-1).cpu().tolist()
                test_preds_all.extend(preds)
                test_labels_all.extend(batch["label"].tolist())

        test_pred_labels = [LABEL_ORDER[i] for i in test_preds_all]
        test_true_labels = [LABEL_ORDER[i] for i in test_labels_all]

        metrics = compute_metrics(test_true_labels, test_pred_labels, "PhoBERTClassifier")

        info = {
            "model_name": MODEL_NAME,
            "max_length": max_length,
            "p95_token_length": p95_length,
            "batch_size": batch_size,
            "learning_rate": lr,
            "actual_epochs": actual_epochs,
            "best_epoch": best_epoch,
            "best_dev_macro_f1": round(best_dev_f1, 4),
            "random_seed": seed,
            "device": str(device),
        }

        return test_pred_labels, metrics, info

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"  SKIP: PhoBERT failed — {error_msg}", file=sys.stderr)
        return [], {}, {"error": error_msg}


def _get_feature_names() -> list[str]:
    from router.features import QueryFeatures
    return QueryFeatures.feature_names()


# ═══════════════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(
    all_results: dict[str, dict],
    all_infos: dict[str, dict],
    timing: dict[str, float],
    dataset_files: list[str],
    split_counts: dict,
) -> str:
    """Generate the experiment_report.md content."""
    lines = []
    lines.append("# Routing Baselines — Experiment Report\n")
    lines.append(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Dataset files
    lines.append("## Dataset Files Used\n")
    for f in dataset_files:
        lines.append(f"- `{f}`")
    lines.append("")

    # Split counts
    lines.append("## Split Counts Verified\n")
    lines.append("| Split | Total | dense_retrieval | graph_traversal | hybrid_reasoning |")
    lines.append("|-------|-------|-----------------|-----------------|------------------|")
    for split_name, counts in split_counts.items():
        total = sum(counts.values())
        lines.append(f"| {split_name} | {total} | {counts.get('dense_retrieval',0)} | "
                     f"{counts.get('graph_traversal',0)} | {counts.get('hybrid_reasoning',0)} |")
    lines.append("")

    # Feature columns
    lines.append("## Feature Columns (LR, RF, Stage1Model)\n")
    feature_names = _get_feature_names()
    lines.append(f"**{len(feature_names)} features** extracted from question text only:\n")
    lines.append("```")
    for i in range(0, len(feature_names), 4):
        lines.append(", ".join(feature_names[i:i+4]))
    lines.append("```\n")

    # Excluded columns
    lines.append("## Columns Excluded (Anti-Leakage)\n")
    excluded = [
        "hop_count", "is_cross_doc", "relevant_articles", "supporting_facts",
        "gold_context", "article_key", "evidence", "law", "doc_number", "url",
        "answer", "difficulty", "has_reasoning", "question_type",
    ]
    lines.append(f"```\n{', '.join(excluded)}\n```\n")

    # Hyperparameters
    lines.append("## Hyperparameter Selection\n")
    for name, info in all_infos.items():
        if not info:
            continue
        lines.append(f"### {name}\n")
        if "best_C" in info:
            lines.append(f"- **Best C**: {info['best_C']}")
            lines.append(f"- **Dev results**: {info['dev_results']}")
        if "best_params" in info:
            lines.append(f"- **Best params**: {info['best_params']}")
            if "dev_results_top5" in info:
                lines.append(f"- **Top 5 dev configs**:")
                for dr in info["dev_results_top5"]:
                    lines.append(f"  - {dr}")
        if "model_name" in info:
            lines.append(f"- **Model**: {info.get('model_name')}")
            lines.append(f"- **max_length**: {info.get('max_length')}")
            lines.append(f"- **batch_size**: {info.get('batch_size')}")
            lines.append(f"- **learning_rate**: {info.get('learning_rate')}")
            lines.append(f"- **actual_epochs**: {info.get('actual_epochs')}")
            lines.append(f"- **best_epoch**: {info.get('best_epoch')}")
            lines.append(f"- **best_dev_macro_f1**: {info.get('best_dev_macro_f1')}")
            lines.append(f"- **seed**: {info.get('random_seed')}")
            lines.append(f"- **device**: {info.get('device')}")
        if "error" in info:
            lines.append(f"- **ERROR**: {info['error']}")
        if "multi_seed" in info:
            ms = info["multi_seed"]
            lines.append(f"- **Multi-seed**: acc={ms['accuracy_mean']}±{ms['accuracy_std']}, "
                         f"macro_f1={ms['macro_f1_mean']}±{ms['macro_f1_std']}, "
                         f"weighted_f1={ms['weighted_f1_mean']}±{ms['weighted_f1_std']}")
        lines.append("")

    # Final test results
    lines.append("## Final Strict-Test Results\n")
    lines.append("| Baseline | Accuracy | Macro-F1 | Weighted-F1 | Oracle? |")
    lines.append("|----------|----------|----------|-------------|---------|")
    for name, result in all_results.items():
        if not result:
            continue
        oracle = "✓" if result.get("is_oracle") else ""
        lines.append(f"| {result.get('baseline', name)} | {result.get('accuracy', 0):.4f} | "
                     f"{result.get('macro_f1', 0):.4f} | {result.get('weighted_f1', 0):.4f} | {oracle} |")
    lines.append("")

    # Timing
    lines.append("## Timing\n")
    for name, elapsed in timing.items():
        lines.append(f"- **{name}**: {elapsed:.1f}s")
    lines.append("")

    # Hardware
    lines.append("## Hardware\n")
    try:
        import torch
        if torch.cuda.is_available():
            lines.append(f"- GPU: {torch.cuda.get_device_name(0)}")
            lines.append(f"- CUDA: {torch.version.cuda}")
    except Exception:
        pass
    import platform
    lines.append(f"- OS: {platform.system()} {platform.release()}")
    lines.append(f"- Python: {platform.python_version()}")
    lines.append("")

    # Errors
    lines.append("## Errors / Baselines Not Run\n")
    errors_found = False
    for name, info in all_infos.items():
        if info and "error" in info:
            lines.append(f"- **{name}**: {info['error']}")
            errors_found = True
    for name, result in all_results.items():
        if not result:
            lines.append(f"- **{name}**: No results produced")
            errors_found = True
    if not errors_found:
        lines.append("None.\n")
    lines.append("")

    # LaTeX tables
    lines.append("## LaTeX Output\n")
    lines.append("### Classification Results\n")
    lines.append("```latex")
    deployable_order = [
        "MajorityRoute", "KeywordRuleRouter", "LogisticRegression",
        "RandomForest", "PhoBERTClassifier", "Stage1Model (XGBoost)",
    ]
    for bname in deployable_order:
        r = all_results.get(bname, {})
        if r:
            acc = r.get("accuracy", 0)
            mf1 = r.get("macro_f1", 0)
            wf1 = r.get("weighted_f1", 0)
            display = r.get("baseline", bname)
            lines.append(f"{display:25s} & {acc:.4f} & {mf1:.4f} & {wf1:.4f} \\\\")
    lines.append("```\n")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    print("=" * 70)
    print("ROUTING BASELINES EXPERIMENT")
    print("=" * 70)

    # --- Step 1: Verify splits ---
    print("\n[1/7] Verifying strict splits...")
    train, dev, test = verify_splits()
    split_counts = {
        "train": dict(Counter(d["routing_label"] for d in train)),
        "dev": dict(Counter(d["routing_label"] for d in dev)),
        "test": dict(Counter(d["routing_label"] for d in test)),
    }

    dataset_files = [
        str(DATA_DIR / "train.json"),
        str(DATA_DIR / "dev.json"),
        str(DATA_DIR / "test.json"),
    ]

    all_results: dict[str, dict] = {}
    all_preds: dict[str, list[str]] = {}
    all_infos: dict[str, dict] = {}
    timing: dict[str, float] = {}
    seeds = [42, 123, 456]

    test_labels = [d["routing_label"] for d in test]
    test_questions = [d["question"] for d in test]

    # --- Step 2: MajorityRoute ---
    print("\n[2/7] Running MajorityRoute...")
    t0 = time.time()
    preds, metrics = run_majority_route(train, test)
    timing["MajorityRoute"] = time.time() - t0
    all_results["MajorityRoute"] = metrics
    all_preds["MajorityRoute"] = preds
    all_infos["MajorityRoute"] = {}

    # --- Step 3: KeywordRuleRouter ---
    print("\n[3/7] Running KeywordRuleRouter...")
    t0 = time.time()
    preds, metrics = run_keyword_rule_router(test)
    timing["KeywordRuleRouter"] = time.time() - t0
    all_results["KeywordRuleRouter"] = metrics
    all_preds["KeywordRuleRouter"] = preds
    all_infos["KeywordRuleRouter"] = {}

    # --- MetadataRuleRouter (oracle) ---
    print("\n[3.5/7] Running MetadataRuleRouter (oracle)...")
    t0 = time.time()
    preds, metrics = run_metadata_rule_router(test)
    timing["MetadataRuleRouter"] = time.time() - t0
    all_results["MetadataRuleRouter (oracle)"] = metrics
    all_preds["MetadataRuleRouter"] = preds
    all_infos["MetadataRuleRouter"] = {"note": "Oracle-style: uses hop_count and is_cross_doc from gold metadata"}

    # --- Step 4: LogisticRegression ---
    print("\n[4/7] Running LogisticRegression...")
    t0 = time.time()
    preds, metrics, info = run_logistic_regression(train, dev, test, seeds=seeds)
    timing["LogisticRegression"] = time.time() - t0
    all_results["LogisticRegression"] = metrics
    all_preds["LogisticRegression"] = preds
    all_infos["LogisticRegression"] = info

    # --- Step 5: RandomForest ---
    print("\n[5/7] Running RandomForest...")
    t0 = time.time()
    preds, metrics, info = run_random_forest(train, dev, test, seeds=seeds)
    timing["RandomForest"] = time.time() - t0
    all_results["RandomForest"] = metrics
    all_preds["RandomForest"] = preds
    all_infos["RandomForest"] = info

    # --- Step 6: PhoBERTClassifier ---
    print("\n[6/7] Running PhoBERTClassifier...")
    t0 = time.time()
    preds, metrics, info = run_phobert_classifier(train, dev, test, seed=42)
    timing["PhoBERTClassifier"] = time.time() - t0
    if metrics:
        all_results["PhoBERTClassifier"] = metrics
        all_preds["PhoBERTClassifier"] = preds
    all_infos["PhoBERTClassifier"] = info

    # --- Step 7: Stage1Model ---
    print("\n[7/7] Re-verifying Stage1Model...")
    t0 = time.time()
    preds, metrics = run_stage1_model(test)
    timing["Stage1Model"] = time.time() - t0
    if metrics:
        all_results["Stage1Model (XGBoost)"] = metrics
        all_preds["Stage1Model"] = preds
    all_infos["Stage1Model"] = {}

    timing["Total"] = time.time() - total_start

    # ═══════════════════════════════════════════════════════════════════════
    # Save outputs
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    # baseline_results.json
    results_json = OUTPUT_DIR / "baseline_results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {results_json}")

    # baseline_results.csv
    rows = []
    for name, r in all_results.items():
        if r:
            rows.append({
                "baseline": r.get("baseline", name),
                "accuracy": r.get("accuracy", 0),
                "macro_f1": r.get("macro_f1", 0),
                "weighted_f1": r.get("weighted_f1", 0),
                "is_oracle": r.get("is_oracle", False),
            })
    results_csv = OUTPUT_DIR / "baseline_results.csv"
    pd.DataFrame(rows).to_csv(results_csv, index=False)
    print(f"  ✓ {results_csv}")

    # per_class_results.csv
    pc_rows = []
    for name, r in all_results.items():
        if r and "per_class" in r:
            for cls, cls_metrics in r["per_class"].items():
                pc_rows.append({
                    "baseline": r.get("baseline", name),
                    "class": cls,
                    **cls_metrics,
                })
    pc_csv = OUTPUT_DIR / "per_class_results.csv"
    pd.DataFrame(pc_rows).to_csv(pc_csv, index=False)
    print(f"  ✓ {pc_csv}")

    # confusion_matrices.json
    cm_data = {}
    for name, r in all_results.items():
        if r and "confusion_matrix" in r:
            cm_data[r.get("baseline", name)] = {
                "matrix": r["confusion_matrix"],
                "labels": r.get("confusion_matrix_labels", LABEL_ORDER),
            }
    cm_json = OUTPUT_DIR / "confusion_matrices.json"
    with open(cm_json, "w", encoding="utf-8") as f:
        json.dump(cm_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {cm_json}")

    # baseline_predictions.csv
    pred_df = pd.DataFrame({
        "index": list(range(len(test))),
        "question": test_questions,
        "gold_label": test_labels,
    })
    for bname, bpreds in all_preds.items():
        if bpreds and len(bpreds) == len(test):
            pred_df[f"pred_{bname}"] = bpreds
    pred_csv = OUTPUT_DIR / "baseline_predictions.csv"
    pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")
    print(f"  ✓ {pred_csv}")

    # experiment_report.md
    report = generate_report(all_results, all_infos, timing, dataset_files, split_counts)

    # Append dataset statistics LaTeX if available
    ds_stats_path = OUTPUT_DIR / "dataset_statistics.json"
    if ds_stats_path.exists():
        with open(ds_stats_path, "r", encoding="utf-8") as f:
            ds_stats = json.load(f)
        report += "\n### Dataset Statistics\n\n```latex\n"
        latex_fields = [
            ("Question length (chars)", "question_length_chars"),
            ("Answer length (chars)", "answer_length_chars"),
            ("Gold context length (chars)", "gold_context_length_chars"),
            ("Hop count: dense class", "hop_count_dense_retrieval"),
            ("Hop count: graph class", "hop_count_graph_traversal"),
            ("Hop count: hybrid class", "hop_count_hybrid_reasoning"),
        ]
        for display, key in latex_fields:
            s = ds_stats.get(key, {})
            report += f"{display:35s} & {s.get('count',0)} & {s.get('mean',0):.2f} & {s.get('std',0):.2f} \\\\\n"
        report += "```\n"

    report_path = OUTPUT_DIR / "experiment_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  ✓ {report_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # Print summary
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Baseline':30s} {'Accuracy':>10s} {'Macro-F1':>10s} {'Weighted-F1':>12s}")
    print("-" * 70)
    for name, r in all_results.items():
        if r:
            oracle_tag = " *oracle*" if r.get("is_oracle") else ""
            print(f"{r.get('baseline', name):30s} {r.get('accuracy', 0):10.4f} "
                  f"{r.get('macro_f1', 0):10.4f} {r.get('weighted_f1', 0):12.4f}{oracle_tag}")
    print("=" * 70)
    print(f"\nTotal time: {timing.get('Total', 0):.1f}s")


if __name__ == "__main__":
    main()
