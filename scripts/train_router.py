"""
train_router.py
===============
Official training script for the XGBoost routing classifier.
"""

import argparse
import json
import os
import pickle
import warnings
warnings.filterwarnings("ignore")
from collections import Counter

import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb

# Import the fixed feature extractor (must be in the same directory)
try:
    from feature_extractor_fixed import VietnameseLegalFeatureExtractor
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from feature_extractor_fixed import VietnameseLegalFeatureExtractor


# ─────────────────────────────────────────────────────────────────────────────
# LABEL MAPPING
# ─────────────────────────────────────────────────────────────────────────────

LABEL2INT = {
    "dense_retrieval":  0,
    "graph_traversal":  1,
    "hybrid_reasoning": 2,
}
INT2LABEL = {v: k for k, v in LABEL2INT.items()}
LABEL_NAMES = ["dense_retrieval", "graph_traversal", "hybrid_reasoning"]


# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def prepare_dataset(benchmark_path: str, extractor: VietnameseLegalFeatureExtractor):
    """
    Load benchmark and extract feature matrix.

    Returns
    -------
    splits : dict with keys 'train', 'dev', 'test', 'all'
             each containing (X: np.ndarray, y: np.ndarray, entries: list)
    feat_names : list[str]
    """
    with open(benchmark_path, encoding="utf-8") as f:
        data = json.load(f)

    feat_names = extractor.feature_names()

    def extract(entries):
        X = np.array([
            [v for v in extractor.extract(d["question"]).values()]
            for d in entries
        ], dtype=np.float32)
        y = np.array([LABEL2INT[d["routing_label"]] for d in entries], dtype=np.int32)
        return X, y

    splits_raw = {
        "train": [d for d in data if d.get("source_split") == "train"],
        "dev":   [d for d in data if d.get("source_split") == "dev"],
        "test":  [d for d in data if d.get("source_split") == "test"],
        "all":   data,
    }

    splits = {}
    for name, entries in splits_raw.items():
        if not entries:
            continue
        X, y = extract(entries)
        splits[name] = {"X": X, "y": y, "entries": entries, "n": len(entries)}

    label_dist = Counter(d["routing_label"] for d in data)
    print(f"  Loaded {len(data)} entries from {benchmark_path}")
    print(f"  Split sizes — train:{splits.get('train', {}).get('n', 0)}  dev:{splits.get('dev', {}).get('n', 0)}  test:{splits.get('test', {}).get('n', 0)}")
    print(f"  Label distribution: {dict(label_dist)}")

    return splits, feat_names


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

BEST_PARAMS = {
    "n_estimators":     200,
    "max_depth":        6,
    "learning_rate":    0.1,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "gamma":            0,
    "eval_metric":      "mlogloss",
    "random_state":     42,
    "verbosity":        0,
}


def train_model(X_train, y_train, X_val, y_val):
    model = xgb.XGBClassifier(**BEST_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def print_eval_report(y_true, y_pred, split_name: str, n: int) -> dict:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(
        y_true, y_pred,
        target_names=LABEL_NAMES,
        output_dict=True,
        digits=3,
    )

    print(f"\n{'═' * 60}")
    print(f"  {split_name.upper()} EVALUATION  (N={n})")
    print(f"{'═' * 60}")
    print(f"  Accuracy:  {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"  Macro F1:  {macro_f1:.4f}")
    print(f"\n  Per-class breakdown:")
    print(f"  {'Label':<25} {'P':>7} {'R':>7} {'F1':>7} {'N':>6}")
    print("  " + "-" * 55)
    for label in LABEL_NAMES:
        s = report[label]
        print(
            f"  {label:<25} {s['precision']:>7.3f} {s['recall']:>7.3f} "
            f"{s['f1-score']:>7.3f} {int(s['support']):>6}"
        )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  Confusion matrix (row=true, col=pred):")
    header = f"  {'':>10}" + "".join(f"  {l[:8]:>10}" for l in LABEL_NAMES)
    print(header)
    for i, label in enumerate(LABEL_NAMES):
        row_str = f"  {label[:10]:>10}" + "".join(f"  {cm[i,j]:>10}" for j in range(len(LABEL_NAMES)))
        print(row_str)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": {
            label: {
                "P": report[label]["precision"],
                "R": report[label]["recall"],
                "F1": report[label]["f1-score"],
                "N": int(report[label]["support"]),
            }
            for label in LABEL_NAMES
        },
    }


def print_feature_importance(model, feat_names: list):
    fi = model.feature_importances_
    print(f"\n{'═' * 60}")
    print("  FEATURE IMPORTANCES (XGBoost gain)")
    print(f"{'═' * 60}")
    for name, imp in sorted(zip(feat_names, fi), key=lambda x: -x[1]):
        if imp > 0.001:
            bar = "█" * max(1, int(imp * 200))
            print(f"  {name:<32} {imp:.4f}  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-VALIDATION (most statistically reliable for N=600)
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_validation(X_all, y_all, n_splits: int = 5) -> dict:
    print(f"\n{'═' * 60}")
    print(f"  {n_splits}-FOLD STRATIFIED CROSS-VALIDATION  (N={len(X_all)})")
    print(f"{'═' * 60}")
    model_cv = xgb.XGBClassifier(**BEST_PARAMS)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_acc   = cross_val_score(model_cv, X_all, y_all, cv=skf, scoring="accuracy")
    cv_f1    = cross_val_score(model_cv, X_all, y_all, cv=skf, scoring="f1_macro")

    print(f"  Accuracy:  {cv_acc.mean()*100:.2f}% ± {cv_acc.std()*100:.2f}%")
    print(f"  Macro F1:  {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    print(f"  Folds:     {[f'{s*100:.2f}%' for s in cv_acc]}")
    print(
        f"\n  Note: CV accuracy is the authoritative number for the thesis because "
        f"the full benchmark file contains only test-split data (strict_split='test'). "
        f"The source_split field provides an internal train/dev/test partition."
    )
    return {
        "cv_accuracy_mean":  float(cv_acc.mean()),
        "cv_accuracy_std":   float(cv_acc.std()),
        "cv_macro_f1_mean":  float(cv_f1.mean()),
        "cv_macro_f1_std":   float(cv_f1.std()),
        "cv_fold_scores":    cv_acc.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(X_all, y_all, feat_names: list):
    """
    Ablation study: which feature groups contribute the most?
    Groups: (A) length features, (B) keyword features, (C) semantic features
    """
    print(f"\n{'═' * 60}")
    print("  ABLATION STUDY — Feature Group Contribution")
    print(f"{'═' * 60}")

    groups = {
        "Length only (chars+words)":
            ["query_length_chars", "query_length_words"],
        "Keyword features":
            ["graph_keyword_count", "legal_reference_count",
             "dieu_reference_count", "multi_article_ref"],
        "Semantic signals":
            ["is_yes_no_question", "is_factoid_question",
             "has_conditional_structure", "has_negation",
             "cross_doc_signal_count", "has_procedure_marker"],
        "All 16 features":
            feat_names,
        "Length + Semantic (no keyword)":
            [f for f in feat_names if f not in
             ["graph_keyword_count", "legal_reference_count",
              "dieu_reference_count", "multi_article_ref"]],
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for group_name, feats in groups.items():
        indices = [feat_names.index(f) for f in feats if f in feat_names]
        X_sub = X_all[:, indices]
        model_abl = xgb.XGBClassifier(**BEST_PARAMS)
        cv_scores = cross_val_score(model_abl, X_sub, y_all, cv=skf, scoring="accuracy")
        mean_acc = cv_scores.mean() * 100
        print(f"  {group_name:<38}  {mean_acc:>6.2f}% ± {cv_scores.std()*100:.2f}%")
        results[group_name] = {"accuracy": float(cv_scores.mean()), "std": float(cv_scores.std())}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark",  required=True, help="Path to test_benchmark_v2.json")
    parser.add_argument("--output-dir", default="router_model", help="Directory to save model")
    parser.add_argument("--ablation",   action="store_true", help="Run ablation study")
    parser.add_argument("--cv-only",    action="store_true", help="Only run CV, skip train/test split")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset and extracting features...")
    extractor = VietnameseLegalFeatureExtractor()
    splits, feat_names = prepare_dataset(args.benchmark, extractor)

    all_results = {}

    # ── Cross-validation on full dataset (authoritative) ────────────────────
    cv_results = run_cross_validation(splits["all"]["X"], splits["all"]["y"])
    all_results["cross_validation"] = cv_results

    if not args.cv_only:
        # ── Train/dev/test split evaluation ─────────────────────────────────
        if "train" in splits and "dev" in splits and "test" in splits:
            print("\nTraining on source_split=train (N=409), validating on dev (N=92)...")
            model = train_model(
                splits["train"]["X"], splits["train"]["y"],
                splits["dev"]["X"],   splits["dev"]["y"],
            )

            val_metrics  = print_eval_report(
                splits["dev"]["y"],  model.predict(splits["dev"]["X"]),
                "VALIDATION (dev)", splits["dev"]["n"]
            )
            test_metrics = print_eval_report(
                splits["test"]["y"], model.predict(splits["test"]["X"]),
                "HELD-OUT TEST",    splits["test"]["n"]
            )
            all_results["validation"] = val_metrics
            all_results["held_out_test"] = test_metrics

            print_feature_importance(model, feat_names)

            # ── Train on train+dev, final model for deployment ──────────────────
            print("\nTraining final model on train+dev (N=501) for deployment...")
            X_traindev = np.vstack([splits["train"]["X"], splits["dev"]["X"]])
            y_traindev = np.concatenate([splits["train"]["y"], splits["dev"]["y"]])
            final_model = train_model(X_traindev, y_traindev, splits["test"]["X"], splits["test"]["y"])
        else:
            print("\nSkipping train/dev/test split because data lacks source_split field.")
            final_model = train_model(splits["all"]["X"], splits["all"]["y"], splits["all"]["X"], splits["all"]["y"])

        # Save
        model_path = os.path.join(args.output_dir, "xgb_router.pkl")
        meta_path  = os.path.join(args.output_dir, "feature_names.json")
        with open(model_path, "wb") as f:
            pickle.dump(final_model, f)
        with open(meta_path, "w") as f:
            json.dump(feat_names, f)
        print(f"  Final model saved to: {model_path}")
        print(f"  Feature names saved to: {meta_path}")

    # ── Ablation study ───────────────────────────────────────────────────────
    if args.ablation:
        ablation_results = run_ablation(splits["all"]["X"], splits["all"]["y"], feat_names)
        all_results["ablation"] = ablation_results

    # ── Save all metrics ─────────────────────────────────────────────────────
    results_path = os.path.join(args.output_dir, "training_report.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nFull training report saved to: {results_path}")

    # ── Summary for thesis ───────────────────────────────────────────────────
    cv = all_results["cross_validation"]
    print(f"\n{'═' * 60}")
    print("  THESIS-READY NUMBERS (cite CV as primary metric)")
    print(f"{'═' * 60}")
    print(f"  Routing Accuracy:  {cv['cv_accuracy_mean']*100:.2f}% ± {cv['cv_accuracy_std']*100:.2f}%")
    print(f"  Routing Macro F1:  {cv['cv_macro_f1_mean']:.4f} ± {cv['cv_macro_f1_std']:.4f}")
    if "held_out_test" in all_results:
        ht = all_results["held_out_test"]
        print(f"  Held-out test acc: {ht['accuracy']*100:.2f}%  (source_split=test)")

if __name__ == "__main__":
    main()
