"""Train the Stage-1 XGBoost router on the ENRICHED feature vector.

Unlike scripts/train_router.py (which builds X directly from the 16 lexical
features), this script builds X from ``router.features.QueryFeatures.to_vector``
so the training vector is guaranteed to match what the live router produces at
inference time (lexical + reasoning features, in FEATURE_NAMES order).

Non-destructive: writes to a NEW output dir so the current 16-feature model is
left untouched. Switch the system over by pointing
``router.stage1.model_path`` in configs/config.yaml at the new router_model.pkl.

Usage (from repo root):
    python scripts/train_router_enriched.py \
        --data-dir qa_pipeline/data/phapdien_strict \
        --output-dir data/router_training/phapdien_strict_enriched
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from router.features import FEATURE_NAMES, FeatureExtractor

# Stage-1 trains 3 retrieval classes (clarify is handled separately by the
# ambiguity detector / Stage-2 verifier and is not in the training labels).
LABEL2INT = {"dense_retrieval": 0, "graph_traversal": 1, "hybrid_reasoning": 2}
LABEL_NAMES = ["dense_retrieval", "graph_traversal", "hybrid_reasoning"]

BEST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "gamma": 0,
    "eval_metric": "mlogloss",
    "random_state": 42,
    "verbosity": 0,
}


def load_split(path: str):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_matrix(entries, extractor):
    X, y, skipped = [], [], 0
    for d in entries:
        label = d.get("routing_label")
        if label not in LABEL2INT:
            skipped += 1
            continue
        X.append(extractor.extract(d["question"]).to_vector())
        y.append(LABEL2INT[label])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32), skipped


def report(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro")
    print(f"\n=== {name} (N={len(y_true)}) ===")
    print(f"  Accuracy: {acc*100:.2f}%   Macro-F1: {macro:.4f}")
    labels = sorted(set(y_true) | set(y_pred))
    names = [LABEL_NAMES[i] for i in labels]
    print(classification_report(y_true, y_pred, labels=labels, target_names=names, digits=3, zero_division=0))
    print("  Confusion (row=true, col=pred):")
    print("  " + str(confusion_matrix(y_true, y_pred, labels=labels).tolist()))
    return {"accuracy": float(acc), "macro_f1": float(macro)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="qa_pipeline/data/phapdien_strict")
    ap.add_argument("--output-dir", default="data/router_training/phapdien_strict_enriched")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    extractor = FeatureExtractor({"language": "vi"})

    train = load_split(os.path.join(args.data_dir, "train.json"))
    dev = load_split(os.path.join(args.data_dir, "dev.json"))
    test = load_split(os.path.join(args.data_dir, "test.json"))
    all_entries = train + dev + test
    print(f"Loaded train={len(train)} dev={len(dev)} test={len(test)} total={len(all_entries)}")
    print(f"Label distribution: {dict(Counter(d.get('routing_label') for d in all_entries))}")
    print(f"Feature vector length: {len(FEATURE_NAMES)} | features: {FEATURE_NAMES}")

    X_all, y_all, skipped = build_matrix(all_entries, extractor)
    print(f"Built matrix {X_all.shape}, skipped {skipped} non-3-class rows")

    results = {}

    # 5-fold CV (authoritative number for the thesis).
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(xgb.XGBClassifier(**BEST_PARAMS), X_all, y_all, cv=skf, scoring="accuracy")
    cv_f1 = cross_val_score(xgb.XGBClassifier(**BEST_PARAMS), X_all, y_all, cv=skf, scoring="f1_macro")
    print(f"\n5-Fold CV  Accuracy: {cv_acc.mean()*100:.2f}% ± {cv_acc.std()*100:.2f}%  "
          f"Macro-F1: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    results["cross_validation"] = {
        "cv_accuracy_mean": float(cv_acc.mean()),
        "cv_accuracy_std": float(cv_acc.std()),
        "cv_macro_f1_mean": float(cv_f1.mean()),
        "cv_macro_f1_std": float(cv_f1.std()),
        "cv_fold_scores": cv_acc.tolist(),
    }

    # Held-out test eval (train+dev -> test), if splits exist.
    if train and dev and test:
        Xtr, ytr, _ = build_matrix(train, extractor)
        Xdv, ydv, _ = build_matrix(dev, extractor)
        Xte, yte, _ = build_matrix(test, extractor)
        m = xgb.XGBClassifier(**BEST_PARAMS)
        m.fit(np.vstack([Xtr, Xdv]), np.concatenate([ytr, ydv]))
        results["held_out_test"] = report(yte, m.predict(Xte), "HELD-OUT TEST")

    # Final deployment model on ALL data.
    final = xgb.XGBClassifier(**BEST_PARAMS)
    final.fit(X_all, y_all)

    print("\n=== FEATURE IMPORTANCES (gain) ===")
    fi = sorted(zip(FEATURE_NAMES, final.feature_importances_.tolist()), key=lambda x: -x[1])
    for name, imp in fi:
        if imp > 0.001:
            print(f"  {name:<30} {imp:.4f}  {'#' * max(1, int(imp * 120))}")
    results["feature_importances"] = {n: float(i) for n, i in fi}

    model_path = os.path.join(args.output_dir, "router_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(final, f)  # bare model — RouterModel.load handles this
    with open(os.path.join(args.output_dir, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(FEATURE_NAMES, f, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "training_report.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved enriched model -> {model_path}")
    print(f"Saved feature_names.json ({len(FEATURE_NAMES)} features) and training_report.json")


if __name__ == "__main__":
    main()