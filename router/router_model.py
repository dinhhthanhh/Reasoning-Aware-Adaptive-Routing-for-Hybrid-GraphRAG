"""Stage 1 XGBoost router classifier.

Provides the fast first-stage routing decision using a trained
XGBoost multi-class classifier on query features.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xgboost as xgb
import yaml
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from router.features import QueryFeatures


@dataclass
class RouterOutput:
    """Output from the router classifier.

    Attributes:
        route: Selected routing destination.
        confidence: Confidence score (0-1).
        feature_importances: Feature importance dict for logging.
    """

    route: Literal["vector", "graph", "clarify"] = "vector"
    confidence: float = 0.0
    feature_importances: dict[str, float] | None = None


@dataclass
class TrainingReport:
    """Report from router model training.

    Attributes:
        accuracy: Overall accuracy on validation set.
        f1_per_class: F1 score for each class.
        confusion_mat: Confusion matrix as 2D list.
        cv_scores: Cross-validation accuracy scores.
        feature_importances: Feature importance dict.
    """

    accuracy: float = 0.0
    f1_per_class: dict[str, float] | None = None
    confusion_mat: list[list[int]] | None = None
    cv_scores: list[float] | None = None
    feature_importances: dict[str, float] | None = None


class RouterModel:
    """XGBoost classifier for initial routing decisions (Stage 1).

    Provides fast classification of legal queries into three categories:
    'vector' (simple lookup), 'graph' (multi-hop reasoning), or
    'clarify' (ambiguous, needs more info).
    """

    CLASS_LABELS: list[str] = ["vector", "graph", "clarify"]

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize router model.

        Args:
            config: Router config dict. If None, loads from configs/config.yaml.
        """
        if config is None:
            config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                full_config = yaml.safe_load(f)
            config = full_config["router"]
        elif "router" in config:
            # Full config passed — extract router sub-config
            config = config["router"]

        self.model_path = Path(config["stage1"]["model_path"])
        self.confidence_threshold: float = config["stage1"].get("confidence_threshold", 0.85)

        self._model: xgb.XGBClassifier | None = None
        self._feature_names: list[str] = QueryFeatures.feature_names()

        logger.info(
            "RouterModel initialized | model_path={} | threshold={}",
            self.model_path,
            self.confidence_threshold,
        )

    def predict(self, features: QueryFeatures) -> RouterOutput:
        """Predict routing destination from query features.

        Args:
            features: Extracted query features.

        Returns:
            RouterOutput with route, confidence, and feature importances.
        """
        if self._model is None:
            if not self.load(str(self.model_path)):
                logger.warning("Model not loaded, using rule-based fallback")
                return self._rule_based_fallback(features)

        # Convert to feature vector
        feature_vec = np.array([features.to_vector()], dtype=np.float32)

        # Predict probabilities
        probas = self._model.predict_proba(feature_vec)[0]
        predicted_idx = int(np.argmax(probas))
        confidence = float(probas[predicted_idx])
        route = self.CLASS_LABELS[predicted_idx]

        # Get feature importances
        importances = dict(zip(
            self._feature_names,
            self._model.feature_importances_.tolist(),
        ))

        logger.debug(
            "Stage 1 prediction | route={} | confidence={:.3f}",
            route,
            confidence,
        )

        return RouterOutput(
            route=route,
            confidence=confidence,
            feature_importances=importances,
        )

    def _rule_based_fallback(self, features: QueryFeatures) -> RouterOutput:
        """Provide a rule-based routing decision when model is unavailable.

        Args:
            features: Extracted query features.

        Returns:
            RouterOutput based on simple heuristics.
        """
        # If highly ambiguous → clarify
        if features.ambiguity_score >= 0.6 or features.has_pronoun:
            return RouterOutput(route="clarify", confidence=0.7)

        # If multi-hop signals or comparison → graph
        if (features.multi_hop_score > 0.5 or features.has_comparison or
                features.legal_reference_count > 1 or features.graph_keyword_count > 1):
            return RouterOutput(route="graph", confidence=0.6)

        # Default to vector
        return RouterOutput(route="vector", confidence=0.65)

    def train(self, X: np.ndarray, y: np.ndarray) -> TrainingReport:
        """Train the XGBoost classifier with cross-validation.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Label array of shape (n_samples,), integer-encoded.

        Returns:
            TrainingReport with accuracy, F1, confusion matrix, CV scores.
        """
        report = TrainingReport()

        # 5-fold stratified cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores: list[float] = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]

            fold_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective="multi:softprob",
                num_class=len(self.CLASS_LABELS),
                eval_metric="mlogloss",
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1,
            )
            fold_model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                verbose=False,
            )
            fold_preds = fold_model.predict(X_fold_val)
            fold_acc = accuracy_score(y_fold_val, fold_preds)
            cv_scores.append(fold_acc)
            logger.info("Fold {}/5 accuracy: {:.4f}", fold, fold_acc)

        report.cv_scores = cv_scores

        # Train final model on full data
        self._model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=len(self.CLASS_LABELS),
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
        )

        # 80/20 split for final evaluation
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        self._model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Evaluate on validation
        val_preds = self._model.predict(X_val)
        report.accuracy = float(accuracy_score(y_val, val_preds))

        # Classification report
        cls_report = classification_report(
            y_val, val_preds,
            target_names=self.CLASS_LABELS,
            output_dict=True,
        )
        report.f1_per_class = {
            label: cls_report[label]["f1-score"]
            for label in self.CLASS_LABELS
            if label in cls_report
        }

        # Confusion matrix
        cm = confusion_matrix(y_val, val_preds)
        report.confusion_mat = cm.tolist()

        # Feature importances
        report.feature_importances = dict(zip(
            self._feature_names,
            self._model.feature_importances_.tolist(),
        ))

        logger.info(
            "Training complete | accuracy={:.4f} | cv_mean={:.4f} | cv_std={:.4f}",
            report.accuracy,
            np.mean(cv_scores),
            np.std(cv_scores),
        )

        return report

    def save(self, path: str | None = None) -> None:
        """Save the trained model to disk.

        Args:
            path: File path. Defaults to config model_path.
        """
        if self._model is None:
            logger.warning("No model to save")
            return

        save_path = Path(path) if path else self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            pickle.dump(self._model, f)

        logger.info("Model saved to {}", save_path)

    def load(self, path: str | None = None) -> bool:
        """Load a trained model from disk.

        Args:
            path: File path. Defaults to config model_path.

        Returns:
            True if loading succeeded.
        """
        load_path = Path(path) if path else self.model_path
        if not load_path.exists():
            logger.warning("Model file not found: {}", load_path)
            return False

        try:
            with open(load_path, "rb") as f:
                self._model = pickle.load(f)
            logger.info("Model loaded from {}", load_path)
            return True
        except Exception as exc:
            logger.error("Failed to load model: {}", exc)
            return False
