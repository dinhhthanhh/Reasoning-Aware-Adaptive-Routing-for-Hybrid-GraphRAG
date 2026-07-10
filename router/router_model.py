"""Stage 1 XGBoost router classifier.

Provides the fast first-stage routing decision using a trained
XGBoost multi-class classifier on query features.

Label schema (consistent with QA dataset):
  - dense_retrieval:   Single-hop, simple lookup questions
  - graph_traversal:   Multi-hop, intra-document reasoning
  - hybrid_reasoning:  Cross-document, complex synthesis
  - clarify:           Ambiguous, need more info (dynamically added)
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xgboost as xgb
import yaml
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from router.features import QueryFeatures


@dataclass
class RouterOutput:
    """Output from the router classifier.

    Attributes:
        route: Selected routing destination.
        confidence: Confidence score (0-1).
        feature_importances: Feature importance dict for logging.
    """

    route: Literal["dense_retrieval", "graph_traversal", "hybrid_reasoning", "clarify"] = "dense_retrieval"
    confidence: float = 0.0
    feature_importances: dict[str, float] | None = None
    stage2_trigger_reasons: list[str] = field(default_factory=list)


@dataclass
class TrainingReport:
    """Report from router model training.

    Attributes:
        accuracy: Overall accuracy on validation set.
        f1_per_class: F1 score for each class.
        confusion_mat: Confusion matrix as 2D list.
        cv_scores: Cross-validation accuracy scores.
        feature_importances: Feature importance dict.
        classification_report_str: Full sklearn classification report string.
    """

    accuracy: float = 0.0
    f1_per_class: dict[str, float] | None = None
    confusion_mat: list[list[int]] | None = None
    cv_scores: list[float] | None = None
    feature_importances: dict[str, float] | None = None
    classification_report_str: str = ""


class RouterModel:
    """XGBoost classifier for initial routing decisions (Stage 1).

    Provides fast classification of legal queries into routing categories
    aligned with the QA dataset labels:
      - 'dense_retrieval'  : simple single-hop lookup
      - 'graph_traversal'  : multi-hop intra-document reasoning
      - 'hybrid_reasoning' : cross-document complex synthesis
      - 'clarify'          : ambiguous query needing clarification

    Uses balanced class weighting to handle the natural imbalance
    in the QA dataset (dense_retrieval dominates at ~75%).
    """

    # Must stay in sync with qa_pipeline routing_label values
    CLASS_LABELS: list[str] = [
        "dense_retrieval",
        "graph_traversal",
        "hybrid_reasoning",
        "clarify",
    ]

    # Map label string → integer index
    LABEL_TO_IDX: dict[str, int] = {
        label: i for i, label in enumerate(CLASS_LABELS)
    }

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
        self.use_calibration: bool = config["stage1"].get("use_calibration", True)
        self.graph_priority_enabled: bool = config["stage1"].get("graph_priority_enabled", False)
        self.graph_priority_threshold: float = config["stage1"].get("graph_priority_threshold", 0.16)
        self.graph_priority_confidence_floor: float = config["stage1"].get(
            "graph_priority_confidence_floor", 0.55
        )

        self._model: CalibratedClassifierCV | xgb.XGBClassifier | None = None
        self._label_encoder: LabelEncoder | None = None
        self._feature_names: list[str] = QueryFeatures.feature_names()
        # Feature schema the loaded model was actually trained on (read from the
        # sibling feature_names.json). Lets an enriched QueryFeatures superset
        # serve both the legacy 16-feature model and the enriched model without
        # a shape mismatch. None until a model is loaded.
        self._train_feature_names: list[str] | None = None

        logger.info(
            "RouterModel initialized | model_path={} | threshold={} | classes={}",
            self.model_path,
            self.confidence_threshold,
            self.CLASS_LABELS,
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

        # Convert to feature vector. If the model declares its training feature
        # schema, build the vector in THAT order/subset so an enriched
        # QueryFeatures superset stays compatible with older models.
        if self._train_feature_names:
            vec = features.to_named_vector(self._train_feature_names)
        else:
            vec = features.to_vector()
        feature_vec = np.array([vec], dtype=np.float32)

        # Predict probabilities (calibrated)
        probas = self._model.predict_proba(feature_vec)[0]
        predicted_idx = int(np.argmax(probas))
        confidence = float(probas[predicted_idx])

        # Map predicted index back to global label string
        model_classes = getattr(self._model, "classes_", None)
        if model_classes is not None and len(model_classes) == len(probas):
            route = self._route_from_class_idx(int(model_classes[predicted_idx]))
        else:
            route = self._route_from_class_idx(predicted_idx)

        graph_confidence = self._class_probability(probas, model_classes, "graph_traversal")
        if self._should_prioritize_graph(route, graph_confidence, features):
            logger.debug(
                "Graph priority override | original_route={} | graph_conf={:.3f} | "
                "complexity_lvl={} | graph_kw={} | refs={}",
                route,
                graph_confidence,
                features.complexity_level,
                features.graph_keyword_count,
                features.legal_reference_count,
            )
            route = "graph_traversal"
            confidence = max(graph_confidence, self.graph_priority_confidence_floor)

        # Get feature importances — handle CalibratedClassifierCV wrapper
        try:
            if hasattr(self._model, "feature_importances_"):
                raw_importances = self._model.feature_importances_
            elif hasattr(self._model, "calibrated_classifiers_"):
                # CalibratedClassifierCV: extract from first calibrated estimator
                base = self._model.calibrated_classifiers_[0].estimator
                raw_importances = base.feature_importances_
            else:
                raw_importances = None

            importances = (
                dict(zip(self._feature_names, raw_importances.tolist()))
                if raw_importances is not None
                else {}
            )
        except Exception:
            importances = {}

        logger.debug(
            "Stage 1 prediction | route={} | confidence={:.3f} | calibrated={}",
            route,
            confidence,
            isinstance(self._model, CalibratedClassifierCV),
        )

        return RouterOutput(
            route=route,
            confidence=confidence,
            feature_importances=importances,
        )

    def _route_from_class_idx(self, class_idx: int) -> str:
        """Map a classifier class index to a routing label string."""
        if self._label_encoder is not None:
            global_idx = int(self._label_encoder.inverse_transform([class_idx])[0])
            return self.CLASS_LABELS[global_idx]
        if 0 <= class_idx < len(self.CLASS_LABELS):
            return self.CLASS_LABELS[class_idx]
        return self.CLASS_LABELS[-1]

    def _class_probability(
        self,
        probas: np.ndarray,
        model_classes: Any,
        label: str,
    ) -> float:
        """Return calibrated probability for a route label."""
        label_idx = self.LABEL_TO_IDX[label]
        if model_classes is not None and len(model_classes) == len(probas):
            for pos, class_idx in enumerate(model_classes):
                global_idx = (
                    int(self._label_encoder.inverse_transform([int(class_idx)])[0])
                    if self._label_encoder is not None
                    else int(class_idx)
                )
                if global_idx == label_idx:
                    return float(probas[pos])
            return 0.0
        if label_idx < len(probas):
            return float(probas[label_idx])
        return 0.0

    def _should_prioritize_graph(
        self,
        route: str,
        graph_confidence: float,
        features: QueryFeatures,
    ) -> bool:
        """Promote relation-heavy legal queries to graph traversal.

        The learned classifier is conservative for graph_traversal because the
        strict split has fewer graph examples than dense examples. This rule is
        a calibrated tie-breaker: it only applies when graph probability is
        non-trivial and the semantic features point to relation traversal, while
        excluding factoid and cross-document cases.
        """
        if not self.graph_priority_enabled:
            return False
        if route not in {"dense_retrieval", "hybrid_reasoning"}:
            return False
        if graph_confidence < self.graph_priority_threshold:
            return False
        if features.is_factoid or features.cross_doc_signals:
            return False
        return features.complexity_level >= 2


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

        # If cross-doc signals → hybrid_reasoning
        if features.cross_doc_signals:
            return RouterOutput(route="hybrid_reasoning", confidence=0.65)

        # If multi-hop signals or comparison → graph_traversal
        if (
            features.multi_hop_score > 0.5
            or features.has_comparison
            or features.legal_reference_count > 1
            or features.graph_keyword_count > 1
        ):
            return RouterOutput(route="graph_traversal", confidence=0.6)

        # Default to dense_retrieval
        return RouterOutput(route="dense_retrieval", confidence=0.65)

    def train(self, X: np.ndarray, y: np.ndarray) -> TrainingReport:
        """Train the XGBoost classifier with cross-validation.

        Uses balanced sample weights to handle the natural class imbalance
        in legal QA data (dense_retrieval is ~75% of data).

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Label array of shape (n_samples,), integer-encoded.

        Returns:
            TrainingReport with accuracy, F1, confusion matrix, CV scores.
        """
        report = TrainingReport()

        # XGBoost expects contiguous class ids 0..n-1
        self._label_encoder = LabelEncoder()
        y = self._label_encoder.fit_transform(y).astype(np.int32)

        # Compute balanced sample weights
        sample_weights = compute_sample_weight(class_weight="balanced", y=y)

        # 5-fold stratified cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores: list[float] = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            sw_fold = sample_weights[train_idx]

            # Determine num_class from actual data
            num_class = len(np.unique(y))

            fold_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                num_class=num_class,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1,
            )
            fold_model.fit(
                X_fold_train, y_fold_train,
                sample_weight=sw_fold,
                eval_set=[(X_fold_val, y_fold_val)],
                verbose=False,
            )
            fold_preds = fold_model.predict(X_fold_val)
            fold_acc = accuracy_score(y_fold_val, fold_preds)
            cv_scores.append(fold_acc)
            logger.info("Fold {}/5 accuracy: {:.4f}", fold, fold_acc)

        report.cv_scores = cv_scores

        # Train final model on full data (80/20 split for final eval)
        num_class = len(np.unique(y))
        base_xgb = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=num_class,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )

        # 80/20 split for final evaluation
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        sw_train = sample_weights[:split]

        base_xgb.fit(
            X_train, y_train,
            sample_weight=sw_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # --- Calibration (Adaptive-RAG: Isotonic Regression) ---
        # Wrap the fitted XGBoost with isotonic calibration on the val split
        # This corrects overconfident softmax probabilities.
        if self.use_calibration:
            logger.info("Applying Isotonic Calibration (Platt scaling variant)...")
            calibrated = CalibratedClassifierCV(
                estimator=base_xgb,
                method="isotonic",
                cv="prefit",  # Already fitted; calibrate on val split
            )
            calibrated.fit(X_val, y_val)
            self._model = calibrated
            logger.info("Calibration applied | model=CalibratedClassifierCV(isotonic)")
        else:
            self._model = base_xgb

        # Evaluate on validation set
        val_preds = self._model.predict(X_val)
        report.accuracy = float(accuracy_score(y_val, val_preds))

        # Get the unique classes actually present in validation
        present_classes = sorted(np.unique(np.concatenate([y_val, val_preds])))
        global_class_ids = [
            int(self._label_encoder.inverse_transform([int(i)])[0])
            for i in present_classes
        ]
        target_names = [
            self.CLASS_LABELS[i] for i in global_class_ids if i < len(self.CLASS_LABELS)
        ]

        # Classification report (full)
        cls_report_str = classification_report(
            y_val, val_preds,
            labels=present_classes,
            target_names=target_names,
            zero_division=0,
        )
        report.classification_report_str = cls_report_str

        # Per-class F1 dict
        cls_report_dict = classification_report(
            y_val, val_preds,
            labels=present_classes,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
        report.f1_per_class = {
            label: cls_report_dict[label]["f1-score"]
            for label in target_names
            if label in cls_report_dict
        }

        # Confusion matrix
        cm = confusion_matrix(y_val, val_preds, labels=present_classes)
        report.confusion_mat = cm.tolist()

        # Feature importances — handle CalibratedClassifierCV wrapper
        try:
            if hasattr(self._model, "feature_importances_"):
                fi = self._model.feature_importances_
            elif hasattr(self._model, "calibrated_classifiers_"):
                fi = self._model.calibrated_classifiers_[0].estimator.feature_importances_
            else:
                fi = None
            if fi is not None:
                report.feature_importances = dict(zip(self._feature_names, fi.tolist()))
        except Exception as exc:
            logger.warning("Could not extract feature importances: {}", exc)


        logger.info(
            "Training complete | accuracy={:.4f} | cv_mean={:.4f} ± {:.4f}",
            report.accuracy,
            float(np.mean(cv_scores)),
            float(np.std(cv_scores)),
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

        payload = {
            "model": self._model,
            "label_encoder": self._label_encoder,
        }
        with open(save_path, "wb") as f:
            pickle.dump(payload, f)

        logger.info("Model saved to {}", save_path)

    def load(self, path: str | None = None) -> bool:
        """Load a trained model from disk.

        Args:
            path: File path. Defaults to config model_path.

        Returns:
            True if loading succeeded.
        """
        import json
        load_path = Path(path) if path else self.model_path
        if not load_path.exists():
            logger.warning("Model file not found: {}", load_path)
            return False

        try:
            with open(load_path, "rb") as f:
                payload = pickle.load(f)
            if isinstance(payload, dict) and "model" in payload:
                self._model = payload["model"]
                self._label_encoder = payload.get("label_encoder")
            else:
                self._model = payload
                self._label_encoder = None

            # Load the feature schema this model was trained on, if present.
            self._train_feature_names = None
            fn_path = load_path.parent / "feature_names.json"
            if fn_path.exists():
                try:
                    import json
                    with open(fn_path, "r", encoding="utf-8") as ff:
                        self._train_feature_names = json.load(ff)
                    self._feature_names = list(self._train_feature_names)
                except Exception as exc:
                    logger.warning("Failed to read feature_names.json: {}", exc)

            # Sanity-check the schema against the model's expected input width.
            n_expected = getattr(self._model, "n_features_in_", None)
            if (
                self._train_feature_names
                and n_expected
                and len(self._train_feature_names) != n_expected
            ):
                logger.warning(
                    "feature_names.json width ({}) != model n_features_in_ ({}); "
                    "predictions may be unreliable — retrain to align.",
                    len(self._train_feature_names),
                    n_expected,
                )

            logger.info(
                "Model loaded from {} | feature_schema={} features",
                load_path,
                len(self._train_feature_names) if self._train_feature_names else "unknown",
            )
            return True
        except Exception as exc:
            logger.error("Failed to load model: {}", exc)
            return False
