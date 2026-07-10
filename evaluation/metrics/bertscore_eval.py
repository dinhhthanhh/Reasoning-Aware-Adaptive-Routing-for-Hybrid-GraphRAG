"""Single canonical BERTScore implementation for Vietnamese legal QA.

The repository previously contained two divergent BERTScore code paths:

* ``scripts/evaluate_pipeline_fixed.py`` -> ``xlm-roberta-large`` vs ``concise_answer``
* ``scripts/evaluate_pipeline.py``       -> (PhoBERT) vs ``gold_context``

Those produced incompatible numbers (0.85 vs 0.69) because they used different
models AND different references. This module is the ONE supported implementation.

Configuration (documented and fixed):
    * Model:     vinai/phobert-base   (Vietnamese-native; override via ``model_type``)
    * Language:  vi
    * Reference: caller chooses, but it MUST be recorded in the output dict.

Note: ``bert-score`` and ``torch`` are heavy optional dependencies. They are
imported lazily so importing this module never forces a model download.
"""

from __future__ import annotations

__all__ = ["BERTSCORE_MODEL", "BERTSCORE_LANG", "compute_bertscore"]

BERTSCORE_MODEL = "vinai/phobert-base"
BERTSCORE_LANG = "vi"


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    *,
    model_type: str = BERTSCORE_MODEL,
    lang: str = BERTSCORE_LANG,
    reference_field: str = "unspecified",
    batch_size: int = 16,
) -> dict[str, object]:
    """Compute corpus BERTScore (P/R/F1) for Vietnamese answers.

    Args:
        predictions: Generated answers.
        references: Reference answers (same length as ``predictions``).
        model_type: HuggingFace model id used by ``bert-score``.
        lang: Language code passed to ``bert-score``.
        reference_field: Name of the dataset field used as the reference
            (e.g. ``"gold_context"`` or ``"answer"``). Recorded for provenance.
        batch_size: Scoring batch size.

    Returns:
        Dict with mean ``precision``/``recall``/``f1``, their ``*_std``, the
        ``model``, ``lang``, ``reference_field`` and ``n``.

    Raises:
        ImportError: If ``bert-score`` is not installed (message explains fix).
        ValueError: If lengths differ.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) and references "
            f"({len(references)}) must have equal length"
        )
    if not predictions:
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "precision_std": 0.0, "recall_std": 0.0, "f1_std": 0.0,
            "model": model_type, "lang": lang,
            "reference_field": reference_field, "n": 0,
        }

    try:
        from bert_score import score as bert_score_fn
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "bert-score is required for compute_bertscore. "
            "Install it with: pip install bert-score"
        ) from exc

    import numpy as np

    kwargs = {
        "lang": lang,
        "model_type": model_type,
        "batch_size": batch_size,
        "rescale_with_baseline": False,
        "verbose": False,
    }
    if "phobert-base" in model_type:
        kwargs["num_layers"] = 12

    # Truncate to first 120 words to fit within 256 token limit of PhoBERT
    def _truncate(text: str) -> str:
        return " ".join(text.split()[:120])
        
    predictions_trunc = [_truncate(p) for p in predictions]
    references_trunc = [_truncate(r) for r in references]

    P, R, F = bert_score_fn(
        predictions_trunc,
        references_trunc,
        **kwargs
    )

    return {
        "precision": float(P.mean()),
        "recall": float(R.mean()),
        "f1": float(F.mean()),
        "precision_std": float(np.std(P.numpy())),
        "recall_std": float(np.std(R.numpy())),
        "f1_std": float(np.std(F.numpy())),
        "model": model_type,
        "lang": lang,
        "reference_field": reference_field,
        "n": len(predictions),
    }
