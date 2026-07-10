"""Vietnamese-aware token-level F1 / Exact Match for legal QA.

This module replaces the previous ``compute_answer_f1`` which returned plain
keyword *recall* over whitespace-split tokens (see
``evaluation/metrics/legacy.py``). That metric was both mislabelled (it was not
F1) and linguistically wrong for Vietnamese, where multi-syllable words are
separated by spaces and must be segmented before token comparison.

Standard (SQuAD-style) token F1 is computed here on top of a Vietnamese word
segmenter:

    F1 = 2 * P * R / (P + R)

with multiset (bag-of-tokens) overlap, after Unicode normalisation, lower-casing
and punctuation stripping.

Tokenizer selection (in priority order):
    1. underthesea.word_tokenize  (default; best Vietnamese segmentation)
    2. pyvi.ViTokenizer.tokenize  (fallback)
    3. whitespace split           (last-resort fallback, emits a warning once)

The active tokenizer name is exposed via :func:`active_tokenizer` so result
files can record exactly how F1 was computed.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

__all__ = [
    "compute_token_f1",
    "compute_corpus_token_f1",
    "vi_tokenize",
    "normalize_text",
    "active_tokenizer",
]


# --------------------------------------------------------------------------- #
# Tokenizer resolution (lazy + cached)
# --------------------------------------------------------------------------- #
_TOKENIZER = None  # type: ignore[var-annotated]
_TOKENIZER_NAME = ""
_WHITESPACE_WARNED = False


def _resolve_tokenizer():
    """Resolve the best available Vietnamese tokenizer exactly once."""
    global _TOKENIZER, _TOKENIZER_NAME
    if _TOKENIZER is not None:
        return _TOKENIZER

    try:
        from underthesea import word_tokenize as _ut

        def _tok(text: str) -> list[str]:
            return _ut(text, format="text").split()

        _TOKENIZER = _tok
        _TOKENIZER_NAME = "underthesea.word_tokenize"
        return _TOKENIZER
    except Exception:  # pragma: no cover - exercised only without underthesea
        pass

    try:
        from pyvi import ViTokenizer

        def _tok(text: str) -> list[str]:
            return ViTokenizer.tokenize(text).split()

        _TOKENIZER = _tok
        _TOKENIZER_NAME = "pyvi.ViTokenizer.tokenize"
        return _TOKENIZER
    except Exception:  # pragma: no cover
        pass

    def _tok(text: str) -> list[str]:
        global _WHITESPACE_WARNED
        if not _WHITESPACE_WARNED:
            import warnings

            warnings.warn(
                "Neither underthesea nor pyvi is available; falling back to "
                "whitespace tokenization. Vietnamese token F1 will be "
                "approximate. Install underthesea for correct results.",
                RuntimeWarning,
                stacklevel=2,
            )
            _WHITESPACE_WARNED = True
        return text.split()

    _TOKENIZER = _tok
    _TOKENIZER_NAME = "whitespace.split"
    return _TOKENIZER


def active_tokenizer() -> str:
    """Return the name of the tokenizer that will be used for F1."""
    _resolve_tokenizer()
    return _TOKENIZER_NAME


# --------------------------------------------------------------------------- #
# Normalisation + tokenisation
# --------------------------------------------------------------------------- #
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def normalize_text(text: str) -> str:
    """Normalise Vietnamese text for token comparison.

    Applies NFC Unicode normalisation, lower-casing, punctuation removal and
    whitespace collapsing. Underscores produced by segmenters (e.g. ``hôn_nhân``)
    are treated as word-internal and preserved by ``\\w``.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def vi_tokenize(text: str) -> list[str]:
    """Vietnamese-aware tokenization, returning a list of normalised tokens."""
    if not text:
        return []
    tokenizer = _resolve_tokenizer()
    # Segment first (segmenter needs original casing/diacritics), then normalise
    # each produced token so punctuation-only tokens are dropped.
    raw_tokens = tokenizer(text)
    tokens: list[str] = []
    for tok in raw_tokens:
        norm = normalize_text(tok).replace(" ", "_")
        if norm:
            tokens.append(norm)
    return tokens


# --------------------------------------------------------------------------- #
# Core metric
# --------------------------------------------------------------------------- #
def compute_token_f1(prediction: str, ground_truth: str) -> dict[str, float | int]:
    """Compute Vietnamese token-level Precision, Recall, F1 and Exact Match.

    Args:
        prediction: Model-generated answer text.
        ground_truth: Reference answer text.

    Returns:
        Dict with keys ``precision``, ``recall``, ``f1`` (floats in ``[0, 1]``)
        and ``exact_match`` (``0`` or ``1``).

    Edge cases:
        * Both empty            -> perfect match (f1 = 1, em = 1).
        * Exactly one empty     -> all zeros.
        * No common tokens      -> precision/recall/f1 = 0.
    """
    pred_norm = normalize_text(prediction)
    gold_norm = normalize_text(ground_truth)

    # Exact match is computed on normalised *surface* text (robust to spacing).
    exact_match = 1 if pred_norm == gold_norm and pred_norm != "" else 0
    if pred_norm == "" and gold_norm == "":
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact_match": 1}

    pred_tokens = vi_tokenize(prediction)
    gold_tokens = vi_tokenize(ground_truth)

    if not pred_tokens or not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": exact_match}

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum((pred_counts & gold_counts).values())

    if overlap == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": exact_match}

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match,
    }


def compute_corpus_token_f1(
    predictions: list[str],
    ground_truths: list[str],
) -> dict[str, float | int]:
    """Macro-average token F1 over a corpus of (prediction, reference) pairs.

    Args:
        predictions: List of generated answers.
        ground_truths: List of reference answers (same length).

    Returns:
        Dict with mean ``precision``, ``recall``, ``f1``, ``exact_match`` and
        ``n`` (number of pairs scored).
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"predictions ({len(predictions)}) and ground_truths "
            f"({len(ground_truths)}) must have equal length"
        )
    n = len(predictions)
    if n == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0, "n": 0}

    agg = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0}
    for pred, gold in zip(predictions, ground_truths):
        scores = compute_token_f1(pred or "", gold or "")
        for key in agg:
            agg[key] += float(scores[key])

    return {key: value / n for key, value in agg.items()} | {"n": n}
