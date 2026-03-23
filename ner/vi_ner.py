"""Vietnamese Named Entity Recognition module.

Uses NlpHUST/ner-vietnamese-electra-base from HuggingFace for NER,
with fallback to vinai/phobert-base. Includes rule-based extraction
for Vietnamese legal terms (Điều, Khoản, Luật, Nghị định, etc.).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from loguru import logger
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


@dataclass
class Entity:
    """A single named entity extracted from text.

    Attributes:
        text: The entity surface form.
        label: Entity type (PERSON, ORGANIZATION, LOCATION, LEGAL_TERM, DATE).
        start: Character start offset in the original text.
        end: Character end offset in the original text.
        confidence: Model confidence score (0-1).
    """

    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0


class ViNER:
    """Vietnamese Named Entity Recognition with legal-term support.

    Combines transformer-based NER with rule-based extraction of
    Vietnamese legal terms. Memory-efficient with batch processing.
    """

    ENTITY_TYPES: list[str] = [
        "PERSON",
        "ORGANIZATION",
        "LOCATION",
        "LEGAL_TERM",
        "DATE",
    ]

    # Rule-based patterns for Vietnamese legal terms
    LEGAL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
        ("LEGAL_TERM", re.compile(r"Điều\s+\d+[a-zđ]?", re.IGNORECASE)),
        ("LEGAL_TERM", re.compile(r"Khoản\s+\d+", re.IGNORECASE)),
        ("LEGAL_TERM", re.compile(r"Điểm\s+[a-zđ]", re.IGNORECASE)),
        ("LEGAL_TERM", re.compile(
            r"(?:Bộ\s+luật|Luật|Nghị\s+định|Thông\s+tư|Quyết\s+định|Nghị\s+quyết)"
            r"\s+(?:số\s+)?\d*[^,.\n]{0,80}",
            re.IGNORECASE,
        )),
        ("LEGAL_TERM", re.compile(
            r"(?:Bộ\s+luật|Luật)\s+[A-ZĐÀÁẢÃẠÈÉẺẼẸÌÍỈĨỊÒÓỎÕỌÙÚỦŨỤỲÝỶỸỴÂĂÊÔƠƯa-zđàáảãạèéẻẽẹìíỉĩịòóỏõọùúủũụỳýỷỹỵâăêôơư]"
            r"[^,.\n]{0,60}",
        )),
    ]

    # Mapping from model labels to our canonical labels
    LABEL_MAP: dict[str, str] = {
        "PER": "PERSON",
        "B-PER": "PERSON",
        "I-PER": "PERSON",
        "PERSON": "PERSON",
        "ORG": "ORGANIZATION",
        "B-ORG": "ORGANIZATION",
        "I-ORG": "ORGANIZATION",
        "ORGANIZATION": "ORGANIZATION",
        "LOC": "LOCATION",
        "B-LOC": "LOCATION",
        "I-LOC": "LOCATION",
        "LOCATION": "LOCATION",
        "MISC": "LEGAL_TERM",
        "B-MISC": "LEGAL_TERM",
        "I-MISC": "LEGAL_TERM",
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize NER model.

        Args:
            config: NER config dict. If None, loads from configs/config.yaml.
        """
        if config is None:
            config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                full_config = yaml.safe_load(f)
            config = full_config["ner"]

        self.model_name: str = config["model_name"]
        self.batch_size: int = config.get("batch_size", 8)
        self.max_length: int = config.get("max_length", 256)
        self.device: str = config.get("device", "auto")
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._pipeline: Any | None = None
        logger.info("ViNER initialized | model={} | device={}", self.model_name, self.device)

    def _load_model(self) -> None:
        """Lazy-load NER pipeline. Falls back to phobert if primary model unavailable."""
        if self._pipeline is not None:
            return

        models_to_try = [
            self.model_name,
            "vinai/phobert-base",
        ]

        for model_name in models_to_try:
            try:
                logger.info("Loading NER model: {}", model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForTokenClassification.from_pretrained(model_name)
                self._pipeline = pipeline(
                    "ner",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if self.device == "cuda" and torch.cuda.is_available() else -1,
                    aggregation_strategy="simple",
                )
                logger.info("NER model loaded successfully: {}", model_name)
                return
            except Exception as exc:
                logger.warning("Failed to load NER model '{}': {}", model_name, exc)

        raise RuntimeError("Could not load any NER model.")

    def extract(self, texts: list[str]) -> list[list[Entity]]:
        """Extract named entities from a batch of texts.

        Args:
            texts: List of text strings to process.

        Returns:
            List of entity lists, one per input text.
        """
        self._load_model()

        all_entities: list[list[Entity]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            for text in batch:
                entities = self._extract_single(text)
                all_entities.append(entities)

        logger.debug(
            "NER extracted entities from {} texts | total_entities={}",
            len(texts),
            sum(len(e) for e in all_entities),
        )
        return all_entities

    def _extract_single(self, text: str) -> list[Entity]:
        """Extract entities from a single text.

        Args:
            text: Input text.

        Returns:
            List of Entity objects.
        """
        entities: list[Entity] = []

        # Transformer-based NER
        if self._pipeline is not None:
            try:
                truncated = text[: self.max_length * 4]  # rough char limit
                results = self._pipeline(truncated)
                for r in results:
                    label = self.LABEL_MAP.get(
                        r.get("entity_group", r.get("entity", "")),
                        r.get("entity_group", r.get("entity", "MISC")),
                    )
                    entities.append(Entity(
                        text=r.get("word", "").strip(),
                        label=label,
                        start=r.get("start", 0),
                        end=r.get("end", 0),
                        confidence=float(r.get("score", 0.0)),
                    ))
            except Exception as exc:
                logger.warning("Transformer NER failed for text: {}", exc)

        # Rule-based legal term extraction
        legal_entities = self.extract_legal_terms(text)
        entities.extend(legal_entities)

        # Deduplicate by span overlap
        entities = self._deduplicate(entities)
        return entities

    def extract_legal_terms(self, text: str) -> list[Entity]:
        """Extract Vietnamese legal terms using rule-based patterns.

        Detects patterns like 'Điều X', 'Khoản Y', 'Luật Z', 'Nghị định N'.

        Args:
            text: Input Vietnamese text.

        Returns:
            List of Entity objects with label 'LEGAL_TERM'.
        """
        entities: list[Entity] = []
        seen_spans: set[tuple[int, int]] = set()

        for label, pattern in self.LEGAL_PATTERNS:
            for match in pattern.finditer(text):
                span = (match.start(), match.end())
                # Skip if overlapping with existing span
                if any(
                    s[0] <= span[0] < s[1] or s[0] < span[1] <= s[1]
                    for s in seen_spans
                ):
                    continue
                seen_spans.add(span)
                entities.append(Entity(
                    text=match.group().strip(),
                    label=label,
                    start=span[0],
                    end=span[1],
                    confidence=0.95,
                ))

        return entities

    @staticmethod
    def _deduplicate(entities: list[Entity]) -> list[Entity]:
        """Remove duplicate entities by preferring higher confidence.

        Args:
            entities: List of entities, possibly overlapping.

        Returns:
            Deduplicated entity list.
        """
        if not entities:
            return entities

        # Sort by start position, then by confidence descending
        entities.sort(key=lambda e: (e.start, -e.confidence))
        result: list[Entity] = [entities[0]]
        for ent in entities[1:]:
            prev = result[-1]
            # If overlapping, keep the higher-confidence one
            if ent.start < prev.end:
                if ent.confidence > prev.confidence:
                    result[-1] = ent
            else:
                result.append(ent)
        return result
