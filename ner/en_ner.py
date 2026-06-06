"""English Named Entity Recognition module.

Uses generic English NER models (like dslim/bert-base-NER) for processing
benchmark datasets such as HotpotQA, MuSiQue, etc.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from loguru import logger
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from ner.vi_ner import Entity


class EnNER:
    """English Named Entity Recognition for standard benchmark datasets.

    Uses transformer-based NER (dslim/bert-base-NER or similar).
    Memory-efficient with batch processing.
    """

    ENTITY_TYPES: list[str] = [
        "PERSON",
        "ORGANIZATION",
        "LOCATION",
        "MISC",
    ]

    # Mapping from model labels to canonical labels
    LABEL_MAP: dict[str, str] = {
        "PER": "PERSON",
        "B-PER": "PERSON",
        "I-PER": "PERSON",
        "ORG": "ORGANIZATION",
        "B-ORG": "ORGANIZATION",
        "I-ORG": "ORGANIZATION",
        "LOC": "LOCATION",
        "B-LOC": "LOCATION",
        "I-LOC": "LOCATION",
        "MISC": "MISC",
        "B-MISC": "MISC",
        "I-MISC": "MISC",
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize NER model.

        Args:
            config: NER config dict.
        """
        if config is None:
            config_path = Path(__file__).resolve().parent.parent / "configs" / "config_en.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                full_config = yaml.safe_load(f)
            config = full_config.get("ner", {})

        self.model_name: str = config.get("model_name", "dslim/bert-base-NER")
        self.batch_size: int = config.get("batch_size", 8)
        self.max_length: int = config.get("max_length", 256)
        self.device: str = config.get("device", "auto")
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._pipeline: Any | None = None
        logger.info("EnNER initialized | model={} | device={}", self.model_name, self.device)

    def _load_model(self) -> None:
        """Lazy-load NER pipeline."""
        if self._pipeline is not None:
            return

        try:
            logger.info("Loading English NER model: {}", self.model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            self._pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                device=0 if self.device == "cuda" and torch.cuda.is_available() else -1,
                aggregation_strategy="simple",
            )
            logger.info("NER model loaded successfully: {}", self.model_name)
        except Exception as exc:
            logger.error("Failed to load NER model '{}': {}", self.model_name, exc)
            raise RuntimeError(f"Could not load NER model: {self.model_name}")

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
        """Extract entities from a single text."""
        entities: list[Entity] = []

        if self._pipeline is not None:
            try:
                truncated = text[: self.max_length * 4]  # rough char limit
                results = self._pipeline(truncated)
                for r in results:
                    label = self.LABEL_MAP.get(
                        r.get("entity_group", r.get("entity", "")),
                        "MISC"
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

        # Rule-based extraction for quoted entities (common in HotpotQA)
        quoted_entities = self._extract_quoted(text)
        entities.extend(quoted_entities)

        return self._deduplicate(entities)

    def _extract_quoted(self, text: str) -> list[Entity]:
        """Extract entities enclosed in double quotes."""
        entities: list[Entity] = []
        # Matches text inside double quotes: "Entity Name"
        pattern = re.compile(r'"([^"]+)"')
        for match in pattern.finditer(text):
            entities.append(Entity(
                text=match.group(1).strip(),
                label="MISC",
                start=match.start(),
                end=match.end(),
                confidence=0.90
            ))
        return entities

    @staticmethod
    def _deduplicate(entities: list[Entity]) -> list[Entity]:
        """Remove duplicate entities by preferring higher confidence."""
        if not entities:
            return entities

        entities.sort(key=lambda e: (e.start, -e.confidence))
        result: list[Entity] = [entities[0]]
        for ent in entities[1:]:
            prev = result[-1]
            if ent.start < prev.end:
                if ent.confidence > prev.confidence:
                    result[-1] = ent
            else:
                result.append(ent)
        return result
