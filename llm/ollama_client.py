"""Ollama LLM client wrapper for local inference.

Provides HTTP-based communication with Ollama server for text generation
and structured JSON output. Handles retries, timeouts, and health checks.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import requests
import yaml
from loguru import logger


class OllamaClient:
    """Wrapper for Ollama local LLM inference.

    Uses the Ollama REST API to generate text responses. Supports
    plain-text and structured JSON generation with automatic retry
    and timeout handling.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize Ollama client from config.

        Args:
            config: Ollama config dict. If None, loads from configs/config.yaml.
        """
        if config is None:
            config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                full_config = yaml.safe_load(f)
            config = full_config["ollama"]

        self.base_url: str = config["base_url"].rstrip("/")
        self.model: str = config["model"]
        self.timeout: int = config.get("timeout_seconds", 120)
        self.max_retries: int = config.get("max_retries", 2)
        self.temperature: float = config.get("temperature", 0.1)
        self.max_tokens: int = config.get("max_tokens", 1024)

        logger.info(
            "OllamaClient initialized | model={} | base_url={}",
            self.model,
            self.base_url,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a text response from the LLM.

        Args:
            prompt: The user prompt to send.
            system_prompt: Optional system-level instruction.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            The generated text response.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_predict": max_tokens if max_tokens is not None else self.max_tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        logger.debug(
            "Ollama generate | prompt_len={} | system_len={}",
            len(prompt),
            len(system_prompt) if system_prompt else 0,
        )

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                text = data.get("response", "")
                logger.debug(
                    "Ollama response | attempt={} | response_len={}",
                    attempt,
                    len(text),
                )
                return text.strip()
            except (requests.RequestException, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning(
                    "Ollama attempt {}/{} failed: {}",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

        raise RuntimeError(
            f"Ollama generation failed after {self.max_retries} attempts: {last_error}"
        )

    def generate_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Generate and parse a JSON response from the LLM.

        Used by the LLM Reasoning Verifier for structured routing decisions.
        Handles malformed JSON by attempting regex-based extraction.

        Args:
            prompt: Prompt instructing the LLM to output JSON.
            system_prompt: Optional system instruction.

        Returns:
            Parsed JSON as a dictionary.

        Raises:
            RuntimeError: If generation or JSON parsing fails after fallbacks.
        """
        raw = self.generate(prompt, system_prompt=system_prompt)

        # Attempt 1: direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("Direct JSON parse failed, trying extraction")

        # Attempt 2: extract JSON block from markdown code fence
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                logger.debug("Code-fence JSON extraction failed")

        # Attempt 3: find first { ... } block
        brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                logger.debug("Brace-extraction JSON parse failed")

        logger.error("Failed to parse JSON from LLM response: {}", raw[:500])
        raise RuntimeError(f"Could not parse JSON from LLM response: {raw[:200]}")

    def health_check(self) -> bool:
        """Check if Ollama server is running and model is available.

        Returns:
            True if the server responds and the model is loaded.
        """
        try:
            # Check server is alive
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]

            # Check if our model is available (handle tag suffix)
            available = any(
                self.model in name or name.startswith(self.model.split(":")[0])
                for name in model_names
            )
            if available:
                logger.info("Ollama health check passed | model={}", self.model)
            else:
                logger.warning(
                    "Ollama running but model '{}' not found. Available: {}",
                    self.model,
                    model_names,
                )
            return available
        except requests.RequestException as exc:
            logger.error("Ollama health check failed: {}", exc)
            return False
