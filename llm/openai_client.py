"""OpenAI-compatible LLM client for remote inference.

Uses the standard /v1/chat/completions endpoint (compatible with vLLM,
LocalAI, LM Studio, etc.) to generate text. Implements the same
interface as OllamaClient for drop-in replacement.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests
import yaml
from loguru import logger


class OpenAIClient:
    """Client for OpenAI-compatible chat completion APIs.

    Works with any server implementing /v1/chat/completions:
    - vLLM, LocalAI, LMStudio, OpenAI, etc.
    - This project: Qwen/Qwen3-32B-AWQ via vLLM
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize client from config.

        Args:
            config: openai config dict. If None, loads from configs/config.yaml.
        """
        if config is None:
            config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                full_config = yaml.safe_load(f)
            config = full_config.get("openai", full_config.get("ollama", {}))

        self.base_url: str = config["base_url"].rstrip("/")
        self.model: str = config["model"]
        self.timeout: int = config.get("timeout_seconds", 120)
        self.max_retries: int = config.get("max_retries", 2)
        self.temperature: float = config.get("temperature", 0.1)
        self.max_tokens: int = config.get("max_tokens", 1024)
        api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
        self.api_key: str = (
            os.getenv(api_key_env)
            or os.getenv("OPENAI_API_KEY")
            or config.get("api_key")
            or "not-required"
        )

        logger.info(
            "OpenAIClient initialized | model={} | base_url={}",
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
        """Generate a text response via chat completions API.

        Args:
            prompt: The user message.
            system_prompt: Optional system instruction.
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.

        Returns:
            The assistant response text.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "stream": False,
            "thinking": False,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        logger.debug(
            "OpenAI generate | prompt_len={} | system_len={}",
            len(prompt),
            len(system_prompt) if system_prompt else 0,
        )

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 2):  # +2 so max_retries=2 gives 3 attempts
            try:
                # Strip /v1 if already present to avoid redundancy
                endpoint = self.base_url.rstrip("/")
                if not endpoint.endswith("/v1") and not endpoint.endswith("/v1/"):
                    endpoint = f"{endpoint}/v1"
                
                response = requests.post(
                    f"{endpoint}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                text = data["choices"][0]["message"]["content"]
                logger.debug(
                    "OpenAI response | attempt={} | response_len={}",
                    attempt,
                    len(text),
                )
                return text.strip()
            except (requests.RequestException, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning(
                    "OpenAI attempt {}/{} failed: {}",
                    attempt,
                    self.max_retries + 1,
                    exc,
                )
                if attempt <= self.max_retries:
                    time.sleep(2 ** attempt)

        raise RuntimeError(
            f"OpenAI generation failed after {self.max_retries + 1} attempts: {last_error}"
        )

    def generate_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Generate and parse a JSON response from the LLM.

        Args:
            prompt: Prompt instructing the LLM to output JSON.
            system_prompt: Optional system instruction.

        Returns:
            Parsed JSON as a dictionary.

        Raises:
            RuntimeError: If generation or JSON parsing fails.
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
        """Check if the API server is reachable via a minimal completion ping.

        Uses /v1/chat/completions with max_tokens=1 since /v1/models returns 404
        on some vLLM deployments.

        Returns:
            True if the server responds successfully.
        """
        try:
            endpoint = self.base_url.rstrip("/")
            if not endpoint.endswith("/v1"):
                endpoint = f"{endpoint}/v1"
                
            resp = requests.post(
                f"{endpoint}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "thinking": False,
                },
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                timeout=15,
            )
            resp.raise_for_status()
            logger.info("OpenAI API health check passed | model={}", self.model)
            return True
        except requests.RequestException as exc:
            logger.warning("OpenAI API health check failed: {}", exc)
            return False

    def _strip_thinking(self, text: str) -> str:
        """Remove <think>...</think> blocks emitted by Qwen3 reasoning mode.

        Args:
            text: Raw LLM output.

        Returns:
            Cleaned text without internal reasoning blocks.
        """
        import re
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.strip()
