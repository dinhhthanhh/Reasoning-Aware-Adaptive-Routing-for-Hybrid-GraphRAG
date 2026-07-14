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
from dotenv import load_dotenv


CYPHER_SYSTEM_PROMPT_VI = """Bạn là một chuyên gia về Neo4j Cypher. Nhiệm vụ của bạn là chuyển đổi câu hỏi bằng tiếng Việt sang câu truy vấn Cypher để tìm kiếm trong Graph Database Luật Việt Nam (Pháp điển).

CẤU TRÚC ĐỒ THỊ (Schema):
- Nodes: 
  + LegalDoc (doc_id, source, type)
  + LegalArticle (article_id, law_id, article_number, title, content, source)
  + Các node ngữ nghĩa: LEGAL_CONCEPT, ACTOR, AUTHORITY, OBLIGATION, RIGHT, PROCEDURE, CONDITION, PENALTY (tất cả đều có thuộc tính 'name')
- Relationships:
  + (LegalDoc)-[:HAS_ARTICLE]->(LegalArticle)
  + (LegalArticle)-[:REGULATES|GOVERNS|APPLIES_TO|OBLIGATES|PERMITS|PROHIBITS|REQUIRES...]->(Node Ngữ Nghĩa)
  + (Node Ngữ Nghĩa)-[:REFERS_TO|RELATED_CONCEPT...]->(Node Ngữ Nghĩa)

QUY TẮC BẮT BUỘC (RULES):
1. Chỉ trả về DUY NHẤT mã Cypher (không giải thích thêm, không bọc trong markdown).
2. Mọi query PHẢI có LIMIT không vượt quá 10 ở cuối cùng.
3. KHÔNG sử dụng các lệnh thay đổi dữ liệu (DELETE, MERGE, CREATE, SET, DROP). Chỉ dùng MATCH, CALL và RETURN.
4. Ưu tiên tìm kiếm sử dụng CONTAINS trên `content` hoặc `title` của LegalArticle. Trả về `node.title` và `node.content` hoặc thuộc tính tương đương.

VÍ DỤ (FEW-SHOT):
Question: Tìm Điều 39 chương 2 về hợp đồng lao động
Cypher: MATCH (a:LegalArticle) WHERE a.title =~ '(?i).*Điều 39.*' AND a.content =~ '(?i).*hợp đồng lao động.*' RETURN coalesce(a.title, "") AS title, coalesce(a.content, "") AS content LIMIT 10

Question: Quy định về giấy phép lao động
Cypher: MATCH (a:LegalArticle) WHERE toLower(a.content) CONTAINS toLower('giấy phép lao động') OR toLower(a.title) CONTAINS toLower('giấy phép lao động') RETURN coalesce(a.title, "") AS title, coalesce(a.content, "") AS content LIMIT 5
"""


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

        load_dotenv()
        self.base_url: str = (os.getenv("OPENAI_BASE_URL") or config["base_url"]).rstrip("/")
        self.model: str = os.getenv("OPENAI_MODEL") or config["model"]
        self.timeout: int = config.get("timeout_seconds", 120)
        self.max_retries: int = config.get("max_retries", 2)
        self.temperature: float = config.get("temperature", 0.1)
        self.max_tokens: int = config.get("max_tokens", 1024)
        api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
        self.api_key: str = (
            config.get("api_key")
            or os.getenv(api_key_env)
            or os.getenv("OPENAI_API_KEY")
            or "not-required"
        )

        logger.info(
            "OpenAIClient initialized | model={} | base_url_configured={}",
            self.model,
            bool(self.base_url),
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
        # Gemini:
        # If base_url points to Google's OpenAI-compatible shim
        # (e.g. .../v1beta/openai/), we must use OpenAI-compatible /chat/completions.
        # Otherwise, fall back to native Gemini generateContent.
        if isinstance(self.model, str) and self.model.lower().startswith("gemini-"):
            if "v1beta/openai" not in self.base_url and "/openai" not in self.base_url:
                return self._generate_gemini(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

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

        last_error: str | None = None
        for attempt in range(1, self.max_retries + 2):  # +2 so max_retries=2 gives 3 attempts
            try:
                # Strip /v1 if already present to avoid redundancy
                endpoint = self.base_url.rstrip("/")
                if not endpoint.endswith("/v1") and "/v1beta/openai" not in endpoint:
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
                last_error = self._format_error(exc)
                logger.warning(
                    "OpenAI attempt {}/{} failed: {}",
                    attempt,
                    self.max_retries + 1,
                    last_error,
                )
                if attempt <= self.max_retries:
                    if "429" in str(exc) or "429" in last_error:
                        logger.info("Rate limit hit (HTTP 429), sleeping 60 seconds...")
                        time.sleep(60)
                    else:
                        time.sleep(2 ** attempt)

        raise RuntimeError(
            f"OpenAI generation failed after {self.max_retries + 1} attempts: {last_error}"
        )

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """Generate a text response via streaming chat completions API.
        Yields text chunks as they are generated.
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
            "stream": True,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        logger.debug(
            "OpenAI generate_stream | prompt_len={} | system_len={}",
            len(prompt),
            len(system_prompt) if system_prompt else 0,
        )

        last_error: str | None = None
        for attempt in range(1, self.max_retries + 2):
            try:
                endpoint = self.base_url.rstrip("/")
                if not endpoint.endswith("/v1") and "/v1beta/openai" not in endpoint:
                    endpoint = f"{endpoint}/v1"
                
                with requests.post(
                    f"{endpoint}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                    stream=True
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            if decoded_line.startswith('data: '):
                                data_str = decoded_line[6:]
                                if data_str.strip() == '[DONE]':
                                    break
                                try:
                                    data = json.loads(data_str)
                                    if "choices" in data and len(data["choices"]) > 0:
                                        delta = data["choices"][0].get("delta", {})
                                        if "content" in delta:
                                            yield delta["content"]
                                except json.JSONDecodeError:
                                    continue
                return  # Stream completed successfully
            except (requests.RequestException, KeyError, json.JSONDecodeError) as exc:
                last_error = self._format_error(exc)
                logger.warning(
                    "OpenAI stream attempt {}/{} failed: {}",
                    attempt,
                    self.max_retries + 1,
                    last_error,
                )
                if attempt <= self.max_retries:
                    if "429" in str(exc) or "429" in last_error:
                        time.sleep(60)
                    else:
                        time.sleep(2 ** attempt)

        raise RuntimeError(
            f"OpenAI stream generation failed after {self.max_retries + 1} attempts: {last_error}"
        )

    def _generate_gemini(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Minimal Gemini integration using Google Generative Language API.

        This avoids requiring an OpenAI-compatible gateway for Gemini.
        """
        if not self.api_key or self.api_key == "not-required":
            raise RuntimeError("Gemini API key missing (set OPENAI_API_KEY env var).")

        model = self.model
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={self.api_key}"
        )

        merged_text = prompt
        if system_prompt:
            merged_text = f"{system_prompt}\n\n{prompt}"

        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": merged_text}]}],
            "generationConfig": {
                "temperature": temperature if temperature is not None else self.temperature,
                "maxOutputTokens": max_tokens if max_tokens is not None else self.max_tokens,
            },
        }

        headers = {"Content-Type": "application/json"}
        resp = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        try:
            return (
                data["candidates"][0]["content"]["parts"][0]["text"]
                .strip()
            )
        except Exception as exc:
            raise RuntimeError(f"Gemini response parse failed: {exc}") from exc

    @staticmethod
    def _format_error(exc: Exception) -> str:
        """Format API errors without leaking endpoint URLs or credentials."""
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            return f"HTTP {exc.response.status_code} {exc.response.reason}: {exc.response.text}"
        if isinstance(exc, requests.Timeout):
            return "request timed out"
        if isinstance(exc, requests.ConnectionError):
            # Include a short message for debugging (does not include credentials).
            # requests.ConnectionError may wrap lower-level socket errors.
            return f"connection error: {exc}"
        return exc.__class__.__name__

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

    def generate_cypher(self, query: str) -> str:
        """Generate a Cypher query from a natural language question.
        
        Args:
            query: The user's natural language query.
            
        Returns:
            A Cypher query string.
        """
        try:
            cypher = self.generate(
                prompt=f"Question: {query}\nCypher:",
                system_prompt=CYPHER_SYSTEM_PROMPT_VI,
                temperature=0.0,
                max_tokens=300
            )
            
            cypher = self._strip_thinking(cypher)
            
            if cypher.startswith("```cypher"):
                cypher = cypher.replace("```cypher", "").replace("```", "").strip()
            elif cypher.startswith("```"):
                cypher = cypher.replace("```", "").strip()
                
            if cypher.lower().startswith("cypher:"):
                cypher = cypher[7:].strip()
            elif cypher.lower().startswith("query:"):
                cypher = cypher[6:].strip()
                
            return cypher
        except Exception as e:
            logger.error("Failed to generate Cypher: {}", e)
            return ""

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
