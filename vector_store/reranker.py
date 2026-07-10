import httpx
import os
from typing import List, Dict, Any
from loguru import logger

class VNLegalReranker:
    """Reranker using Infinity API running on Docker to save RAM.
    
    Or falls back to local sentence-transformers if configured.
    """
    def __init__(self, config: dict[str, Any] = None):
        if config is None:
            config = {}
            
        reranker_cfg = config.get("reranker", {})
        self.api_url = reranker_cfg.get("api_url", os.getenv("RERANK_API_URL", "http://localhost:7997/rerank"))
        self.model_name = reranker_cfg.get("model_name", "BAAI/bge-reranker-base")
        self.timeout = float(reranker_cfg.get("timeout_seconds", 2.0))
        self.disable_after_failure = reranker_cfg.get("disable_after_failure", True)
        self.enabled = reranker_cfg.get("enabled", False)
        
        self._disabled_reason = None
        if self.enabled:
            logger.info(f"Reranker enabled using API at {self.api_url} with model {self.model_name}")

    def rerank(self, query: str, documents: List[str], top_n: int = None) -> List[Dict[str, Any]]:
        """Rerank using Infinity API."""
        if not self.enabled or not documents:
            return self._fallback(documents, top_n)

        if self._disabled_reason:
            return self._fallback(documents, top_n)

        payload = {
            "query": query,
            "documents": documents,
            "model": self.model_name
        }
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(self.api_url, json=payload)
                response.raise_for_status()
                data = response.json()
                
                # Infinity format: {"results": [{"relevance_score": 0.9, "index": 0}, ...]}
                results = []
                for res in data.get("results", []):
                    results.append({
                        "index": res["index"],
                        "relevance_score": float(res["relevance_score"])
                    })
                    
                # Sort descending
                results.sort(key=lambda x: x["relevance_score"], reverse=True)
                
                if top_n:
                    results = results[:top_n]
                return results
                
        except Exception as e:
            if self.disable_after_failure:
                self._disabled_reason = str(e)
                logger.warning(
                    "Infinity Reranker failed: {}. "
                    "Disabling reranker for this process and returning original order.", e
                )
            else:
                logger.warning("Infinity Reranker failed: {}. Returning original order.", e)
            return self._fallback(documents, top_n)

    @staticmethod
    def _fallback(documents: List[str], top_n: int = None) -> List[Dict[str, Any]]:
        """Preserve original order when external reranker is unavailable or disabled."""
        results = [{"index": i, "relevance_score": 1.0 - (i / len(documents))} for i in range(len(documents))]
        if top_n:
            results = results[:top_n]
        return results
