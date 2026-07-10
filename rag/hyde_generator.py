from loguru import logger
from typing import Any

class HyDEGenerator:
    """Hypothetical Document Embeddings (HyDE) Generator.
    
    Uses the LLM to generate a hypothetical, ideal legal text snippet
    that answers the user's query. This snippet is then embedded and 
    used for vector search to bridge the vocabulary gap between user
    queries and formal legal text.
    """

    PROMPT_TEMPLATE = (
        "Viết một đoạn văn bản pháp luật giả định (sử dụng ngôn ngữ pháp lý chuẩn xác) "
        "để trả lời cho câu hỏi sau. Không cần giải thích, chỉ viết đoạn văn bản luật:\n\n"
        "Câu hỏi: {query}\n\n"
        "Đoạn văn bản luật giả định:"
    )

    def __init__(self, llm_client: Any, enabled: bool = False):
        self.llm = llm_client
        self.enabled = enabled
        if self.enabled:
            logger.info("HyDE Generator initialized and enabled.")

    def generate(self, query: str) -> str:
        """Generate a hypothetical document based on the query."""
        if not self.enabled:
            return query
            
        try:
            prompt = self.PROMPT_TEMPLATE.format(query=query)
            hypothetical_doc = self.llm.generate(prompt)
            # Safe strip thinking
            strip_fn = getattr(self.llm, "_strip_thinking", lambda x: x)
            hypothetical_doc = strip_fn(hypothetical_doc).strip()
            
            logger.debug("HyDE generated document: {}", hypothetical_doc)
            
            # Combine query and hypothetical doc for best of both worlds
            return f"{query}\n\n{hypothetical_doc}"
        except Exception as e:
            logger.warning("HyDE generation failed: {}. Falling back to raw query.", e)
            return query
