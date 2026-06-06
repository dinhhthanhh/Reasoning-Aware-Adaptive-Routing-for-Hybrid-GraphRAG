"""i18n module for multi-language support in Hybrid GraphRAG."""

TEMPLATES = {
    "en": {
        "clarify_system": "You are a professional legal assistant.",
        "clarify_prompt": (
            "The user asked an ambiguous question: \"{query}\"\n\n"
            "History: {history}\n\n"
            "Please:\n"
            "1. Briefly summarize the general legal context for their topic (1-2 sentences).\n"
            "2. Explain why their current question is ambiguous.\n"
            "3. Ask clarifying questions to help them provide more detail.\n"
            "Respond in English, professionally and concisely."
        ),
        "no_context_found": "No relevant information found in the databases.",
        "vector_header": "=== DOCUMENT INFORMATION (VECTOR) ===",
        "graph_header": "=== GRAPH INFORMATION (GRAPH) ===",
        "no_graph_found": "No relevant information found in the knowledge graph.",
        "fallback_llm_fail": "Based on the context:\n\n{context}\n\n(Note: LLM failed to synthesize answer)",
    },
    "vi": {
        "clarify_system": "Bạn là chuyên gia tư vấn pháp luật Việt Nam.",
        "clarify_prompt": (
            "Bạn là một trợ lý luật sư ảo chuyên nghiệp và tận tâm.\n"
            "Người dùng đã gửi một câu hỏi mơ hồ: \"{query}\"\n"
            "Lịch sử hội thoại: {history}\n\n"
            "Yêu cầu:\n"
            "1. Hãy tóm tắt ngắn gọn (1-2 câu) quy định chung của luật pháp Việt Nam về chủ đề người dùng đang quan tâm.\n"
            "2. Sau đó, hãy giải thích tại sao câu hỏi hiện tại của họ chưa đủ rõ (ví dụ: thiếu chủ ngữ, thiếu văn bản luật cụ thể).\n"
            "3. Đặt các câu hỏi gợi ý để họ cung cấp thêm thông tin cần thiết.\n"
            "Phản hồi chuyên nghiệp, lịch sự và BẮT BUỘC bằng tiếng Việt."
        ),
        "no_context_found": "Không tìm thấy thông tin liên quan trong cả hai cơ sở dữ liệu.",
        "vector_header": "=== THÔNG TIN VĂN BẢN (VECTOR) ===",
        "graph_header": "=== THÔNG TIN ĐỒ THỊ (GRAPH) ===",
        "no_graph_found": "Không tìm thấy thông tin cụ thể trong đồ thị tri thức.",
        "fallback_llm_fail": "Dựa trên ngữ cảnh:\n\n{context}\n\n(Lưu ý: Không thể kết nối LLM để tổng hợp câu trả lời)",
    }
}

def get_template(language: str, key: str, **kwargs) -> str:
    """Retrieve a formatted template based on language and key.
    
    Args:
        language: Language code ('en' or 'vi').
        key: Template key.
        **kwargs: Formatting arguments.
        
    Returns:
        Formatted string template.
    """
    lang_templates = TEMPLATES.get(language, TEMPLATES["en"])
    template = lang_templates.get(key, TEMPLATES["en"][key])
    return template.format(**kwargs) if kwargs else template


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic Answer Formats (Task-specific prompt tailoring)
# ─────────────────────────────────────────────────────────────────────────────

ANSWER_FORMATS = {
    "short_factoid": {
        "system_suffix": (
            "Return ONLY the minimal answer string — a name, place, number, or short phrase. "
            "No explanation, no sentences, no 'Based on context'. "
            "BAD: 'Chief of Protocol of the United States' → GOOD: 'Chief of Protocol'. "
            "If context is insufficient, return your best short guess."
        ),
        "prompt_suffix": "Answer (one short phrase only):",
    },
    "yes_no": {
        "system_suffix": (
            "Return ONLY 'yes' or 'no'. Nothing else."
        ),
        "prompt_suffix": "Answer (yes or no only):",
    },
    "long_explanation": {
        "system_suffix": (
            "Provide a complete explanation based on the context. "
            "Include relevant details and reasoning. Write in full sentences."
        ),
        "prompt_suffix": "Answer:",
    },
    "legal_citation": {
        "system_suffix": (
            "Trả lời đầy đủ bằng tiếng Việt. Trích dẫn điều luật cụ thể nếu có. "
            "Giải thích rõ ràng, có cấu trúc."
        ),
        "prompt_suffix": "Trả lời:",
    },
    "legal_eval": {
        "system_suffix": (
            "Trả lời bằng tiếng Việt thật ngắn gọn để đánh giá QA tự động. "
            "Ưu tiên kết luận trực tiếp trong 1 câu hoặc 1 cụm từ. "
            "Chỉ nêu căn cứ pháp lý nếu ngữ cảnh có điều/khoản rõ ràng. "
            "Không mở đầu bằng 'Theo ngữ cảnh', không giải thích dài."
        ),
        "prompt_suffix": "Trả lời ngắn gọn:",
    },
}

def get_answer_format(task_type: str) -> dict:
    """Retrieve the prompt suffixes for a specific task type."""
    return ANSWER_FORMATS.get(task_type, ANSWER_FORMATS["short_factoid"])
