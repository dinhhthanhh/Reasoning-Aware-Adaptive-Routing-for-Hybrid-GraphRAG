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
            "Bạn là một trợ lý AI tư vấn pháp luật Việt Nam có tên là 'AI Legal', luôn chuyên nghiệp, vui vẻ và tận tâm.\n"
            "Lịch sử hội thoại: {history}\n\n"
            "Người dùng vừa nhắn: \"{query}\"\n\n"
            "Yêu cầu:\n"
            "- Nếu người dùng chỉ đang chào hỏi đơn thuần (ví dụ: 'hi', 'chào', 'hello'), hãy đáp lại một cách thân thiện, vui vẻ và chủ động hỏi xem họ cần tư vấn vấn đề pháp lý gì.\n"
            "- Nếu người dùng đang hỏi một vấn đề pháp lý nhưng câu hỏi quá ngắn, mơ hồ hoặc thiếu thông tin, hãy làm theo 3 bước sau:\n"
            "  1. Tóm tắt rất ngắn gọn (1-2 câu) góc nhìn chung của pháp luật Việt Nam về chủ đề đó.\n"
            "  2. Giải thích nhẹ nhàng tại sao câu hỏi chưa đủ để đưa ra câu trả lời chính xác (ví dụ: thiếu chi tiết tình huống cụ thể).\n"
            "  3. Gợi ý họ cung cấp thêm thông tin.\n"
            "Phản hồi tự nhiên, lịch sự và BẮT BUỘC bằng tiếng Việt."
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
            "CHÚ Ý ĐẶC BIỆT: BẤT KỂ câu hỏi là gì, BẮT BUỘC chép y nguyên TOÀN BỘ nội dung của điều luật (bao gồm tất cả các khoản, điểm) từ ngữ cảnh. "
            "TUYỆT ĐỐI KHÔNG tóm tắt, KHÔNG giải thích, KHÔNG diễn đạt lại hay cắt xén."
        ),
        "prompt_suffix": "Trả lời TOÀN BỘ nguyên văn:",
    },
    "legal_reasoning": {
        "system_suffix": (
            "Bạn là chuyên gia tư vấn pháp luật Việt Nam. Người dùng là người không có chuyên môn pháp lý.\n\n"
            "Nguyên tắc bắt buộc:\n"
            "1. LUÔN ưu tiên thông tin từ ngữ cảnh pháp luật được cung cấp bên dưới. "
            "Chỉ dùng kiến thức nền khi ngữ cảnh không đủ, và phải ghi rõ 'theo hiểu biết chung'.\n"
            "2. Phân tích tình huống theo các bước: (a) xác định lĩnh vực pháp luật liên quan, "
            "(b) ánh xạ tình huống vào điều luật cụ thể từ ngữ cảnh, "
            "(c) giải thích ý nghĩa bằng ngôn ngữ đơn giản, "
            "(d) đưa ra kết luận và hành động cụ thể người dùng nên làm.\n"
            "3. Không chỉ nêu vi phạm — hãy phân tích theo góc nhìn phù hợp với câu hỏi: "
            "quyền lợi, nghĩa vụ, thủ tục, trách nhiệm, hay vi phạm.\n"
            "4. Nếu ngữ cảnh không có đủ căn cứ pháp lý, hãy nói rõ và khuyên người dùng "
            "tham khảo luật sư hoặc cơ quan có thẩm quyền.\n"
            "5. Trả lời bằng tiếng Việt, rõ ràng, dễ hiểu với người không có nền tảng pháp luật."
        ),
        "prompt_suffix": "Phân tích tình huống pháp lý và tư vấn cụ thể:",
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
