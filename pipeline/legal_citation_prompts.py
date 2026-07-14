"""
legal_citation_prompts.py (v2)
===============================
Prompt templates cho Vietnamese Legal QA — trích dẫn NGUYÊN VĂN điều luật
nhưng có GIỚI HẠN PHẠM VI để tránh hai vấn đề phát sinh ở lần chạy trước:

  (a) Latency tăng 5-8x (vd Two-stage: 3.0s -> 20.4s) vì model bị buộc
      "trích NGUYÊN VĂN TỪNG CÂU CHỮ" toàn bộ tài liệu được cấp, thường
      là cả một Điều dài dù câu hỏi chỉ liên quan 1 khoản.
  (b) Token F1 vs concise_answer/gold_context giảm so với bản paraphrase
      cũ, vì output dài hơn nhiều -> Precision sụp đổ.

THAY ĐỔI SO VỚI v1:
  - Bỏ khối "[CHỈ THỊ ĐẶC BIỆT]" trong build_user_prompt (yêu cầu
    "PHẢI trích dẫn NGUYÊN VĂN TỪNG CÂU CHỮ" + một câu "không tìm thấy"
    KHÁC với system prompt). Hai chỉ thị chồng chéo/mâu thuẫn này là
    nguyên nhân chính khiến model dump toàn bộ tài liệu.
  - LEGAL_SYSTEM_PROMPT giờ yêu cầu: (1) chỉ trích khoản/điểm/đoạn TRỰC
    TIẾP liên quan — trích cả Điều khi không cần là SAI; (2) cho phép nêu
    câu trả lời ngắn (Có/Không/số liệu/ngày/tên cơ quan) TRƯỚC phần trích.
  - format_evidence_for_citation() làm sạch nhãn "article" nội bộ kiểu
    "14.5. Điều 14.5.NĐ.10.18. Hiệu lực của phép bay" -> "Điều 18. Hiệu
    lực của phép bay" để giảm nhiễu khi so khớp với gold_context.
  - Thống nhất MỘT câu duy nhất cho trường hợp "không tìm thấy".

Cách dùng:
    from pipeline.legal_citation_prompts import (
        LEGAL_SYSTEM_PROMPT,
        build_user_prompt,
        format_evidence_for_citation,
        wrap_evidence_string,
        CLARIFY_SYSTEM_PROMPT,
        build_clarify_prompt,
    )
"""
from __future__ import annotations

import re

# -----------------------------------------------------------------------
# Câu "không tìm thấy" — DUY NHẤT, dùng ở mọi nơi để tránh hai chỉ thị
# mâu thuẫn như ở v1 (system prompt nói một câu, user prompt nói câu khác).
# -----------------------------------------------------------------------
NOT_FOUND_PHRASE = (
    "Không tìm thấy quy định liên quan trong các văn bản được cung cấp."
)

# -----------------------------------------------------------------------
# SYSTEM PROMPT — trích nguyên văn CÓ GIỚI HẠN PHẠM VI
# -----------------------------------------------------------------------
LEGAL_SYSTEM_PROMPT = (
    "Bạn là trợ lý pháp lý chuyên về pháp luật Việt Nam.\n"
    "Nhiệm vụ của bạn là trả lời câu hỏi bằng cách TRÍCH DẪN NGUYÊN VĂN phần "
    "quy định pháp luật trực tiếp liên quan trong [TÀI LIỆU PHÁP LÝ]. KHÔNG "
    "paraphrase, KHÔNG tóm tắt lại nội dung quy phạm pháp luật bằng lời của "
    "bạn.\n\n"
    "QUY TẮC BẮT BUỘC:\n"
    "1. CHỈ trích nguyên văn từ [TÀI LIỆU PHÁP LÝ] bên dưới. Không bịa đặt, "
    "không suy luận thêm nội dung pháp lý ngoài tài liệu được cấp.\n"
    "2. PHẠM VI TRÍCH DẪN: chỉ trích khoản/điểm/đoạn TRỰC TIẾP trả lời câu "
    "hỏi. Nếu một Điều có nhiều khoản nhưng chỉ 1-2 khoản liên quan, CHỈ "
    "trích đúng các khoản đó. Trích toàn bộ Điều khi không cần thiết bị "
    "coi là TRẢ LỜI SAI (lan man, không tập trung).\n"
    "3. Ghi nguồn theo định dạng: Theo [Tên văn bản], [Điều/Khoản]: "
    "\"[nguyên văn phần liên quan]\"\n"
    "4. Nếu câu hỏi có thể trả lời trực tiếp bằng một câu/cụm từ ngắn (Có/"
    "Không, một con số, một mốc thời gian, tên cơ quan, tên văn bản...), "
    "hãy nêu câu trả lời ngắn đó TRƯỚC TIÊN, sau đó trích dẫn căn cứ pháp "
    "lý nguyên văn ngay tiếp theo để chứng minh.\n"
    f"5. Nếu [TÀI LIỆU PHÁP LÝ] không chứa quy định liên quan đến câu hỏi, "
    f"trả lời đúng MỘT câu duy nhất, không thêm gì khác: "
    f"\"{NOT_FOUND_PHRASE}\"\n"
    "6. TUYỆT ĐỐI KHÔNG tự suy diễn về tình trạng hiệu lực của văn bản. Nếu "
    "tài liệu không ghi rõ văn bản bị thay thế, bãi bỏ hay hết hiệu lực, KHÔNG "
    "được tự ý kết luận văn bản đã hết hiệu lực dựa vào kiến thức bên ngoài.\n"
)


# -----------------------------------------------------------------------
# Làm sạch nhãn "article" nội bộ (best-effort, an toàn: không khớp -> giữ
# nguyên). Một số chunk trả về nhãn dạng "14.5. Điều 14.5.NĐ.10.18. Hiệu
# lực của phép bay" (chứa ID nội bộ kiểu <chương>.<mục>.<LOẠI>.<văn
# bản>.<điều> hoặc <chương>.<mục>.<LOẠI>.<điều>). LLM được yêu cầu trích
# NGUYÊN VĂN nên sẽ echo lại cả phần ID này vào câu trả lời, tạo nhiễu khi
# so khớp với gold_context (thường chỉ viết "Điều 18. Hiệu lực của phép
# bay"). Hàm dưới đây rút gọn về "Điều N[. Tên điều]" khi nhận diện được.
# -----------------------------------------------------------------------
_ARTICLE_PATTERNS = [
    # "<x.y>.<LOẠI>.<a>.<b>. Tên điều" -> Điều b. Tên điều
    # vd "14.5.NĐ.10.18. Hiệu lực của phép bay" -> "Điều 18. Hiệu lực của phép bay"
    re.compile(r"^[\d.]+\.[A-ZĐ]+\.\d+\.(\d+)\.\s*(.*)$"),
    # "<x.y>.<LOẠI>.<n>. Tên điều" -> Điều n. Tên điều
    # vd "19.5.LQ.53. Cấp, sử dụng..." -> "Điều 53. Cấp, sử dụng..."
    re.compile(r"^[\d.]+\.[A-ZĐ]+\.(\d+)\.\s*(.*)$"),
]
_PREFIX_RE = re.compile(r"^(?:[\d.]+\.\s*)?Điều\s+(.*)$")


def clean_article_label(raw: str) -> str:
    """
    Chuẩn hoá nhãn "article" về dạng "Điều N[. Tên điều]" nếu nhận diện
    được pattern ID nội bộ đã biết. Best-effort: nếu không khớp pattern
    nào, trả về `raw` KHÔNG thay đổi (an toàn cho các nhãn đã sạch như
    "Điều 4, khoản 4").
    """
    if not raw:
        return raw
    m_prefix = _PREFIX_RE.match(raw.strip())
    body = m_prefix.group(1) if m_prefix else raw.strip()
    for pat in _ARTICLE_PATTERNS:
        m = pat.match(body)
        if m:
            num = m.group(1)
            title = (m.group(2) or "").strip()
            return "Điều " + num + (f". {title}" if title else "")
    return raw


# -----------------------------------------------------------------------
# Hàm format evidence — trình bày rõ ràng từng điều luật với metadata
# -----------------------------------------------------------------------
def format_evidence_for_citation(evidence_chunks: list[dict]) -> str:
    """
    Chuyển danh sách retrieved chunks thành block tài liệu có cấu trúc.

    Mỗi chunk nên có:
        - 'content'  : nội dung điều luật (str)
        - 'source'   : tên văn bản (str), ví dụ "Nghị định 100/2019/NĐ-CP"
        - 'article'  : số điều (str), ví dụ "Điều 5" (sẽ được làm sạch
                        qua clean_article_label() nếu chứa ID nội bộ)
        - 'doc_id'   : ID văn bản (str, optional)

    Trả về chuỗi văn bản được format sẵn để nhét vào prompt.
    """
    if not evidence_chunks:
        return "[Không có tài liệu nào được truy xuất]"

    blocks = []
    for i, chunk in enumerate(evidence_chunks, start=1):
        source = chunk.get("source", chunk.get("doc_id", f"Văn bản {i}"))
        article = clean_article_label(
            chunk.get("article", chunk.get("article_key", ""))
        )
        content = chunk.get("content", chunk.get("text", "")).strip()

        header = f"[Văn bản {i}] {source}"
        if article:
            header += f" — {article}"

        blocks.append(f"{header}\n{content}")

    return "\n\n".join(blocks)


# -----------------------------------------------------------------------
# Hàm build user prompt — ghép evidence + câu hỏi
#
# KHÔNG còn khối "[CHỈ THỊ ĐẶC BIỆT]" như v1. Mọi quy tắc (verbatim, phạm
# vi trích dẫn, câu "không tìm thấy") đã nằm trong LEGAL_SYSTEM_PROMPT.
# Lặp lại một bộ quy tắc KHÁC ở đây (như v1 đã làm) tạo ra hai chỉ thị
# chồng chéo và đẩy model về phía "trích càng nhiều càng an toàn".
# -----------------------------------------------------------------------
def build_user_prompt(
    query: str,
    evidence_chunks: list[dict],
    history: list[dict] | None = None,
) -> str:
    """
    Tạo user message để gửi cho LLM.

    Args:
        query          : câu hỏi của người dùng
        evidence_chunks: danh sách chunks từ retrieval (dense hoặc graph)
        history        : lịch sử hội thoại [(q, a), ...] hoặc None

    Returns:
        Chuỗi user message đã format
    """
    evidence_block = format_evidence_for_citation(evidence_chunks)

    history_block = ""
    if history:
        turns = []
        for turn in history[-3:]:  # chỉ lấy 3 lượt gần nhất
            q = turn.get("query", turn.get("question", ""))
            a = turn.get("answer", "")
            if q:
                turns.append(f"Người dùng: {q}")
            if a:
                turns.append(f"Trợ lý: {a}")
        if turns:
            history_block = (
                "LỊCH SỬ HỘI THOẠI GẦN NHẤT:\n" + "\n".join(turns) + "\n\n"
            )

    prompt = (
        f"{history_block}"
        f"[TÀI LIỆU PHÁP LÝ]\n"
        f"{evidence_block}\n\n"
        f"[CÂU HỎI]\n{query}\n\n"
        f"[TRẢ LỜI]\n"
    )
    return prompt


# -----------------------------------------------------------------------
# Helper: wrap plain string evidence -> list[dict]
# -----------------------------------------------------------------------
def wrap_evidence_string(
    evidence_str: str,
    source: str = "Văn bản pháp luật",
    article: str = "",
) -> list[dict]:
    """
    Chuyển evidence string thành list[dict] cho build_user_prompt.

    Thử tách thành nhiều đoạn nếu có nhiều điều luật (pattern "Điều X.").
    """
    if not evidence_str or not evidence_str.strip():
        return []

    chunks: list[dict] = []
    parts = re.split(r"\n(?=Điều\s+\d+[\.\s])", evidence_str)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        art_match = re.match(r"(Điều\s+\d+[\.\s]?)", part)
        art_name = art_match.group(1).strip() if art_match else article
        chunks.append({
            "content": part,
            "source": source,
            "article": art_name,
        })

    if not chunks:
        chunks = [{"content": evidence_str, "source": source, "article": article}]
    return chunks


# -----------------------------------------------------------------------
# Prompt đặc biệt cho CLARIFY route — không thay đổi so với v1, không liên
# quan tới vấn đề verbatim/latency (output ngắn, chỉ là 1 câu hỏi lại).
# -----------------------------------------------------------------------
CLARIFY_SYSTEM_PROMPT = (
    "Bạn là trợ lý pháp lý. Câu hỏi của người dùng chưa rõ đối tượng văn bản "
    "pháp lý cụ thể. Hãy đặt MỘT câu hỏi ngắn gọn để làm rõ, không hỏi nhiều "
    "hơn một lần. BẮT BUỘC: Bạn phải gợi ý từ 2 đến 4 đáp án cụ thể để người "
    "dùng chọn. Mỗi đáp án phải nằm trên một dòng riêng biệt, bắt đầu bằng "
    "định dạng chính xác '- [Option] '.\n"
    "LƯU Ý QUAN TRỌNG: Tuyệt đối không được lấy lại câu hỏi để làm thành "
    "đáp án gợi ý. Các đáp án gợi ý PHẢI là tên của các luật, nghị định, hoặc "
    "văn bản pháp luật cụ thể.\n"
    "Ví dụ:\n"
    "Bạn muốn hỏi theo quy định nào?\n"
    "- [Option] Luật Hôn nhân và gia đình\n"
    "- [Option] Bộ luật Dân sự\n"
)


def build_clarify_prompt(query: str, history: list[dict] | None = None) -> str:
    """Build user message for the clarification route."""
    history_block = ""
    if history:
        turns = [f"Người dùng: {t.get('query', '')}" for t in history[-2:]]
        history_block = "Lịch sử: " + " / ".join(turns) + "\n\n"
    return (
        f"{history_block}"
        f"Câu hỏi không rõ ràng: {query}\n\n"
        "Hỏi lại để làm rõ văn bản/điều luật/đối tượng cụ thể:"
    )
