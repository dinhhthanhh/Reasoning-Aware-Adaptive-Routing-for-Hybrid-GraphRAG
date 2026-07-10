import re
from typing import Dict, List


# ─────────────────────────────────────────────────────────────────────────────
# Vietnamese legal keyword sets
# ─────────────────────────────────────────────────────────────────────────────

# Keywords that strongly indicate graph traversal (multi-hop, same-doc)
# Deliberately excludes generic legal vocabulary like "quy định", "thẩm quyền"
# that appear in ALL question types and cause noise.
GRAPH_KEYWORDS_VI: List[str] = [
    "điều kiện",
    "trường hợp",
    "trừ khi",
    "ngoại trừ",
    "được quy định tại",
    "theo quy định tại",
    "căn cứ vào",
    "dẫn chiếu",
    "áp dụng.*điều",
    "thực hiện theo",
    "hướng dẫn tại",
    "liên quan đến.*điều",
    "quy định tại khoản",
    "quy định tại điểm",
    "quy định tại điều",
]

# Keywords strongly indicating hybrid_reasoning (cross-doc multi-hop)
# These patterns reference two conceptually different domains in one query.
CROSS_DOC_SIGNALS_VI: List[str] = [
    # Connectives that bridge two separate legal provisions
    "đồng thời",
    "bên cạnh đó",
    "ngoài ra",
    "trong khi đó",
    # Multiple law/decree references in one sentence
    r"\d+/\d{4}/[A-ZĐ\-]+.*và.*\d+/\d{4}/[A-ZĐ\-]+",
    r"nghị định.*và.*thông tư",
    r"luật.*và.*nghị định",
    r"quyết định.*và.*nghị định",
    r"thông tư.*và.*luật",
    r"chỉ thị.*và.*nghị định",
]

# Legal reference patterns for Vietnamese documents
# Covers: NĐ-CP, QĐ-TTg, QĐ-BTC, QĐ-BCT, TT-BKHCN, VBHN-VPQH,
#         VBHN-BXD, VBHN-BYT, CT-TTg, CĐ-BXD, HD-UBTVQH, QĐ-UBND, etc.
LEGAL_REF_PATTERNS: List[str] = [
    r"\d+/\d{4}/[A-ZĐ\-]{2,}",      # e.g. 77/2026/NĐ-CP, 10/2015/TT-BKHCN
    r"\d+/[A-ZĐ\-]{2,}-[A-ZĐ]+",    # e.g. 12/VBHN-BXD, 10/VBHN-BYT
    r"\d+/[A-ZĐ\-]{2,}",            # e.g. 464/QĐ-BCT, 441/QĐ-TTg
    r"[Nn]ghị\s+định\s+(?:số\s+)?\d+",
    r"[Tt]hông\s+tư\s+(?:số\s+)?\d+",
    r"[Qq]uyết\s+định\s+(?:số\s+)?\d+",
    r"[Ll]uật\s+[A-ZĐÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ]",
    r"[Cc]hỉ\s+thị\s+(?:số\s+)?\d+",
]

# Điều reference patterns — detecting references to specific articles
DIEU_PATTERNS: List[str] = [
    r"[Ðđ][Ii][Ềề][Uu]\s+\d+",
    r"[Ðđ]i[eề]u\s+\d+",            # ASCII fallback
    r"kho[aả]n\s+\d+\s+[Ðđ][Iiì][Ềề][Uu]",
    r"[Ðđ]i[eề]u\s+\d+.*[Ðđ]i[eề]u\s+\d+",   # two Điều in same query
]


# ─────────────────────────────────────────────────────────────────────────────
# Yes/No question detection
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that identify yes/no question structure in Vietnamese.
# These are the STRONGEST single predictor of hybrid_reasoning in the benchmark:
# 68.7% of all yes_no questions are hybrid_reasoning.
YES_NO_PATTERNS: List[str] = [
    r"\bcó\b.{1,60}\bkhông\b",            # "có ... không"
    r"\bliệu\b.{1,60}\bkhông\b",          # "liệu ... không"
    r"\bđúng\b.{0,20}\bkhông\b",          # "đúng không"
    r"\bsai\b.{0,20}\bkhông\b",           # "sai không"
    r"\bđược\b.{0,40}\bhay\s+không\b",    # "được ... hay không"
    r"\bphải\b.{0,40}\bhay\s+không\b",    # "phải ... hay không"
    r"\bcó\s+thể\b.{0,40}\bkhông\b",      # "có thể ... không"
    r"\bcó\s+được\b.{0,40}\bkhông\b",     # "có được ... không"
    r"\bnhư\s+vậy\b.{0,50}\bkhông\b",     # "như vậy ... không"
    r"\bviệc.{0,40}\bcó\b.{0,30}\bkhông\b",  # "việc ... có ... không"
]


# ─────────────────────────────────────────────────────────────────────────────
# Factoid question detection (strong predictor of dense_retrieval)
# ─────────────────────────────────────────────────────────────────────────────

FACTOID_PATTERNS: List[str] = [
    r"^[Aa]i\b",                          # "Ai là..."
    r"^[Cc]ái\s+gì\b",                    # "Cái gì..."
    r"^[Gg]ì\b",
    r"\blà\s+gì\b",                       # "... là gì?"
    r"\bđược\s+định\s+nghĩa\b",           # definition questions
    r"\bkhái\s+niệm\b",
    r"\bý\s+nghĩa\b",
    r"\bđịnh\s+nghĩa\b",
    r"^[Nn]hững?\s+",                     # "Những..." (listing)
    r"^[Cc]ác\s+",                        # "Các..." (enumeration)
    r"\btên\s+g[oọ]i\b",
    r"\bký\s+hiệu\b",
    r"\bthuộc\s+loại\b",
    r"\bbao\s+nhiêu\b",
    r"\bmức\s+phạt\b",
    r"\bmức\s+xử\s+phạt\b",
    r"\bphạt\s+bao\s+nhiêu\b",
    r"\bthời\s+hạn\b",
    r"\bthời\s+gian\b",
    r"\bbao\s+lâu\b",
]


# ─────────────────────────────────────────────────────────────────────────────
# Ambiguity / clarification detection
# ─────────────────────────────────────────────────────────────────────────────

AMBIGUITY_PATTERNS: List[str] = [
    r"\bkhông\s+rõ\b",
    r"\bchưa\s+rõ\b",
    r"\bý\s+kiến\b",
    r"\btheo\s+bạn\b",
    r"\btheo\s+ý\s+kiến\b",
    r"\bgiải\s+thích\s+giúp\b",
    r"\bcó\s+nghĩa\s+là\s+gì\b",
]


# ─────────────────────────────────────────────────────────────────────────────
# Feature extractor class
# ─────────────────────────────────────────────────────────────────────────────

class VietnameseLegalFeatureExtractor:
    """
    Extracts a fixed-length numerical feature vector from a Vietnamese legal
    query.  Features are designed to discriminate between the three routing
    targets:  dense_retrieval, graph_traversal, hybrid_reasoning.

    Feature vector (16 dimensions, all non-negative scalars):
        0  query_length_chars
        1  query_length_words
        2  graph_keyword_count
        3  legal_reference_count
        4  dieu_reference_count
        5  multi_article_ref          (>= 2 Điều refs in query)
        6  is_yes_no_question         (binary; strongest hybrid signal)
        7  is_factoid_question        (binary; strongest dense signal)
        8  cross_doc_signal_count     (multi-doc language patterns)
        9  has_conditional_structure  (nếu/khi/trường hợp patterns)
        10 has_negation               (không, chưa, chẳng)
        11 has_comparison             (so với, khác biệt, hơn, tương tự)
        12 has_enumeration            (liệt kê, bao gồm, gồm có)
        13 has_procedure_marker       (thủ tục, quy trình, các bước, hướng dẫn)
        14 ambiguity_score            (count of ambiguity pattern matches)
        15 has_multi_hop_connector    (và, hoặc, đồng thời between two legal refs)
    """

    def __init__(self):
        # Pre-compile all patterns for efficiency
        flags = re.IGNORECASE | re.UNICODE

        self._graph_kw_patterns = [re.compile(p, flags) for p in GRAPH_KEYWORDS_VI]
        self._cross_doc_patterns = [re.compile(p, flags) for p in CROSS_DOC_SIGNALS_VI]
        self._legal_ref_patterns = [re.compile(p, flags) for p in LEGAL_REF_PATTERNS]
        self._dieu_patterns      = [re.compile(p, flags) for p in DIEU_PATTERNS]
        self._yes_no_patterns    = [re.compile(p, flags) for p in YES_NO_PATTERNS]
        self._factoid_patterns   = [re.compile(p, flags) for p in FACTOID_PATTERNS]
        self._ambiguity_patterns = [re.compile(p, flags) for p in AMBIGUITY_PATTERNS]

        self._conditional_pattern = re.compile(
            r"\b(nếu|khi|trường\s+hợp|trong\s+trường\s+hợp|giả\s+sử|giả\s+định)\b",
            flags,
        )
        self._negation_pattern = re.compile(
            r"\b(không|chưa|chẳng|chẳng\s+phải|không\s+phải|không\s+được)\b",
            flags,
        )
        self._comparison_pattern = re.compile(
            r"\b(so\s+với|khác\s+biệt|hơn|tương\s+tự|giống|khác\s+nhau|phân\s+biệt)\b",
            flags,
        )
        self._enumeration_pattern = re.compile(
            r"\b(liệt\s+kê|bao\s+gồm|gồm\s+có|bao\s+gồm\s+những|các\s+trường\s+hợp)\b",
            flags,
        )
        self._procedure_pattern = re.compile(
            r"\b(thủ\s+tục|quy\s+trình|các\s+bước|hướng\s+dẫn|quy\s+định\s+về\s+việc|"
            r"trình\s+tự|quy\s+trình\s+thực\s+hiện)\b",
            flags,
        )
        self._multi_hop_connector = re.compile(
            r"(điều\s+\d+.{0,20}(?:và|hoặc|cùng\s+với).{0,20}điều\s+\d+|"
            r"\d+/\d{4}/.{0,10}(?:và|hoặc).{0,10}\d+/\d{4}/)",
            flags,
        )

    # ── Core count helpers ────────────────────────────────────────────────────

    def _count_matches(self, patterns, text: str) -> int:
        return sum(1 for p in patterns if p.search(text))

    def _count_legal_refs(self, text: str) -> int:
        """
        Count unique legal document references in the query.
        Use a set to avoid counting the same reference multiple times across
        overlapping patterns.
        """
        matches = set()
        for p in self._legal_ref_patterns:
            for m in p.finditer(text):
                matches.add(m.group().strip().lower())
        return len(matches)

    def _count_dieu_refs(self, text: str) -> int:
        """Count references to specific Điều (article numbers)."""
        # Use the most inclusive pattern, deduplicate by position
        combined = re.compile(r"[Ðđ]i[eềèé]\s*u\s+\d+", re.IGNORECASE | re.UNICODE)
        return len(combined.findall(text))

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, query: str) -> Dict[str, float]:
        """
        Extract features from a single query string.

        Returns a dict {feature_name: float_value}.
        """
        text = query.strip()
        words = text.split()

        graph_kw_count   = self._count_matches(self._graph_kw_patterns, text)
        legal_ref_count  = self._count_legal_refs(text)
        dieu_ref_count   = self._count_dieu_refs(text)
        multi_article    = 1.0 if dieu_ref_count >= 2 else 0.0
        is_yes_no        = 1.0 if self._count_matches(self._yes_no_patterns, text) > 0 else 0.0
        is_factoid       = 1.0 if self._count_matches(self._factoid_patterns, text) > 0 else 0.0
        cross_doc        = float(self._count_matches(self._cross_doc_patterns, text))
        has_conditional  = 1.0 if self._conditional_pattern.search(text) else 0.0
        has_negation     = 1.0 if self._negation_pattern.search(text) else 0.0
        has_comparison   = 1.0 if self._comparison_pattern.search(text) else 0.0
        has_enumeration  = 1.0 if self._enumeration_pattern.search(text) else 0.0
        has_procedure    = 1.0 if self._procedure_pattern.search(text) else 0.0
        ambiguity_score  = float(self._count_matches(self._ambiguity_patterns, text))
        multi_hop_conn   = 1.0 if self._multi_hop_connector.search(text) else 0.0

        return {
            "query_length_chars":       float(len(text)),
            "query_length_words":       float(len(words)),
            "graph_keyword_count":      float(graph_kw_count),
            "legal_reference_count":    float(legal_ref_count),
            "dieu_reference_count":     float(dieu_ref_count),
            "multi_article_ref":        multi_article,
            "is_yes_no_question":       is_yes_no,
            "is_factoid_question":      is_factoid,
            "cross_doc_signal_count":   cross_doc,
            "has_conditional_structure": has_conditional,
            "has_negation":             has_negation,
            "has_comparison":           has_comparison,
            "has_enumeration":          has_enumeration,
            "has_procedure_marker":     has_procedure,
            "ambiguity_score":          ambiguity_score,
            "has_multi_hop_connector":  multi_hop_conn,
        }

    def extract_batch(self, queries: List[str]) -> List[Dict[str, float]]:
        return [self.extract(q) for q in queries]

    def feature_names(self) -> List[str]:
        return list(self.extract("dummy").keys())


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    extractor = VietnameseLegalFeatureExtractor()

    test_cases = [
        # (query, expected_route)
        ("Ai có thẩm quyền quyết định thành lập đơn vị trực thuộc Cục Hoá chất?",
         "dense_retrieval"),

        ("Trong trường hợp chuyến bay cất cánh từ sân bay dự bị, thời gian hiệu lực "
         "của phép bay được tính như thế nào và ai chịu trách nhiệm?",
         "graph_traversal"),

        ("Nếu một địa phương chưa bố trí được nhân sự giữ chức Chủ tịch Hội đồng nhân "
         "dân đúng hạn theo Nghị định 77/2026/NĐ-CP và Luật Giao dịch điện tử, "
         "có được gia hạn không?",
         "hybrid_reasoning"),

        ("Quỹ Đổi mới công nghệ quốc gia thực hiện việc cấp kinh phí như thế nào?",
         "dense_retrieval"),

        ("Điều 18 và Điều 35 của văn bản 12/VBHN-BXD quy định gì về hiệu lực phép bay?",
         "graph_traversal"),
    ]

    print(f"{'Query':<65} {'y_no':>4} {'fact':>4} {'xdoc':>4} {'dieu':>4} {'gkw':>4} {'lref':>4} | Expected")
    print("-" * 110)
    for query, expected in test_cases:
        f = extractor.extract(query)
        print(
            f"  {query[:63]:<63} "
            f"{int(f['is_yes_no_question']):>4} "
            f"{int(f['is_factoid_question']):>4} "
            f"{int(f['cross_doc_signal_count']):>4} "
            f"{int(f['dieu_reference_count']):>4} "
            f"{int(f['graph_keyword_count']):>4} "
            f"{int(f['legal_reference_count']):>4} "
            f"| {expected}"
        )
