# Chunk Audit

**Chroma chunks:** 199,847 (sample n=500)

| Metric | Value |
|--------|------:|
| Single-Điều chunks | 2% |
| Multi-Điều chunks | 65% |
| No Điều marker | 33% |
| Mean chars (sample) | 5825.9 |

**Indexing:** build_vectordb.py stores first 8000 chars of each source document as ONE chunk per doc (no Điều-aware splitting).

PD articles (typically 1 Điều each) are stored whole per record, but HF/long documents are truncated to 8000 chars — boundaries cut across Điều.

**Target:** 1 LegalArticle = 1 chunk (PD); Khoản splits only if >512 tokens.