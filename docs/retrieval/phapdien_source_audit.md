# Pháp Điển Source Audit

**Date:** 2026-06-23
**Source:** `C:\Users\Admin\Documents\Reasoning-Aware-Adaptive-Routing-for-Hybrid-GraphRAG\data\processed\phapdien_processed.jsonl`

## Summary

| Metric | Value |
|--------|------:|
| Total articles | 70,075 |
| Unique laws (extracted doc codes) | 4,619 |
| law_number present | 98.2% |
| article_number present | 100.0% |
| full article text (≥20 chars) | 100.0% |
| missing law_number | 1.8% |

## Law types (unique laws)

- **Thông tư:** 3,154
- **Nghị định:** 812
- **Quyết định:** 391
- **Luật:** 213
- **other:** 49

## Gate

✓ Source passes mandatory gate (law_number + full text available).

Full article text is in `content_markdown` (not truncated preview). Canonical `law_number` must be extracted from `source` / `ghi_chu` at rebuild time.