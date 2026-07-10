# Retrieval ID Audit

**Source:** `C:\Users\Admin\Documents\Reasoning-Aware-Adaptive-Routing-for-Hybrid-GraphRAG\results_final_unified\e2e_benchmark\predictions.json`
**Unresolvable ID rate:** 73.6%

## Category breakdown

| Category | Fraction |
|----------|---------:|
| phapdien_structural_title | 39.2% |
| canonical_resolvable | 26.4% |
| hf_document_id | 19.4% |
| unresolvable_other | 15.0% |

## Root cause

61.8% unresolvable is primarily a **corpus architecture** problem: retrieved IDs are Pháp Điển structural titles (`19.2. Điều 19.2.TT.10.15...`), HF opaque IDs (`Document 4266`), or partial codes — not canonical `law_number::Điều_N` keys used by the gold test set.