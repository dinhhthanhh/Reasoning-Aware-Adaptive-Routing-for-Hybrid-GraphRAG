# Cross-Reference Extraction Results (2026-06-23)

Run of `scripts/extract_cross_references.py` against the rebuilt graph.

## Result

| Metric | Value |
|---|---:|
| LegalArticle nodes scanned (with content) | 70,075 |
| Citation patterns applied | same-doc ("Điều N của Luật này", "căn cứ Điều N") |
| Candidate edges resolved | 0 |
| CROSS_REFERENCES edges inserted | 0 |

## Honest finding — why yield is ~0

The extraction is correct for **canonical** article numbering ("Điều 5",
"Điều 12"). However, the Pháp Điển corpus (the only corpus with `LegalArticle`
nodes) uses **structural codes**, not simple integers:

- `article_id`: `pd_007_003_0044`
- title: `8.4. Điều 8.4.LQ.8. Điều kiện kết hôn`

A citation such as "Điều 5 của Luật này" inside a Pháp Điển article refers to
article 5 of the *original* law, which in Pháp Điển is encoded as a structural
segment (e.g. `...LQ.5`), not as a node whose number is simply `5`. The integer
"5" therefore does not match any sibling article's number, so no edge is created.

This is the same root issue documented in
`docs/architecture/graph_known_limitations.md` (Limitation 6: Pháp Điển IDs not
resolved to canonical legal IDs) and `audit/critical_graph_risks.md` (Risk 7).

A further constraint: `LegalArticle` nodes store `content_preview` (truncated to
~1,200 chars), not full article text, so citations beyond the preview window are
not visible to the extractor.

## Decision (no fabrication)

Per the rebuild prompt constraint "DO NOT fabricate benchmark numbers" and the
4-hour cap on any single fix, no synthetic edges were inserted. The extraction
infrastructure (`scripts/extract_cross_references.py`) is in place and correct;
realising meaningful yield requires a Pháp-Điển-structural-code resolver that maps
"Điều N" citations to the corresponding `...<TYPE>.N` structural segment within
the same parent group. This is scoped as **future work**.

## Impact

CROSS_REFERENCES coverage remains effectively at the pre-extraction level. This
does not affect the primary thesis contribution (the two-stage router), and is
already disclosed as a known limitation. The honest framing in
`docs/defense_materials/graph_contribution_statement.md` stands unchanged.
