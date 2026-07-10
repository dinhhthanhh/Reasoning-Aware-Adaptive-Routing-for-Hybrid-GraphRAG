# Test File Audit

| File | Questions | Unique Laws | Sample Gold IDs |
|---|---|---|---|
| `legal_strict/test.json` | 600 | 33 | `77/2026/NĐ-CP::Điều 18`<br>`12/VBHN-BXD::Điều 18`<br>`12/VBHN-BXD::Điều
35` |
| `legal_strict/test_benchmark_v2.json` | 600 | 33 | `77/2026/NĐ-CP::Điều 18`<br>`12/VBHN-BXD::Điều 18`<br>`12/VBHN-BXD::Điều
35` |
| `legal_strict/test_retry_77.json` | 77 | 25 | `12/2026/QĐ-UBND::Điều
33`<br>`49/VBHN-VPQH::Điều 53`<br>`12/2026/QĐ-UBND::Điều 10` |
| `legal_strict_clean/test.json` | 541 | 26 | `77/2026/NĐ-CP::Điều 18`<br>`12/VBHN-BXD::Điều 18`<br>`12/VBHN-BXD::Điều
35` |

## Decision rule from prompt:
- Keep the file already used in benchmark (600 questions, `legal_strict/test.json`)
- If `legal_strict_clean/test.json` has canonical IDs already, note this as potentially better.
- DO NOT delete any file yet — wait for user decision.