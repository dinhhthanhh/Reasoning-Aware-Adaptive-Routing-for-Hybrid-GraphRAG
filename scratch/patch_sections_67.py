import re

with open("docs/AI(PM)_ver 2.3.tex", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update Limitations (Section 6)
old_limitation = r"""benchmarks, rather than optimizing only one setting. Fifth,
strict-test routing accuracy decreases slightly from
$0.9350$ to $0.9283$ when Stage~2 is enabled. This happens
because the strict benchmark contains no intended clarify
queries, so some Stage~2 interventions are unnecessary in
that setting.

The Phase~3 conversation-aware extension introduces additional"""

new_limitation = r"""benchmarks, rather than optimizing only one setting. Fifth,
the Oracle baseline outperformance suggests that the pipeline
currently relies heavily on the expensive Stage~2 LLM not just
for routing verification, but as a crucial reasoning
(Chain-of-Thought) step. A pure single-stage router, even
if highly accurate, struggles to match this generation quality.
Future work should explore distilling this reasoning trace
directly into the final generator.

The Phase~3 conversation-aware extension introduces additional"""

if old_limitation in content:
    content = content.replace(old_limitation, new_limitation)
    print("Replaced Limitation 5 successfully!")
else:
    print("WARNING: Could not find exact match for Limitation 5.")

# 2. Update Conclusion (Section 7)
old_conclusion = r"""Two-stage Hybrid system obtains F1 $=0.7733$, outperforming
even the Oracle Router ($0.7503$). This confirms that Stage~2
verification acts as a critical Chain-of-Thought step, providing
vital reasoning context for
every query.

The ambiguity benchmarks show"""

new_conclusion = r"""Two-stage Hybrid system obtains F1 $=0.7733$, outperforming
even the Oracle Router ($0.7503$). This confirms that Stage~2
verification acts as a critical Chain-of-Thought step, providing
vital reasoning context for
complex and ambiguous queries.

The ambiguity benchmarks show"""

if old_conclusion in content:
    content = content.replace(old_conclusion, new_conclusion)
    print("Replaced Conclusion successfully!")
else:
    print("WARNING: Could not find exact match for Conclusion.")

with open("docs/AI(PM)_ver 2.3.tex", "w", encoding="utf-8") as f:
    f.write(content)

print("Patch script finished.")
