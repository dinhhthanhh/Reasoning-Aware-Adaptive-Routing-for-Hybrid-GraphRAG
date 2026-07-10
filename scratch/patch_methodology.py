import re

with open("docs/AI(PM)_ver 2.3.tex", "r", encoding="utf-8") as f:
    content = f.read()

# 3. Update Methodology (Section 3.3)
old_meth = r"""Stage~2 is an LLM-based verifier $f_2$ invoked only for
selected cases. Let $b=\phi_{\mathrm{amb}}(q,H)$ be the
ambiguity score."""

new_meth = r"""Stage~2 is an LLM-based verifier $f_2$ invoked only for
selected cases. This verifier acts as a Chain-of-Thought mechanism,
explicitly reasoning over the query's legal constraints before finalizing
the route, which helps resolve complex or ambiguous nuances that a
single-stage classifier might miss. Let $b=\phi_{\mathrm{amb}}(q,H)$ be the
ambiguity score."""

if old_meth in content:
    content = content.replace(old_meth, new_meth)
    print("Replaced Methodology successfully!")
else:
    print("WARNING: Could not find exact match for Methodology.")

with open("docs/AI(PM)_ver 2.3.tex", "w", encoding="utf-8") as f:
    f.write(content)

print("Patch script finished.")
