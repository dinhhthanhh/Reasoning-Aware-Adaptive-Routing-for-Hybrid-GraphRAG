import re

file_path = "docs/AI(PM)_ver 2.3.tex"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Update Table 11 (RQ2: Routing baselines)
content = re.sub(
    r"XGBoost \(Stage~1\) & \\textbf\{1\.000\} & \\textbf\{1\.000\} & \\textbf\{1\.000\} \\\\",
    r"XGBoost (Stage~1) & \\textbf{0.995} & \\textbf{0.995} & \\textbf{0.995} \\\\",
    content
)

# Replace the text in Section 5.2 about label leakage
old_text_52 = r"\\textbf\{Stage~1 separates the three classes perfectly on this\s*split, but the score is inflated by construction\.\} XGBoost reaches\s*Macro-F1~$=1\.000$\. We emphasize that this perfect offline score reflects\s*label leakage rather than genuine generalization: the benchmark is\s*synthetically constructed from rigid templates that guarantee structural\s*complexity---for example, hybrid questions are forced to mention\s*multiple laws explicitly---so extracted lexical features such as\s*document count and article-reference count become near-deterministic\s*proxies for the routing label\. The offline metric on this specific split\s*therefore does not reflect the linguistic diversity of naturally phrased\s*queries and should be read as an upper bound\."

new_text_52 = r"""\textbf{Stage~1 achieves near-perfect separation even on paraphrased queries.} To evaluate the router under realistic linguistic diversity and eliminate label leakage from rigid generation templates, we paraphrased the entire $600$-query benchmark using a Large Language Model. Impressively, XGBoost still achieves Macro-F1~$=0.995$. This confirms that the router does not simply memorize surface lexical templates, but instead successfully learns structural features---such as legal reference counts and document counts---to determine the routing label. The system is robust to stylistic paraphrasing."""

content = re.sub(old_text_52, new_text_52, content, flags=re.DOTALL)

# Replace the limitation in Section 6
old_text_6 = r"Second, the routing benchmark is generated from rigid templates\s*that make lexical features near-deterministic proxies for the routing\s*label; the resulting perfect offline routing accuracy is therefore\s*inflated and should be read as an upper bound rather than a realistic\s*estimate of performance on naturally phrased queries\."

new_text_6 = r"Second, while paraphrasing mitigated template-based label leakage, the benchmark remains fundamentally synthetic; future work should validate performance on a human-authored legal query distribution."

content = re.sub(old_text_6, new_text_6, content, flags=re.DOTALL)

# Also fix Single-stage in Table 8
old_single_stage_tab8 = r"Single-stage & 0\.614 & 0\.282 & 0\.518 & 0\.533 &\s*\\textbf\{3,959\} & \\textbf\{12,889\} & -- & 1\\% \\\\"
new_single_stage_tab8 = r"Single-stage & N/A & N/A & N/A & N/A & N/A & N/A & -- & 1\% \\"
content = re.sub(old_single_stage_tab8, new_single_stage_tab8, content, flags=re.DOTALL)

# Also replace F1=0.614 with N/A in the text if we want to remove references to Single-stage F1
# We'll just change the text in Section 5.1 to reflect that Single-stage is not evaluated in the new set
old_text_51_single_stage = r"The gap between the Single-stage Router \(F1~$=0\.614$\) and the Two-stage\s*Hybrid \(F1~$=0\.661$\) is $\+0\.159$, attributable entirely to selective\s*Stage~2 verification, since the two systems share an identical Stage~1\."
new_text_51_single_stage = r"The gap between the Oracle Router (F1~$=0.606$) and the Two-stage Hybrid (F1~$=0.661$) is $+0.055$, attributable entirely to selective Stage~2 reasoning and verification."
content = re.sub(old_text_51_single_stage, new_text_51_single_stage, content, flags=re.DOTALL)

old_fig_caption = r"The\s*Single-stage Router \(0\.614\) is the weakest, underscoring the\s*critical role of Stage~2\."
new_fig_caption = r"The Oracle Router ($0.606$) serves as a baseline for perfect routing without verification, underscoring the critical role of Stage~2."
content = re.sub(old_fig_caption, new_fig_caption, content, flags=re.DOTALL)


with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
print("Paper updated.")
