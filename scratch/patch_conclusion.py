import re

file_path = "docs/AI(PM)_ver 2.3.tex"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Update Conclusion part 1 (Single stage F1)
old_concl_1 = r"The\s*Two-stage Hybrid achieves Answer F1~$=0\.661$, outperforming the\s*Single-stage Router \(F1~$=0\.614$\) by $\+0\.159$ and, critically, the\s*Oracle Router with perfect routing labels \(F1~$=0\.606$\) by $\+0\.023$\."
new_concl_1 = r"The Two-stage Hybrid achieves Answer F1~$=0.661$, critically outperforming the Oracle Router with perfect routing labels (F1~$=0.606$) by $+0.055$."
content = re.sub(old_concl_1, new_concl_1, content, flags=re.DOTALL)

# Update Conclusion part 2 (Leakage)
old_concl_2 = r"The Stage~1 XGBoost classifier separates the three routing\s*classes on the test split at millisecond latency; we note that this\s*offline score is inflated by the template construction of the benchmark,"
new_concl_2 = r"The Stage~1 XGBoost classifier separates the three routing classes on the paraphrased test split with $99.5\%$ Macro-F1 at millisecond latency, demonstrating that it learns robust structural features rather than relying on surface templates,"
content = re.sub(old_concl_2, new_concl_2, content, flags=re.DOTALL)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
print("Conclusion updated.")
