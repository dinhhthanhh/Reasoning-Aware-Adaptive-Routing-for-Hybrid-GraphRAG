import re

with open("docs/AI(PM)_ver 2.3.tex", "r", encoding="utf-8") as f:
    text = f.read()

# Update Table 4 (Performance)
text = re.sub(
    r"Pure Vector\s+&\s+\\textbf\{0\.836\}.*?\\\\",
    r"Pure Vector  & \\textbf{0.701} & 0.170 & \\textbf{0.588} & \\textbf{0.697} & 5,682 & 16,895 & -- & -- \\\\",
    text, flags=re.DOTALL
)
text = re.sub(
    r"Pure Graph\s+&\s+0\.632.*?\\\\",
    r"Pure Graph   & 0.632 & \\textbf{0.327} & 0.542 & 0.653 & 5,588 & 17,332 & -- & -- \\\\",
    text, flags=re.DOTALL
)
text = re.sub(
    r"Pure Hybrid\s+&\s+0\.803.*?\\\\",
    r"Pure Hybrid  & 0.656 & 0.098 & 0.003 & 0.007 & 6,815 & 20,174 & -- & -- \\\\",
    text, flags=re.DOTALL
)
text = re.sub(
    r"Single-stage Router\s+&\s+0\.614.*?\\\\",
    r"Single-stage Router & N/A & N/A & N/A & N/A & N/A & N/A & -- & -- \\\\",
    text, flags=re.DOTALL
)
text = re.sub(
    r"Two-stage Hybrid\s+&\s+0\.773.*?\\\\",
    r"Two-stage Hybrid & 0.661 & 0.267 & 0.410 & 0.473 & 9,950 & 23,686 & 78.3\\% & 5.3\\% \\\\",
    text, flags=re.DOTALL
)
text = re.sub(
    r"Oracle Router\s+&\s+0\.750.*?\\\\",
    r"Oracle Router & 0.606 & 0.127 & 0.282 & 0.360 & 5,975 & 19,330 & -- & -- \\\\",
    text, flags=re.DOTALL
)

# Update Table 5 (Cost Quality)
text = re.sub(
    r"Pure Vector & 0\.836 & 7,424 & 15,716 & 0\\% & 0\\% \\\\",
    r"Pure Vector & 0.701 & 5,682 & 16,895 & 0\\% & 0\\% \\\\",
    text
)
text = re.sub(
    r"Pure Graph & 0\.632 & 5,675 & 14,354 & 0\\% & 0\\% \\\\",
    r"Pure Graph & 0.632 & 5,588 & 17,332 & 0\\% & 0\\% \\\\",
    text
)
text = re.sub(
    r"Pure Hybrid & 0\.803 & 6,575 & 14,084 & 0\\% & 0\\% \\\\",
    r"Pure Hybrid & 0.656 & 6,815 & 20,174 & 0\\% & 0\\% \\\\",
    text
)
text = re.sub(
    r"Single-stage & 0\.614 & \\textbf\{3,959\} & \\textbf\{12,889\} & 0\\% & 1\\% \\\\",
    r"Single-stage & N/A & N/A & N/A & 0\\% & 1\\% \\\\",
    text
)
text = re.sub(
    r"Two-stage & \\textbf\{0\.773\} & 8,273 & 20,133 & 50\\% & 7\\% \\\\",
    r"Two-stage & \\textbf{0.661} & 9,950 & 23,686 & 78.3\\% & 5.3\\% \\\\",
    text
)

# Update texts
text = text.replace("0.836", "0.701")
text = text.replace("0.773", "0.661")
text = text.replace("0.750", "0.606")
text = text.replace("0.803", "0.656")

with open("docs/AI(PM)_ver 2.3.tex", "w", encoding="utf-8") as f:
    f.write(text)
print("Done")
