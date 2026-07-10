with open("docs/AI(PM)_ver 2.3.tex", "r", encoding="utf-8") as f:
    content = f.read()

# Replace bell character + "ddplot" with "\addplot"
content = content.replace("\x07ddplot", "\\addplot")

with open("docs/AI(PM)_ver 2.3.tex", "w", encoding="utf-8") as f:
    f.write(content)
print("Fixed bell character.")
