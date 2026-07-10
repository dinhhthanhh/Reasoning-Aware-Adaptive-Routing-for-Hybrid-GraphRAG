import re

with open("docs/AI(PM)_ver 2.3.tex", "r", encoding="utf-8") as f:
    content = f.read().replace("\r\n", "\n")

with open("scratch/update_tex.py", "r", encoding="utf-8") as f:
    script_content = f.read()

# Execute the replacements logic manually here to be safe
# But it's easier to just patch update_tex.py
patched_script = script_content.replace(
    'content = f.read()',
    'content = f.read().replace("\\r\\n", "\\n")'
)

with open("scratch/update_tex.py", "w", encoding="utf-8") as f:
    f.write(patched_script)

print("Patched update_tex.py to handle CRLF")
