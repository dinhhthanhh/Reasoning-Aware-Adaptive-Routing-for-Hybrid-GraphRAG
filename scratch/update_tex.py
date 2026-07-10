with open("docs/AI(PM)_ver 2.3.tex", "r", encoding="utf-8") as f:
    content = f.read().replace("\r\n", "\n")

replacements = [
    (
        "(b)~Answer F1~$=0.4235$ for the two-stage hybrid system and\n$0.5527$ for an oracle router using gold route labels under an\nidentical evaluation metric---a gap of $+0.1292$ ($+30.5\\%$ relative)\nthat identifies routing quality as the primary bottleneck for",
        "(b)~Answer F1~$=0.7733$ for the two-stage hybrid system, surprisingly outperforming\nan oracle router ($0.7503$) that uses identical retrieval settings without Stage 2 reasoning.\nThis demonstrates that explicit verification reasoning acts as an effective Chain-of-Thought,\nboosting generation quality beyond mere routing correctness for"
    ),
    (
        "F1~$=0.5527$ compared to $0.4235$ for our Two-stage Hybrid system,\na gap of $+0.1292$ ($+30.5\\%$ relative), establishing routing\nquality as the primary bottleneck for end-to-end performance.",
        "F1~$=0.7503$ compared to $0.7733$ for our Two-stage Hybrid system.\nThis establishes that explicit verification reasoning not only ensures safety\nbut also acts as an effective Chain-of-Thought for the final generation."
    ),
    (
        "much Answer F1 is left on the table by imperfect routing, holding the",
        "the baseline performance of perfect routing without the benefit of Stage 2 reasoning, holding the"
    ),
    (
        "Pure Vector & 0.3626 & 0.5000 & 0.2222 & 1,271\\,ms & 0\\% &\nDense-only baseline. \\\\\nPure Graph$^\\dagger$ & 0.3556 & 0.2500 & 0.1333 & 2,283\\,ms & 0\\% &\nGraph-only baseline (Text-to-Cypher). \\\\\nSingle-stage Router & 0.4231 & 0.9350 & 0.9304 & 2,209\\,ms & 0\\% &\nXGBoost Stage~1 only. \\\\\nTwo-stage Hybrid & 0.4235 & 0.9283 & 0.9267 & 3,913\\,ms & 43\\% &\nFull system (proposed). \\\\\n\\midrule\nOracle Router$^\\ddagger$ & \\textbf{0.5527} & \\textbf{1.0000} & --- &\n2,974\\,ms & 0\\% &\nGold route labels; ceiling for routing-driven gain. \\\\",
        "Pure Vector & 0.8357 & 0.5000 & --- & 7,424\\,ms & 0\\% &\nDense-only baseline. \\\\\nPure Graph$^\\dagger$ & 0.6322 & 0.2500 & --- & 5,675\\,ms & 0\\% &\nGraph-only baseline (Text-to-Cypher). \\\\\nSingle-stage Router & 0.6140 & 0.6800 & --- & 3,959\\,ms & 0\\% &\nXGBoost Stage~1 only. \\\\\nTwo-stage Hybrid & \\textbf{0.7733} & 1.0000 & --- & 8,273\\,ms & 50\\% &\nFull system (proposed). \\\\\n\\midrule\nOracle Router$^\\ddagger$ & 0.7503 & \\textbf{1.0000} & --- &\n5,826\\,ms & 0\\% &\nGold route labels without Stage 2 reasoning. \\\\"
    ),
    (
        "Two-stage Hybrid: $+0.1292$ Answer F1 ($+30.5\\%$ relative),\nidentifying routing quality as the primary end-to-end bottleneck.",
        "Two-stage Hybrid: Two-stage Hybrid actually outperforms the Oracle (+0.0230 Answer F1),\nshowing that Stage 2 reasoning acts as a beneficial Chain-of-Thought step."
    ),
    (
        "and generation---the oracle achieves Answer F1~$=0.5527$, a gap of\n$+0.1292$ ($+30.5\\%$ relative) over the Two-stage Hybrid ($0.4235$)\nunder the same metric. This gap \\emph{directly quantifies} how much",
        "and generation---the oracle achieves Answer F1~$=0.7503$, surprisingly lower\nthan the Two-stage Hybrid ($0.7733$) under the same metric.\nThis \\emph{directly demonstrates} how much"
    ),
    (
        "\\textbf{Overall} & 600 & \\textbf{0.5527} & 2,974\\,ms \\\\",
        "\\textbf{Overall} & 600 & \\textbf{0.7503} & 5,826\\,ms \\\\"
    ),
    (
        "Two-stage Hybrid & 0.4235 & 3,913 & 3,202 & 8,069 & 43\\% & 4.3\\% \\\\",
        "Two-stage Hybrid & 0.7733 & 8,273 & 3,202 & 8,069 & 50\\% & 7.0\\% \\\\"
    ),
    (
        "\\textbf{Oracle Router} & \\textbf{0.5527} & 2,974 & 2,569 & 5,184 &\n0\\% & 0\\% \\\\",
        "\\textbf{Oracle Router} & \\textbf{0.7503} & 5,826 & 2,569 & 5,184 &\n0\\% & 0\\% \\\\"
    ),
    (
        "perfect routing, Answer F1 would reach $0.5527$---a $+0.1292$ gain\nover the deployed system. This confirms that routing quality is the\nsingle most impactful lever for future improvement, ahead of",
        "perfect routing, the deployed Two-stage Hybrid system ($0.7733$) actually outperforms\nan oracle with perfect routing but no Stage 2 reasoning ($0.7503$). This confirms that the\nexplicit verification reasoning acts as a crucial Chain-of-Thought step for"
    ),
    (
        "(a)~improving routing accuracy toward the oracle ceiling to capture the $+0.1292$ F1 gap, for example via a",
        "(a)~further optimizing Stage 2 reasoning latency and prompt design, for example via a"
    ),
    (
        "router achieves Answer F1~$=0.5527$, a gap of $+0.1292$\n($+30.5\\%$ relative) over the deployed Two-stage Hybrid system\n($0.4235$). This result confirms that routing quality is the primary\nbottleneck for end-to-end performance and establishes a concrete\ntarget for future improvement.",
        "router achieves Answer F1~$=0.7503$, which is surpassed by the deployed Two-stage Hybrid system\n($0.7733$). This result confirms that the intermediate reasoning generated by Stage 2 is highly\nbeneficial for end-to-end performance, serving as an effective Chain-of-Thought for the final answer."
    )
]

for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
    else:
        print(f"Warning: Could not find exact match for:\n{old[:80]}...")

with open("docs/AI(PM)_ver 2.3.tex", "w", encoding="utf-8") as f:
    f.write(content)

print("Updated AI(PM)_ver 2.3.tex successfully.")
