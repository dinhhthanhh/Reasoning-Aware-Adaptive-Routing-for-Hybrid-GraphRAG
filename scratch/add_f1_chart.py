import re

with open("docs/AI(PM)_ver 2.3.tex", "r", encoding="utf-8") as f:
    content = f.read()

target = r"""\end{tabular}%
}
\end{table}"""

replacement = r"""\end{tabular}%
}
\end{table}

\begin{figure}[!htbp]
\centering
\begin{tikzpicture}
\begin{axis}[
    ybar,
    width=0.95\columnwidth,
    height=5.2cm,
    ymin=0.5, ymax=0.9,
    ylabel={Answer F1},
    symbolic x coords={Vector, Graph, Hybrid, Adaptive, Oracle},
    xtick=data,
    nodes near coords,
    nodes near coords style={font=\scriptsize},
    tick label style={font=\scriptsize},
    label style={font=\scriptsize},
    enlarge x limits=0.15
]
\addplot[fill=teal!30, draw=teal!70, thick] coordinates {
    (Vector,0.8357)
    (Graph,0.6322)
    (Hybrid,0.8031)
    (Adaptive,0.7733)
    (Oracle,0.7503)
};
\end{axis}
\end{tikzpicture}
\caption{End-to-end Answer F1. Pure Vector provides a unexpectedly strong baseline. The Two-stage Adaptive router (0.77) outperforms the Oracle router (0.75) by leveraging explicit Stage 2 verification as a Chain-of-Thought mechanism, demonstrating that explicit reasoning context is more critical than perfect routing labels alone.}
\label{fig:f1_bar}
\end{figure}"""

if target in content:
    content = content.replace(target, replacement, 1)
    print("Added F1 Bar Chart successfully!")
else:
    print("WARNING: Could not find target to insert F1 bar chart.")

with open("docs/AI(PM)_ver 2.3.tex", "w", encoding="utf-8") as f:
    f.write(content)

print("Patch script finished.")
