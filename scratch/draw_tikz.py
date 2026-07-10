import re

tikz_code = r"""
\vspace{1em}
\begin{minipage}{\textwidth}
\centering
\textbf{(a) Pipeline Định tuyến (Routing Pipeline)} \\[1em]
\resizebox{0.95\textwidth}{!}{
\begin{tikzpicture}[
    >=latex,
    node distance=1.5cm and 2.5cm,
    font=\small\sffamily,
    % Styles
    data/.style={draw=blue!60, fill=blue!5, rectangle, rounded corners, thick, text width=2.2cm, align=center, minimum height=1.2cm},
    process/.style={draw=orange!80, fill=orange!10, rectangle, thick, text width=2.8cm, align=center, minimum height=1.2cm, blur shadow={shadow blur steps=5}},
    decision/.style={draw=teal!80, fill=teal!10, diamond, thick, text width=2cm, align=center, aspect=1.5, blur shadow={shadow blur steps=5}},
    llm/.style={draw=purple!80, fill=purple!10, rectangle, thick, rounded corners, text width=2.8cm, align=center, minimum height=1.2cm, blur shadow={shadow blur steps=5}},
    output/.style={draw=red!80, fill=red!10, rectangle, thick, rounded corners, text width=2.5cm, align=center, minimum height=1.2cm},
    line/.style={draw, thick, -latex}
]

    % Nodes
    \node [data] (input) {Input Query $q_t$ \\ History $H$};
    \node [process, right=of input] (extract) {Feature Extraction \\ \scriptsize (27 features)};
    \node [process, right=of extract] (stage1) {Stage 1 Routing \\ \scriptsize (XGBoost)};
    \node [decision, right=of stage1] (trigger) {Trigger \\ $\Gamma(q_t) \ge \tau$?};
    \node [llm, below=of trigger] (stage2) {Stage 2 Verification \\ \scriptsize (LLM CoT)};
    \node [output, right=of trigger] (route) {Final Route \\ $d^*$};
    
    \node [process, right=of route, text width=3cm] (exec) {Retrieval \& \\ Generation};

    % Edges
    \path [line] (input) -- (extract);
    \path [line] (extract) -- (stage1);
    \path [line] (stage1) -- (trigger);
    
    \path [line] (trigger) -- node[above] {No} (route);
    \path [line] (trigger) -- node[left] {Yes} (stage2);
    \path [line] (stage2) -| node[near start, right] {Verified Route} (route);
    
    \path [line] (route) -- (exec);
    
    % Background box for Stage 1 & 2
    \begin{pgfonlayer}{background}
        \node [fill=gray!5, rounded corners, draw=gray!20, dashed, fit=(extract) (stage1) (trigger) (stage2), inner sep=0.4cm] (router_box) {};
        \node [above right, text=gray!80] at (router_box.south west) {\textbf{Two-stage Adaptive Router}};
    \end{pgfonlayer}

\end{tikzpicture}
}
\end{minipage}

\vspace{2em}

\begin{minipage}{\textwidth}
\centering
\textbf{(b) Bốn lộ trình định tuyến (Routing Outcomes)} \\[1em]
\renewcommand{\arraystretch}{1.5}
\begin{tabularx}{0.95\textwidth}{>{\hsize=0.2\hsize\bfseries}X >{\hsize=0.8\hsize}X}
\toprule
Route & Description \& Example \\
\midrule
Dense ($d_v$) & \textbf{Direct lookup:} Single-hop queries easily resolved by semantic similarity. \\
& \textit{Ex:} ``Mức phạt đối với hành vi vượt đèn đỏ là bao nhiêu?'' \\
Graph ($d_g$) & \textbf{Relational traversal:} Queries requiring explicit legal relations (amends, guides). \\
& \textit{Ex:} ``Văn bản nào hướng dẫn thi hành Điều 15 Luật Đầu tư?'' \\
Hybrid ($d_h$) & \textbf{Cross-document synthesis:} Complex multi-hop reasoning across multiple documents. \\
& \textit{Ex:} ``Sự khác biệt về thủ tục hải quan giữa Luật Hải quan và Hiệp định EVFTA?'' \\
Clarify ($d_c$) & \textbf{Ambiguity resolution:} Underspecified queries requiring user clarification before retrieval. \\
& \textit{Ex:} ``Luật Đất đai quy định thế nào?'' (Too vague, requires specifying the exact topic). \\
\bottomrule
\end{tabularx}
\end{minipage}
"""

with open("docs/AI(PM)_ver 2.3.tex", "r", encoding="utf-8") as f:
    content = f.read()

# Replace the includegraphics
target = r"\includegraphics[width=\textwidth]{figs/system_pipeline.png}"
if target in content:
    content = content.replace(target, tikz_code)
    with open("docs/AI(PM)_ver 2.3.tex", "w", encoding="utf-8") as f:
        f.write(content)
    print("Replaced successfully!")
else:
    print("Target not found!")
