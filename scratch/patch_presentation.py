import re

file_path = 'docs/presentation.tex'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix baseline Macro F1
old_coords = r'symbolic x coords=\{BM25, SVM, PhoBERT, XGBoost\},'
new_coords = r'symbolic x coords={Rule-based, LogReg, PhoBERT, XGBoost},'
content = re.sub(old_coords, new_coords, content)

old_plot = r'\\addplot\[fill=blue!50\] coordinates \{\(BM25,0\.497\) \(SVM,0\.763\) \(PhoBERT,0\.901\) \(XGBoost,0\.995\)\};'
new_plot = r'\\addplot[fill=blue!50] coordinates {(Rule-based,0.497) (LogReg,0.781) (PhoBERT,0.901) (XGBoost,0.995)};'
content = re.sub(old_plot, new_plot, content)

# Improve Architecture TikZ
old_tikz = r'\\begin\{tikzpicture\}\[\s*node distance=1\.5cm and 2\.5cm,[\s\S]*?\\end\{tikzpicture\}'

new_tikz = r'''\begin{tikzpicture}[
        node distance=1.5cm and 2.5cm,
        box/.style={draw=blue!80!black, thick, rounded corners=3pt, minimum width=2.8cm, minimum height=1.2cm, align=center, fill=blue!5, font=\small\bfseries, drop shadow={opacity=0.2, shadow xshift=1pt, shadow yshift=-1pt}},
        db/.style={draw=orange!80!black, thick, cylinder, shape border rotate=90, aspect=0.25, minimum height=1.6cm, minimum width=2.2cm, align=center, fill=orange!10, font=\small\bfseries, drop shadow={opacity=0.2, shadow xshift=1pt, shadow yshift=-1pt}},
        arrow/.style={-Latex, thick, draw=blue!70!black},
        highlight/.style={draw=red!80!black, fill=red!5, thick}
    ]
    % Nodes
    \node[box, fill=green!10, draw=green!60!black] (query) {User Query\\$(q_t, H)$};
    \node[box, fill=yellow!15, draw=yellow!60!black, right=of query] (router) {Two-stage\\Adaptive Router};
    \node[box, fill=purple!10, draw=purple!60!black, above right=of router, xshift=1cm, yshift=-0.5cm] (clarify) {Clarification\\Generator};
    
    \node[db, right=of router, xshift=1cm, yshift=-1.5cm] (graph) {Neo4j\\Graph DB};
    \node[db, below=of graph, yshift=0.5cm] (vector) {Chroma\\Vector DB};
    
    \node[box, highlight, right=of graph, xshift=1.5cm, yshift=-1cm] (llm) {LLM Verifier\\\& Generator};
    \node[box, fill=green!10, draw=green!60!black, right=of llm] (answer) {Final Answer};

    % Background Boxes
    \begin{scope}[on background layer]
        \node[draw=gray!50, dashed, thick, rounded corners, fill=gray!5, fit=(graph) (vector), inner sep=15pt] (kbbg) {};
        \node[above=0.1cm of kbbg.north, font=\small\bfseries\color{gray!80!black}] {Dual Knowledge Base};
    \end{scope}
    
    % Edges
    \draw[arrow] (query) -- (router);
    \draw[arrow] (router) |- node[pos=0.7, above, font=\scriptsize] {Clarify} (clarify);
    
    \draw[arrow] (router) -| node[pos=0.2, above, font=\scriptsize] {Graph / Hybrid} (graph);
    \draw[arrow] (router) |- node[pos=0.7, above, font=\scriptsize] {Vector / Hybrid} (vector);
    
    \draw[arrow] (graph) -| node[pos=0.2, above, font=\scriptsize] {Cypher context} (llm);
    \draw[arrow] (vector) -| node[pos=0.2, below, font=\scriptsize] {Text context} (llm);
    \draw[arrow] (llm) -- (answer);
    
    \draw[arrow, dashed] (clarify) -| node[pos=0.2, above, font=\scriptsize] {Ask Back} (answer);
    \end{tikzpicture}'''

if 'shadows' not in content:
    content = content.replace('backgrounds}', 'backgrounds, shadows}')

content = re.sub(old_tikz, new_tikz, content)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print('Patched successfully')
