import json
import os
import sys
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def predict_keyword_rule(question: str) -> str:
    q = question.lower()
    if "sửa đổi" in q or "thay thế" in q or "bổ sung" in q or "hiệu lực" in q:
        return "graph_traversal"
    # Basic heuristic to capture hybrid
    if len(re.findall(r"(nghị định|luật|thông tư|bộ luật)", q)) > 1:
        return "hybrid_reasoning"
    return "dense_retrieval"

import re

def evaluate_rule_based(test_data):
    y_true = []
    y_pred = []
    for d in test_data:
        label = d.get("routing_label")
        if label not in ["dense_retrieval", "graph_traversal", "hybrid_reasoning"]:
            continue
        y_true.append(label)
        y_pred.append(predict_keyword_rule(d["question"]))
    
    from sklearn.metrics import accuracy_score
    return (
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro"),
        f1_score(y_true, y_pred, average="weighted")
    )

def evaluate_logreg(train_data, test_data):
    X_tr = [d["question"] for d in train_data if d.get("routing_label") in ["dense_retrieval", "graph_traversal", "hybrid_reasoning"]]
    y_tr = [d["routing_label"] for d in train_data if d.get("routing_label") in ["dense_retrieval", "graph_traversal", "hybrid_reasoning"]]
    
    X_te = [d["question"] for d in test_data if d.get("routing_label") in ["dense_retrieval", "graph_traversal", "hybrid_reasoning"]]
    y_te = [d["routing_label"] for d in test_data if d.get("routing_label") in ["dense_retrieval", "graph_traversal", "hybrid_reasoning"]]
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X_tr_vec = vectorizer.fit_transform(X_tr)
    X_te_vec = vectorizer.transform(X_te)
    
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_tr_vec, y_tr)
    y_pred = clf.predict(X_te_vec)
    from sklearn.metrics import accuracy_score
    logreg_acc = accuracy_score(y_te, y_pred)
    logreg_f1 = f1_score(y_te, y_pred, average="macro")
    logreg_wf1 = f1_score(y_te, y_pred, average="weighted")
    
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(X_tr_vec, y_tr)
    rf_pred = rf.predict(X_te_vec)
    rf_acc = accuracy_score(y_te, rf_pred)
    rf_f1 = f1_score(y_te, rf_pred, average="macro")
    rf_wf1 = f1_score(y_te, rf_pred, average="weighted")
    
    return (logreg_acc, logreg_f1, logreg_wf1), (rf_acc, rf_f1, rf_wf1)

if __name__ == "__main__":
    base_dir = "qa_pipeline/data/phapdien_strict"
    train_data = load_data(f"{base_dir}/train.json")
    test_data = load_data(f"{base_dir}/test.json")
    
    rule_res = evaluate_rule_based(test_data)
    logreg_res, rf_res = evaluate_logreg(train_data, test_data)
    
    print(f"KeywordRuleRouter: Acc={rule_res[0]:.3f}, Mac={rule_res[1]:.3f}, Wt={rule_res[2]:.3f}")
    print(f"LogisticRegression: Acc={logreg_res[0]:.3f}, Mac={logreg_res[1]:.3f}, Wt={logreg_res[2]:.3f}")
    print(f"RandomForest: Acc={rf_res[0]:.3f}, Mac={rf_res[1]:.3f}, Wt={rf_res[2]:.3f}")
