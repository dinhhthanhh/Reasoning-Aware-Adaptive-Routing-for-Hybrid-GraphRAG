import argparse
import json
import logging
import statistics
import sys
from collections import Counter
from pathlib import Path
import re

try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Try importing project modules for feature extraction and Stage 1 model
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

try:
    from router.features import FeatureExtractor
    from router.router_model import RouterModel
    HAS_PROJECT_MODULES = True
except ImportError:
    HAS_PROJECT_MODULES = False

# Regex for KeywordRuleRouter
LEGAL_RELATION_PATTERNS = re.compile(
    r"(sửa đổi|bổ sung|thay thế|bãi bỏ|hết hiệu lực|căn cứ|dẫn chiếu|quy định tại|theo Điều|liên quan đến văn bản)",
    re.IGNORECASE
)
CROSS_DOC_MULTI_HOP_PATTERNS = re.compile(
    r"(đồng thời|kết hợp|đối chiếu|giữa các văn bản|so sánh)",
    re.IGNORECASE
)

def keyword_rule_router(question: str) -> str:
    """Keyword-based routing baseline."""
    if CROSS_DOC_MULTI_HOP_PATTERNS.search(question):
        return "hybrid_reasoning"
    if LEGAL_RELATION_PATTERNS.search(question):
        return "graph_traversal"
    return "dense_retrieval"

def metadata_rule_router(item: dict) -> str:
    """Oracle-style metadata routing baseline."""
    question = item.get("question", "")
    if "is_cross_doc" not in item and "hop_count" not in item:
        return keyword_rule_router(question)
    
    is_cross_doc = item.get("is_cross_doc", False)
    hop_count = item.get("hop_count", 1)
    
    if is_cross_doc:
        return "hybrid_reasoning"
    if hop_count >= 2:
        return "graph_traversal"
    return "dense_retrieval"

def compute_metrics(y_true, y_pred, labels):
    """Compute standard classification metrics."""
    if not HAS_SKLEARN:
        return {"error": "scikit-learn is not installed."}
    
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    
    per_class = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    per_class_results = {}
    for i, label in enumerate(labels):
        per_class_results[label] = {
            "precision": float(per_class[0][i]),
            "recall": float(per_class[1][i]),
            "f1": float(per_class[2][i]),
            "support": int(per_class[3][i])
        }
        
    return {
        "accuracy": float(acc),
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "per_class": per_class_results,
        "confusion_matrix": cm.tolist()
    }

def generate_latex_table(baseline_results: dict) -> str:
    """Generate LaTeX table for overall baseline results."""
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Routing baseline results on the Vietnamese legal QA test set.}",
        "\\label{tab:baseline_results}",
        "\\small",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Method} & \\textbf{Accuracy} & \\textbf{Macro-F1} & \\textbf{Weighted-F1} \\\\",
        "\\midrule"
    ]
    
    for baseline_name, metrics in baseline_results.items():
        if "test" in metrics and "accuracy" in metrics["test"]:
            test_m = metrics["test"]
            lines.append(f"{baseline_name} & {test_m['accuracy']:.4f} & {test_m['macro_f1']:.4f} & {test_m['weighted_f1']:.4f} \\\\")
            
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    return "\n".join(lines)

def generate_distribution_figure(total_counts: dict) -> str:
    """Generate pgfplots LaTeX snippet for label distribution."""
    # Ensure correct order
    labels = ["dense_retrieval", "graph_traversal", "hybrid_reasoning"]
    counts = [total_counts.get(l, 0) for l in labels]
    
    labels_str = ','.join(labels).replace('_', '\\_')
    latex = f"""\\begin{{figure}}[!htbp]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    ybar,
    symbolic x coords={{{labels_str}}},
    xtick=data,
    nodes near coords,
    ymin=0,
    ylabel={{Number of samples}},
    xlabel={{Route Label}},
    width=0.8\\textwidth,
    height=6cm
]
\\addplot coordinates {{(dense\\_retrieval,{counts[0]}) (graph\\_traversal,{counts[1]}) (hybrid\\_reasoning,{counts[2]})}};
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{Route-label distribution of the main Vietnamese legal QA routing dataset. The distribution is skewed toward dense retrieval, which motivates reporting Macro-F1 together with Accuracy.}}
\\label{{fig:dataset_distribution}}
\\end{{figure}}"""
    return latex

def main():
    parser = argparse.ArgumentParser(description="Run Midterm Evaluation Utilities")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to dataset directory")
    parser.add_argument("--run-llm-baseline", action="store_true", help="Whether to run LLM-only baseline")
    args = parser.parse_args()
    
    # 1. Locate dataset
    data_dir = args.data_dir
    if not data_dir:
        legal_strict_path = ROOT_DIR / "qa_pipeline" / "data" / "legal_strict"
        final_path = ROOT_DIR / "qa_pipeline" / "data" / "final"
        
        if (legal_strict_path / "train.json").exists():
            data_dir = legal_strict_path
        elif (final_path / "train.json").exists():
            data_dir = final_path
        else:
            print("Error: Could not find dataset directory automatically.")
            return
            
    data_dir = Path(data_dir)
    print(f"Using dataset directory: {data_dir}")
    
    # 2. Load dataset
    splits = ["train", "dev", "test"]
    dataset = {}
    for split in splits:
        file_path = data_dir / f"{split}.json"
        if not file_path.exists():
            print(f"Error: Missing file {file_path}")
            return
        with open(file_path, "r", encoding="utf-8") as f:
            dataset[split] = json.load(f)
            
    # 3. Compute stats
    stats_splits = {}
    total_counter = Counter()
    
    for split in splits:
        data = dataset[split]
        labels = [item.get("routing_label", "unknown") for item in data]
        split_counter = Counter(labels)
        
        # Lengths
        q_chars = [len(item.get("question", "")) for item in data]
        q_tokens = [len(item.get("question", "").split()) for item in data]
        
        stats_splits[split] = {
            "total_samples": len(data),
            "label_counts": dict(split_counter),
            "mean_q_chars": statistics.mean(q_chars) if q_chars else 0,
            "mean_q_tokens": statistics.mean(q_tokens) if q_tokens else 0,
        }
        total_counter.update(split_counter)
        
    counts = list(total_counter.values())
    mean_count = statistics.mean(counts)
    pop_std = statistics.pstdev(counts)
    sample_std = statistics.stdev(counts) if len(counts) > 1 else 0.0
    
    # Global distributions
    hop_counter = Counter()
    cross_doc_counter = Counter()
    q_type_counter = Counter()
    aug_counter = Counter()
    
    for split in splits:
        for item in dataset[split]:
            if "hop_count" in item:
                hop_counter[item["hop_count"]] += 1
            if "is_cross_doc" in item:
                cross_doc_counter[item["is_cross_doc"]] += 1
            if "question_type" in item:
                q_type_counter[item["question_type"]] += 1
            if "augmented" in item:
                aug_counter[item["augmented"]] += 1
                
    dataset_stats = {
        "dataset_path": str(data_dir),
        "total_samples": sum(total_counter.values()),
        "label_counts": dict(total_counter),
        "mean_samples_per_class": round(mean_count, 2),
        "population_std_per_class": round(pop_std, 2),
        "sample_std_per_class": round(sample_std, 2),
        "splits": stats_splits,
        "hop_count_dist": dict(hop_counter),
        "is_cross_doc_dist": dict(cross_doc_counter),
        "question_type_dist": dict(q_type_counter),
        "augmented_dist": dict(aug_counter)
    }
    
    # Create output dir
    out_dir = ROOT_DIR / "reports" / "midterm"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. Save dataset stats
    with open(out_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(dataset_stats, f, indent=2, ensure_ascii=False)
        
    # Latex table for dataset split
    ds_latex = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Vietnamese legal QA routing dataset split by route label.}",
        "\\label{tab:dataset}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Label} & \\textbf{Train} & \\textbf{Dev} & \\textbf{Test} & \\textbf{Total} \\\\",
        "\\midrule"
    ]
    for label in ["dense_retrieval", "graph_traversal", "hybrid_reasoning"]:
        tr_c = stats_splits["train"]["label_counts"].get(label, 0)
        dv_c = stats_splits["dev"]["label_counts"].get(label, 0)
        te_c = stats_splits["test"]["label_counts"].get(label, 0)
        tot_c = total_counter.get(label, 0)
        escaped_label = label.replace('_', '\\_')
        ds_latex.append(f"{escaped_label} & {tr_c} & {dv_c} & {te_c} & {tot_c} \\\\")
        
    ds_latex.append("\\bottomrule")
    ds_latex.append("\\end{tabular}")
    ds_latex.append("\\end{table}")
    
    with open(out_dir / "dataset_split_table.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(ds_latex))
        
    # Figure snippet
    with open(out_dir / "dataset_distribution_figure.tex", "w", encoding="utf-8") as f:
        f.write(generate_distribution_figure(total_counter))
        
    # 5 & 6. Baselines
    majority_label = total_counter.most_common(1)[0][0]
    
    baseline_results = {}
    skipped_baselines = []
    
    labels_eval = ["dense_retrieval", "graph_traversal", "hybrid_reasoning"]
    
    # Init Stage1 if available
    stage1_router = None
    extractor = None
    if HAS_PROJECT_MODULES:
        try:
            stage1_router = RouterModel()
            extractor = FeatureExtractor()
            if not stage1_router.load():
                stage1_router = None # model loading failed
        except Exception as e:
            print(f"Warning: Could not initialize Stage1Model: {e}")
            stage1_router = None
            
    eval_splits = ["dev", "test"]
    
    for baseline in ["MajorityRoute", "KeywordRuleRouter", "MetadataRuleRouter", "Stage1Model", "LLMOnlyRouter"]:
        if baseline == "Stage1Model" and not stage1_router:
            skipped_baselines.append({"name": baseline, "reason": "Trained RouterModel not found or loaded."})
            continue
        if baseline == "LLMOnlyRouter" and not args.run_llm_baseline:
            skipped_baselines.append({"name": baseline, "reason": "--run-llm-baseline flag not provided."})
            continue
            
        print(f"Running baseline: {baseline}...")
        metrics_by_split = {}
        for split in eval_splits:
            data = dataset[split]
            y_true = []
            y_pred = []
            for item in data:
                gt = item.get("routing_label")
                if gt not in labels_eval:
                    continue
                y_true.append(gt)
                
                pred = "dense_retrieval"
                if baseline == "MajorityRoute":
                    pred = majority_label
                elif baseline == "KeywordRuleRouter":
                    pred = keyword_rule_router(item.get("question", ""))
                elif baseline == "MetadataRuleRouter":
                    pred = metadata_rule_router(item)
                elif baseline == "Stage1Model":
                    feat = extractor.extract(item.get("question", ""))
                    pred = stage1_router.predict(feat).route
                
                if pred not in labels_eval:
                    pred = "dense_retrieval" # fallback
                y_pred.append(pred)
                
            metrics_by_split[split] = compute_metrics(y_true, y_pred, labels_eval)
            
        baseline_results[baseline] = metrics_by_split

    # 7. Save baseline results
    with open(out_dir / "baseline_results.json", "w", encoding="utf-8") as f:
        json.dump({"results": baseline_results, "skipped": skipped_baselines}, f, indent=2, ensure_ascii=False)
        
    latex_table = generate_latex_table(baseline_results)
    with open(out_dir / "baseline_results_table.tex", "w", encoding="utf-8") as f:
        f.write(latex_table)
        
    # Generate CSV and per-class TeX manually
    csv_lines = ["Method,Split,Accuracy,Macro_F1,Weighted_F1"]
    for b_name, b_metrics in baseline_results.items():
        for split_name, mets in b_metrics.items():
            if "accuracy" in mets:
                csv_lines.append(f"{b_name},{split_name},{mets['accuracy']},{mets['macro_f1']},{mets['weighted_f1']}")
    with open(out_dir / "baseline_results.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))
        
    # 9. Markdown report
    md_content = f"""# Midterm Evaluation Summary

**Dataset Path Used**: `{data_dir}`
**Total Samples**: {dataset_stats['total_samples']}
**Splits**: Train={stats_splits['train']['total_samples']}, Dev={stats_splits['dev']['total_samples']}, Test={stats_splits['test']['total_samples']}

## Label Distribution (Overall)
"""
    for k, v in dataset_stats["label_counts"].items():
        md_content += f"- **{k}**: {v}\n"
        
    md_content += f"\n**Population Standard Deviation**: {pop_std:.2f}\n"
    md_content += f"**Mean Samples per Class**: {mean_count:.2f}\n"

    md_content += "\n## Baselines Run\n"
    for b in baseline_results.keys():
        acc = baseline_results[b]['test'].get('accuracy', 0)
        md_content += f"- **{b}**: Test Accuracy = {acc:.4f}\n"
        
    md_content += "\n## Baselines Skipped\n"
    for b in skipped_baselines:
        md_content += f"- **{b['name']}**: {b['reason']}\n"
        
    md_content += "\n## Important Warnings\n"
    md_content += "- **MetadataRuleRouter** is an oracle-style baseline as it uses `is_cross_doc` and `hop_count` from the ground truth annotations.\n"
    if not HAS_SKLEARN:
        md_content += "- **scikit-learn** is missing. Metrics could not be computed properly.\n"
        
    with open(out_dir / "midterm_eval_summary.md", "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print(f"Dataset path: {data_dir}")
    print(f"Total samples: {dataset_stats['total_samples']}")
    print(f"Label distribution: {dict(total_counter)}")
    print(f"Run baselines: {list(baseline_results.keys())}")
    print(f"Skipped baselines: {[b['name'] for b in skipped_baselines]}")
    print(f"Output directory: {out_dir}")
    print("="*50)


if __name__ == "__main__":
    main()
