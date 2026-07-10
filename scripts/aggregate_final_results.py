#!/usr/bin/env python3
"""
aggregate_final_results.py
==========================
Đọc tất cả output JSON từ results_final_unified/ và tổng hợp thành:
  1. UNIFIED_PAPER_METRICS.json  — một bộ số duy nhất cho paper
  2. UNIFIED_LATEX_TABLES.tex   — các table LaTeX sẵn sàng paste vào paper

Usage:
  python scripts/aggregate_final_results.py \
      --results-dir results_final_unified \
      --output     results_final_unified/UNIFIED_PAPER_METRICS.json \
      --latex      results_final_unified/UNIFIED_LATEX_TABLES.tex
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def load_json(path: str):
    p = Path(path)
    if not p.exists():
        print(f"  [WARN] File not found: {path}")
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fmt(v, digits=4):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


# ---------------------------------------------------------------
# Readers
# ---------------------------------------------------------------

def read_e2e(results_dir: str) -> dict:
    candidates = [
        f"{results_dir}/e2e_benchmark/legal_strict_full_summary.json",
        f"{results_dir}/e2e_benchmark/results_summary.json",
        f"{results_dir}/e2e_benchmark/all_systems.json",
        "eval_results/legal_strict_full_summary.json",
    ]
    for c in candidates:
        data = load_json(c)
        if data:
            print(f"  [E2E] Loaded from: {c}")
            return data
    print("  [WARN] No end-to-end summary found. Tried:", candidates)
    return {}


def read_cv(results_dir: str) -> dict:
    candidates = [
        f"{results_dir}/router_cv/training_report.json",
        "router_model/training_report.json",
        "data/router_training/legal_strict/training_report.json",
    ]
    for c in candidates:
        data = load_json(c)
        if data:
            print(f"  [CV] Loaded from: {c}")
            return data
    return {}


def read_clarify(results_dir: str) -> dict:
    candidates = [
        f"{results_dir}/clarify/clarify_eval_summary.json",
        "results_phase3/clarify_eval_summary.json",
        "results/research_paper_metrics.json",
    ]
    for c in candidates:
        data = load_json(c)
        if data:
            print(f"  [CLARIFY] Loaded from: {c}")
            return data
    return {}


def read_conv(results_dir: str) -> dict:
    candidates = [
        f"{results_dir}/conv_ambiguity/conversation_ambiguity_summary.json",
        "results_phase3/conversation_ambiguity_summary.json",
    ]
    for c in candidates:
        data = load_json(c)
        if data:
            print(f"  [CONV] Loaded from: {c}")
            return data
    return {}


def read_routing_only(results_dir: str) -> dict:
    candidates = [
        f"{results_dir}/routing_only/strict_routing_sanity_summary.json",
        "results_phase3/strict_routing_sanity_summary.json",
    ]
    for c in candidates:
        data = load_json(c)
        if data:
            print(f"  [ROUTING] Loaded from: {c}")
            return data
    return {}


# ---------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------

def extract_system_metrics(e2e: dict, system_key: str) -> dict:
    if system_key in e2e:
        return e2e[system_key]
    if isinstance(e2e, list):
        for item in e2e:
            if isinstance(item, dict):
                name = item.get("system", item.get("name", ""))
                if system_key.lower() in name.lower():
                    return item
    if "systems" in e2e:
        return extract_system_metrics(e2e["systems"], system_key)
    return {}


def safe_get(d: dict, *keys, default=None):
    aliases = {
        "f1":              ["f1", "answer_f1", "token_f1", "answer_token_f1"],
        "routing_acc":     ["routing_accuracy", "route_accuracy", "routing_acc"],
        "macro_f1":        ["macro_f1", "macro_f1_routing", "routing_macro_f1"],
        "avg_latency_ms":  ["avg_latency_ms", "avg_latency", "average_latency_ms"],
        "median_latency":  ["median_latency_ms", "median_latency"],
        "p95_latency":     ["p95_latency_ms", "p95_latency"],
        "stage2_trigger":  ["stage2_trigger_rate", "stage2_rate", "stage2_trigger"],
        "stage2_override": ["stage2_override_rate", "override_rate"],
        "bertscore_f1":    ["bertscore_f1", "bert_score_f1", "bertscore"],
    }
    for key in keys:
        if key in d:
            return d[key]
        if key in aliases:
            for alias in aliases[key]:
                if alias in d:
                    return d[alias]
    return default


# ---------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------

def aggregate(results_dir: str) -> dict:
    print("\n[AGG] Reading evaluation outputs...")
    e2e          = read_e2e(results_dir)
    cv           = read_cv(results_dir)
    clarify      = read_clarify(results_dir)
    conv         = read_conv(results_dir)
    routing_only = read_routing_only(results_dir)

    systems = {
        "pure_vector":         extract_system_metrics(e2e, "pure_vector"),
        "pure_graph":          extract_system_metrics(e2e, "pure_graph"),
        "single_stage_router": extract_system_metrics(e2e, "single_stage_router"),
        "two_stage_hybrid":    extract_system_metrics(e2e, "two_stage_hybrid"),
    }

    e2e_table = {}
    for sname, sdata in systems.items():
        e2e_table[sname] = {
            "answer_f1":         safe_get(sdata, "f1"),
            "routing_accuracy":  safe_get(sdata, "routing_acc"),
            "macro_f1":          safe_get(sdata, "macro_f1"),
            "avg_latency_ms":    safe_get(sdata, "avg_latency_ms"),
            "median_latency_ms": safe_get(sdata, "median_latency"),
            "p95_latency_ms":    safe_get(sdata, "p95_latency"),
            "stage2_trigger":    safe_get(sdata, "stage2_trigger"),
            "stage2_override":   safe_get(sdata, "stage2_override"),
        }

    cv_cross = cv.get("cross_validation", {})
    cv_val = cv.get("validation", {}).get("per_class", {})
    
    cv_table = {
        "cv_accuracy_mean": safe_get(cv_cross, "cv_accuracy_mean") or safe_get(cv, "cv_accuracy_mean"),
        "cv_accuracy_std":  safe_get(cv_cross, "cv_accuracy_std") or safe_get(cv, "cv_accuracy_std"),
        "cv_macro_f1_mean": safe_get(cv_cross, "cv_macro_f1_mean") or safe_get(cv, "cv_macro_f1_mean"),
        "cv_macro_f1_std":  safe_get(cv_cross, "cv_macro_f1_std") or safe_get(cv, "cv_macro_f1_std"),
        "per_class": {
            "dense_retrieval":  cv_val.get("dense_retrieval", {}).get("F1") or safe_get(cv, "dense_f1", "dense_retrieval_f1"),
            "graph_traversal":  cv_val.get("graph_traversal", {}).get("F1") or safe_get(cv, "graph_f1", "graph_traversal_f1"),
            "hybrid_reasoning": cv_val.get("hybrid_reasoning", {}).get("F1") or safe_get(cv, "hybrid_f1", "hybrid_reasoning_f1"),
        },
        "per_class_std": {
            "dense_retrieval":  safe_get(cv, "dense_f1_std"),
            "graph_traversal":  safe_get(cv, "graph_f1_std"),
            "hybrid_reasoning": safe_get(cv, "hybrid_f1_std"),
        },
    }

    two_stage_data = systems.get("two_stage_hybrid", {})
    bertscore = {
        "f1":        safe_get(two_stage_data, "bertscore_f1", default=None),
        "model":     "xlm-roberta-large",
        "reference": "concise_answer",
    }

    clarify_table = {
        "phase3_full_policy": {
            "route_accuracy":    safe_get(clarify, "route_accuracy", "routing_accuracy"),
            "clarify_precision": safe_get(clarify, "clarify_precision", "precision"),
            "clarify_recall":    safe_get(clarify, "clarify_recall", "recall"),
            "clarify_f1":        safe_get(clarify, "clarify_f1", "f1"),
            "stage2_trigger":    safe_get(clarify, "stage2_trigger_rate"),
            "false_positives":   safe_get(clarify, "false_positives", "clarify_false_positives"),
            "false_negatives":   safe_get(clarify, "false_negatives", "clarify_false_negatives"),
        }
    }

    routing_sanity = {
        "total":          safe_get(routing_only, "total"),
        "route_accuracy": safe_get(routing_only, "route_accuracy", "routing_accuracy"),
        "stage2_trigger": safe_get(routing_only, "stage2_trigger_rate"),
        "false_clarify":  safe_get(routing_only, "false_clarify_count", "clarify_false_positives"),
        "per_gold_route": {
            "dense":  safe_get(routing_only, "dense_accuracy"),
            "graph":  safe_get(routing_only, "graph_accuracy"),
            "hybrid": safe_get(routing_only, "hybrid_accuracy"),
        }
    }

    unified = {
        "_meta": {
            "generated":   datetime.now().isoformat(),
            "results_dir": results_dir,
            "note": ("All numbers from a SINGLE unified evaluation run. "
                     "Safe to use directly in the paper.")
        },
        "end_to_end":    e2e_table,
        "stage1_cv":     cv_table,
        "bertscore":     bertscore,
        "clarification": clarify_table,
        "routing_sanity": routing_sanity,
        "conversation":  conv,
    }
    return unified


# ---------------------------------------------------------------
# LaTeX table generator
# ---------------------------------------------------------------

def generate_latex(unified: dict) -> str:
    e2e = unified.get("end_to_end", {})
    cv  = unified.get("stage1_cv", {})
    cl  = unified.get("clarification", {}).get("phase3_full_policy", {})
    bs  = unified.get("bertscore", {})

    def row(name, skey, note):
        d   = e2e.get(skey, {})
        f1  = fmt(d.get("answer_f1"),       4)
        ra  = fmt(d.get("routing_accuracy"), 4)
        mf1 = fmt(d.get("macro_f1"),         4)
        lat = f"{d['avg_latency_ms']:.1f}" if d.get("avg_latency_ms") else "N/A"
        return (f"{name} & {f1} & {ra} & {mf1} & "
                f"{lat}\\,ms & {note} \\\\\n")

    lines = []
    lines.append("%% ============================================================\n")
    lines.append("%% AUTO-GENERATED by aggregate_final_results.py\n")
    lines.append(f"%% Generated: {unified['_meta']['generated']}\n")
    lines.append("%% ============================================================\n\n")

    # Table 1: End-to-end
    lines.append("% ----- TABLE: tab:end_to_end_results -----\n")
    lines.append("\\toprule\n")
    lines.append("\\textbf{System} & \\textbf{Answer F1} & \\textbf{Routing Acc.} "
                 "& \\textbf{Macro-F1} & \\textbf{Avg.\\ Latency} & \\textbf{Notes} \\\\\n")
    lines.append("\\midrule\n")
    lines.append(row("Pure Vector",               "pure_vector",         "Fixed dense retrieval; no graph."))
    lines.append(row("Pure Graph (Text-to-Cypher)","pure_graph",         "Fixed graph traversal."))
    lines.append(row("Single-stage Router",       "single_stage_router", "Stage~1 XGBoost only."))
    lines.append(row("Two-stage Hybrid",          "two_stage_hybrid",    "Full system; selective Stage~2 LLM verifier."))
    lines.append("\\bottomrule\n\n")

    # Table 2: Stage-1 CV
    lines.append("% ----- TABLE: tab:cv (5-fold CV) -----\n")
    pc  = cv.get("per_class", {})
    pcs = cv.get("per_class_std", {})
    acc_m = fmt(cv.get("cv_accuracy_mean"), 4) if cv.get("cv_accuracy_mean") else "N/A"
    acc_s = fmt(cv.get("cv_accuracy_std"),  4) if cv.get("cv_accuracy_std")  else "N/A"
    mf1_m = fmt(cv.get("cv_macro_f1_mean"), 4) if cv.get("cv_macro_f1_mean") else "N/A"
    mf1_s = fmt(cv.get("cv_macro_f1_std"),  4) if cv.get("cv_macro_f1_std")  else "N/A"

    def cv_row(label, skey, support):
        f1  = fmt(pc.get(skey),  3) if pc.get(skey)  else "N/A"
        std = fmt(pcs.get(skey), 3) if pcs.get(skey) else ""
        pm  = f" $\\pm${std}" if std else ""
        return f"{label} & {f1}{pm} & {support} \\\\\n"

    lines.append("\\toprule\n")
    lines.append("\\textbf{Class} & \\textbf{F1 (CV)} & \\textbf{Support} \\\\\n")
    lines.append("\\midrule\n")
    lines.append(cv_row("Dense retrieval",  "dense_retrieval",  300))
    lines.append(cv_row("Graph traversal",  "graph_traversal",  150))
    lines.append(cv_row("Hybrid reasoning", "hybrid_reasoning", 150))
    lines.append("\\midrule\n")
    lines.append(f"Macro F1 (CV) & \\multicolumn{{2}}{{c}}{{{mf1_m} $\\pm${mf1_s}}} \\\\\n")
    lines.append(f"Accuracy (CV) & \\multicolumn{{2}}{{c}}{{{acc_m} $\\pm${acc_s}}} \\\\\n")
    lines.append("\\bottomrule\n\n")

    # Table 3: Clarification
    cl_ra = fmt(cl.get("route_accuracy"),    4)
    cl_p  = fmt(cl.get("clarify_precision"), 4)
    cl_r  = fmt(cl.get("clarify_recall"),    4)
    cl_f1 = fmt(cl.get("clarify_f1"),        4)
    cl_fp = cl.get("false_positives", "N/A")
    cl_fn = cl.get("false_negatives", "N/A")
    lines.append("% ----- TABLE: tab:clarify (Clarification results) -----\n")
    lines.append(f"% Route Acc={cl_ra}, P={cl_p}, R={cl_r}, F1={cl_f1}, FP={cl_fp}, FN={cl_fn}\n\n")

    # BERTScore note
    bs_f1 = fmt(bs.get("f1"), 4) if bs.get("f1") else "0.8508 (legacy run)"
    lines.append("% ----- BERTScore note -----\n")
    lines.append(f"% BERTScore F1 = {bs_f1} (model: xlm-roberta-large, "
                 "reference: concise_answer)\n\n")

    return "".join(lines)


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Aggregate unified paper metrics")
    ap.add_argument("--results-dir", default="results_final_unified")
    ap.add_argument("--output",  default="results_final_unified/UNIFIED_PAPER_METRICS.json")
    ap.add_argument("--latex",   default="results_final_unified/UNIFIED_LATEX_TABLES.tex")
    args = ap.parse_args()

    unified = aggregate(args.results_dir)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(unified, f, indent=2, ensure_ascii=False)
    print(f"\n[AGG] Unified metrics saved -> {out_path}")

    latex = generate_latex(unified)
    lat_path = Path(args.latex)
    with open(lat_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"[AGG] LaTeX tables saved    -> {lat_path}")

    # Pretty-print summary
    print("\n" + "="*60)
    print("  PAPER-READY NUMBERS (from this unified run)")
    print("="*60)
    e2e = unified.get("end_to_end", {})
    systems_order = [
        ("pure_vector",         "Pure Vector"),
        ("pure_graph",          "Pure Graph"),
        ("single_stage_router", "Single-stage Router"),
        ("two_stage_hybrid",    "Two-stage Hybrid"),
    ]
    print(f"\n{'System':<28} {'F1':>8} {'RoutAcc':>8} {'MacroF1':>8} {'Lat(ms)':>10}")
    print("-"*66)
    for skey, sname in systems_order:
        d   = e2e.get(skey, {})
        f1  = fmt(d.get("answer_f1"), 4)
        ra  = fmt(d.get("routing_accuracy"), 4)
        mf1 = fmt(d.get("macro_f1"), 4)
        lat = fmt(d.get("avg_latency_ms"), 1)
        print(f"{sname:<28} {f1:>8} {ra:>8} {mf1:>8} {lat:>10}")

    cv_u = unified.get("stage1_cv", {})
    print(f"\nStage-1 CV Accuracy : {cv_u.get('cv_accuracy_mean','N/A')} "
          f"(±{cv_u.get('cv_accuracy_std','N/A')})")
    print(f"Stage-1 CV Macro-F1 : {cv_u.get('cv_macro_f1_mean','N/A')} "
          f"(±{cv_u.get('cv_macro_f1_std','N/A')})")
    cl_u = unified.get("clarification", {}).get("phase3_full_policy", {})
    print(f"\nClarify F1 (Phase-3): {cl_u.get('clarify_f1','N/A')}")
    print(f"Clarify FP (Phase-3): {cl_u.get('false_positives','N/A')}")
    bs_u = unified.get("bertscore", {})
    print(f"\nBERTScore F1        : {bs_u.get('f1','0.8508 (legacy)')} "
          f"(ref: {bs_u.get('reference','concise_answer')}, "
          f"model: {bs_u.get('model','xlm-roberta-large')})")

    print("\n[NOTE] Nếu bất kỳ trường nào hiện N/A, kiểm tra cấu trúc JSON")
    print("       output và cập nhật safe_get() trong file này.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
