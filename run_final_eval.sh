#!/usr/bin/env bash
# ==============================================================
# run_final_eval.sh  (Linux / Mac / WSL)
# UNIFIED FINAL EVALUATION
# Usage: ./run_final_eval.sh [config] [out_dir]
# ==============================================================

set -e

CONFIG="${1:-configs/config.yaml}"
OUT_DIR="${2:-results_final_unified}"
TEST_FILE="qa_pipeline/data/legal_strict/test.json"
CLARIFY_FILE="evaluation/legal_clarify_eval.json"
CONV_FILE="evaluation/conversation_ambiguity_eval.json"
TS=$(date +"%Y%m%d_%H%M")
PYTHON=".venv/bin/python"
export PYTHONPATH="."

echo ""
echo "========================================"
echo "  UNIFIED FINAL EVALUATION  [$TS]"
echo "========================================"

# 0. Prerequisites
echo ""
echo "[0/6] Checking prerequisites..."
for f in "$CONFIG" "$TEST_FILE" "$CLARIFY_FILE" "$CONV_FILE" \
         "data/router_training/legal_strict/router_model.pkl"; do
    [ -f "$f" ] || { echo "  MISSING: $f"; exit 1; }
    echo "  OK: $f"
done

$PYTHON -c "
import sys, importlib
for m in ['chromadb','xgboost']:
    try: importlib.import_module(m); print(f'  OK: {m}')
    except: print(f'  MISSING: {m}'); sys.exit(1)
"

mkdir -p "$OUT_DIR"
echo "  Output dir: $OUT_DIR"

# 1. Stage-1 5-fold CV
echo ""
echo "[1/6] Stage-1 XGBoost 5-fold CV..."
$PYTHON router/train_router.py \
    --config "$CONFIG" \
    --cv-folds 5 \
    --output-dir "$OUT_DIR/router_cv"
echo "  Done -> $OUT_DIR/router_cv/training_report.json"

# 2. End-to-end benchmark (all 4 systems)
echo ""
echo "[2/6] End-to-end benchmark (4 systems)..."
$PYTHON scripts/run_benchmark_eval.py \
    --config "$CONFIG" \
    --dataset legal_strict \
    --eval-file "$TEST_FILE" \
    --systems pure_vector,pure_graph,single_stage_router,two_stage_hybrid \
    --eval-answer-style \
    --output-dir "$OUT_DIR/e2e_benchmark"
echo "  Done -> $OUT_DIR/e2e_benchmark/"

# 3. Routing-only sanity check
echo ""
echo "[3/6] Routing-only sanity check..."
$PYTHON scripts/evaluate_strict_routing_only.py \
    --config "$CONFIG" \
    --test-file "$TEST_FILE" \
    --output-dir "$OUT_DIR/routing_only"
echo "  Done -> $OUT_DIR/routing_only/"

# 4. Clarification benchmark
echo ""
echo "[4/6] Clarification benchmark (234 queries)..."
mkdir -p "$OUT_DIR/clarify"
$PYTHON scripts/run_clarify_eval.py \
    --config "$CONFIG" \
    --eval-file "$CLARIFY_FILE" \
    --output "$OUT_DIR/clarify/clarify_eval_summary.json" \
    --csv-output "$OUT_DIR/clarify/clarify_eval_results.csv"
echo "  Done -> $OUT_DIR/clarify/"

# 5. Conversation ambiguity stress test
echo ""
echo "[5/6] Conversation ambiguity stress test (160 queries)..."
$PYTHON scripts/evaluate_conversation_ambiguity.py \
    --config "$CONFIG" \
    --eval-file "$CONV_FILE" \
    --output-dir "$OUT_DIR/conv_ambiguity"
echo "  Done -> $OUT_DIR/conv_ambiguity/"

# 6. Aggregate
echo ""
echo "[6/6] Aggregating all results..."
$PYTHON scripts/aggregate_final_results.py \
    --results-dir "$OUT_DIR" \
    --output "$OUT_DIR/UNIFIED_PAPER_METRICS.json" \
    --latex "$OUT_DIR/UNIFIED_LATEX_TABLES.tex"

echo ""
echo "========================================"
echo "  ALL DONE"
echo "========================================"
echo "  Unified metrics : $OUT_DIR/UNIFIED_PAPER_METRICS.json"
echo "  LaTeX tables    : $OUT_DIR/UNIFIED_LATEX_TABLES.tex"
echo "  Timestamp       : $TS"
echo "========================================"
echo ""
