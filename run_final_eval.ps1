# ==============================================================
# run_final_eval.ps1
# UNIFIED FINAL EVALUATION — chạy 1 lần lấy toàn bộ số liệu paper
# Chạy từ root folder của project
# Usage: .\run_final_eval.ps1
# ==============================================================

param(
    [string]$Config      = "configs/config.yaml",
    [string]$OutDir      = "results_final_unified",
    [string]$TestFile    = "qa_pipeline/data/legal_strict/test.json",
    [string]$ClarifyFile = "evaluation/legal_clarify_eval.json",
    [string]$ConvFile    = "evaluation/conversation_ambiguity_eval.json"
)

$ErrorActionPreference = "Continue"
$ts = Get-Date -Format "yyyyMMdd_HHmm"
$env:PYTHONPATH = "."

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  UNIFIED FINAL EVALUATION  [$ts]" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ----------------------------------------------------------
# 0. PREREQUISITE CHECK
# ----------------------------------------------------------
Write-Host "[0/6] Checking prerequisites..." -ForegroundColor Yellow

function Check-File($path, $label) {
    if (-not (Test-Path $path)) {
        Write-Host "  MISSING: $label -> $path" -ForegroundColor Red
        exit 1
    }
    Write-Host "  OK: $label" -ForegroundColor Green
}

Check-File $Config      "Config file"
Check-File $TestFile    "Strict test split"
Check-File $ClarifyFile "Clarify eval file"
Check-File $ConvFile    "Conversation eval file"
Check-File "data/router_training/legal_strict/router_model.pkl" "Router checkpoint"

.venv\Scripts\python -c @"
import sys, importlib
for m in ['chromadb', 'xgboost']:
    try:
        importlib.import_module(m)
        print(f'  OK: {m}')
    except:
        print(f'  MISSING pip package: {m}')
        sys.exit(1)
"@
if ($LASTEXITCODE -ne 0) { exit 1 }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Write-Host "  Output dir: $OutDir"
Write-Host ""

# ----------------------------------------------------------
# 1. STAGE 1 — 5-fold Cross-Validation
# ----------------------------------------------------------
Write-Host "[1/6] Stage-1 XGBoost 5-fold CV..." -ForegroundColor Yellow

.venv\Scripts\python router/train_router.py `
    --config $Config `
    --cv-folds 5 `
    --output-dir "$OutDir/router_cv"

if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at Step 1" -ForegroundColor Red; exit 1 }
Write-Host "  Done -> $OutDir/router_cv/training_report.json" -ForegroundColor Green
Write-Host ""

# ----------------------------------------------------------
# 2. END-TO-END BENCHMARK — all 4 systems
# ----------------------------------------------------------
Write-Host "[2/6] End-to-end benchmark (4 systems)..." -ForegroundColor Yellow

# run_benchmark_eval.py outputs to eval_results/ by default (no --output-dir flag)
.venv\Scripts\python scripts/run_benchmark_eval.py `
    --config $Config `
    --dataset legal_strict `
    --eval-file $TestFile `
    --systems pure_vector,pure_graph,single_stage_router,two_stage_hybrid `
    --eval-answer-style

if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at Step 2" -ForegroundColor Red; exit 1 }

# Copy output to unified output dir
New-Item -ItemType Directory -Force -Path "$OutDir/e2e_benchmark" | Out-Null
Copy-Item -Path "eval_results/*" -Destination "$OutDir/e2e_benchmark/" -Recurse -Force
Write-Host "  Done -> $OutDir/e2e_benchmark/" -ForegroundColor Green
Write-Host ""

# ----------------------------------------------------------
# 3. ROUTING-ONLY SANITY CHECK
# ----------------------------------------------------------
Write-Host "[3/6] Routing-only sanity check (no LLM generation)..." -ForegroundColor Yellow

.venv\Scripts\python scripts/evaluate_strict_routing_only.py `
    --config $Config `
    --test-file $TestFile `
    --output-dir "$OutDir/routing_only"

if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at Step 3" -ForegroundColor Red; exit 1 }
Write-Host "  Done -> $OutDir/routing_only/" -ForegroundColor Green
Write-Host ""

# ----------------------------------------------------------
# 4. CLARIFICATION BENCHMARK
# ----------------------------------------------------------
Write-Host "[4/6] Clarification benchmark (234 queries)..." -ForegroundColor Yellow

New-Item -ItemType Directory -Force -Path "$OutDir/clarify" | Out-Null
.venv\Scripts\python scripts/run_clarify_eval.py `
    --config $Config `
    --eval-file $ClarifyFile `
    --output "$OutDir/clarify/clarify_eval_summary.json" `
    --csv-output "$OutDir/clarify/clarify_eval_results.csv"

if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at Step 4" -ForegroundColor Red; exit 1 }
Write-Host "  Done -> $OutDir/clarify/" -ForegroundColor Green
Write-Host ""

# ----------------------------------------------------------
# 5. CONVERSATION AMBIGUITY STRESS TEST
# ----------------------------------------------------------
Write-Host "[5/6] Conversation ambiguity stress test (160 queries)..." -ForegroundColor Yellow

.venv\Scripts\python scripts/evaluate_conversation_ambiguity.py `
    --config $Config `
    --eval-file $ConvFile `
    --output-dir "$OutDir/conv_ambiguity"

if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at Step 5" -ForegroundColor Red; exit 1 }
Write-Host "  Done -> $OutDir/conv_ambiguity/" -ForegroundColor Green
Write-Host ""

# ----------------------------------------------------------
# 6. AGGREGATE ALL RESULTS
# ----------------------------------------------------------
Write-Host "[6/6] Aggregating all results into unified metrics..." -ForegroundColor Yellow

.venv\Scripts\python scripts/aggregate_final_results.py `
    --results-dir $OutDir `
    --output "$OutDir/UNIFIED_PAPER_METRICS.json" `
    --latex "$OutDir/UNIFIED_LATEX_TABLES.tex"

if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at Step 6" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  ALL DONE" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Unified metrics : $OutDir/UNIFIED_PAPER_METRICS.json"
Write-Host "  LaTeX tables    : $OutDir/UNIFIED_LATEX_TABLES.tex"
Write-Host "  Timestamp       : $ts"
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
