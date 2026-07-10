$ErrorActionPreference = "Stop"

Write-Host "Starting paraphrasing..."
python scripts\phapdien_pipeline\paraphrase_benchmark.py
if ($LASTEXITCODE -ne 0) { throw "Paraphrasing failed" }

Write-Host "`nStarting Router Training..."
python scripts\train_router_enriched.py --data-dir qa_pipeline\data\phapdien_strict --output-dir data\router_training\phapdien_strict
if ($LASTEXITCODE -ne 0) { throw "Router training failed" }

Write-Host "`nStarting End-to-End Evaluation..."
python scripts\run_comparison_eval.py --test-path qa_pipeline\data\phapdien_strict\test.json --configs pure_vector,pure_graph,pure_hybrid,router,oracle --max-samples 600 --stratified --output-dir eval_results\comparison
if ($LASTEXITCODE -ne 0) { throw "Evaluation failed" }

Write-Host "`nAll completed successfully."
