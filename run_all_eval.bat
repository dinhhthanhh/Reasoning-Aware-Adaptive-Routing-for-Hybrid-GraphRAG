@echo off
echo ==================================================
echo PRIMARY LEGAL QA ROUTING EXPERIMENT
echo ==================================================

echo ==================================================



echo [1/4] Checking Legal Neo4j graph quality...
python scripts\check_neo4j_graph_quality.py --config configs\config_legal.yaml --output eval_results\legal_graph_quality.json

echo [2/4] Training legal router...
python scripts\run_router_training.py --config configs\config_legal.yaml --train_path qa_pipeline\data\final\train.json --dev_path qa_pipeline\data\final\dev.json --test_path qa_pipeline\data\final\test.json

echo [3/4] Running legal end-to-end baseline comparison...
python scripts\run_benchmark_eval.py --config configs\config_legal.yaml --dataset legal --systems all

echo [4/4] Done.

echo ==================================================
echo LEGAL QA EXPERIMENT COMPLETE!
echo Results saved in eval_results\
echo ==================================================
pause
