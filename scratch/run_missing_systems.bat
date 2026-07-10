python -W ignore -m evaluation.run_full_eval --systems single_stage --out eval_results --verbose > eval_run_single_stage.log 2>&1
python -W ignore -m evaluation.run_full_eval --systems always_on --out eval_results --verbose > eval_run_always_on.log 2>&1
python -W ignore -m evaluation.run_full_eval --systems pure_vector --out eval_results --verbose > eval_run_pure_vector.log 2>&1
python -W ignore -m evaluation.run_full_eval --systems pure_graph --out eval_results --verbose > eval_run_pure_graph.log 2>&1
