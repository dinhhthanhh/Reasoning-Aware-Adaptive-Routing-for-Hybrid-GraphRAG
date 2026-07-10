python -W ignore -m evaluation.run_full_eval --systems pure_vector --out eval_results --verbose > eval_run_pv.log 2>&1
python -W ignore -m evaluation.run_full_eval --systems pure_graph --out eval_results --verbose > eval_run_pg.log 2>&1
python -W ignore -m evaluation.run_full_eval --systems pure_hybrid --out eval_results --verbose > eval_run_ph.log 2>&1
