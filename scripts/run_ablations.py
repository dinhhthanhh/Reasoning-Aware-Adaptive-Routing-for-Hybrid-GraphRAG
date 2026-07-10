import os
import subprocess

def run_ablations():
    print("Running RQ3: conversation ambiguity (160 queries)...")
    subprocess.run(["python", "-W", "ignore", "scripts/evaluate_conversation_ambiguity.py", "--output-dir", "eval_results"], check=True)
    
    print("Running RQ4: clarify routing benchmark (234 queries)...")
    subprocess.run(["python", "-W", "ignore", "scripts/run_clarify_eval.py", "--output", "eval_results/run_clarify_results.json"], check=True)
    
if __name__ == '__main__':
    run_ablations()
