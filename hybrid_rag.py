import os
import sys
import argparse
from pathlib import Path
import yaml
from loguru import logger

# Add current directory to path
sys.path.append(os.getcwd())

from pipeline.hybrid_pipeline import HybridPipeline

def main():
    parser = argparse.ArgumentParser(description="Hybrid RAG Router for Vietnamese Legal QA")
    parser.add_argument("--query", type=str, help="Single query to process")
    parser.add_argument("--eval", action="store_true", help="Run evaluation suite")
    parser.add_argument("--session", type=str, default="cli_session", help="Session ID")
    parser.add_argument("--verbose", action="store_true", help="Print detailed routing info")
    args = parser.parse_args()

    # 1. Initialize Pipeline
    logger.info("Initializing Hybrid Pipeline...")
    pipeline = HybridPipeline()

    # 2. Process single query
    if args.query:
        logger.info(f"Query: {args.query}")
        response = pipeline.query(args.query, session_id=args.session, verbose=args.verbose)
        
        print("\n" + "="*50)
        print(f"ROUTE: {response.route_used} (Conf: {response.confidence:.2f})")
        print(f"REASONING: {response.router_reasoning}")
        print("-" * 50)
        print(f"ANSWER:\n{response.answer}")
        print("-" * 50)
        if response.sources:
            print(f"SOURCES: {', '.join(response.sources)}")
        print(f"LATENCY: {response.latency_ms:.0f}ms")
        print("="*50 + "\n")
        return

    # 3. Process evaluation
    if args.eval:
        logger.info("Running evaluation suite...")
        # Placeholder for eval logic (usually in evaluation/ scripts)
        # For now, we simulate a small run if evaluation/data.json exists
        eval_data_path = Path("data/evaluation/test_set.json")
        if eval_data_path.exists():
            import json
            with open(eval_data_path, "r", encoding="utf-8") as f:
                test_set = json.load(f)
            
            results = []
            for item in test_set[:10]: # First 10 for quick eval
                q = item["query"]
                print(f"Evaluating: {q}")
                res = pipeline.query(q)
                results.append({
                    "query": q,
                    "target_route": item.get("target_route"),
                    "actual_route": res.route_used,
                    "latency": res.latency_ms
                })
            
            output_path = Path("output/eval_results.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Evaluation results saved to {output_path}")
        else:
            logger.warning("No evaluation data found at data/evaluation/test_set.json. Skipping evaluation.")
        return

    # 4. Interactive mode
    print("\nVN Legal Hybrid RAG CLI - Gõ 'exit' để thoát")
    while True:
        try:
            query = input("User: ")
            if query.lower() in ["exit", "quit", "thoát"]:
                break
            if not query.strip():
                continue
                
            response = pipeline.query(query, session_id=args.session, verbose=True)
            print(f"\n[Route: {response.route_used} | {response.latency_ms:.0f}ms]")
            print(f"Assistant: {response.answer}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()
