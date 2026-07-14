import sys
import os
from pipeline.hybrid_pipeline import HybridPipeline

def main():
    pipeline = HybridPipeline()
    print("--- Single Query ---")
    q = "Người dưới 18 tuổi đi xe trên 70 cc có bị phạt không?"
    resp = pipeline.query(query=q, session_id="demo3", verbose=False)
    print(f"Question: {q}")
    print(f"Route: {resp.route_used} | Conf: {resp.confidence:.1%} | Stage 1: {resp.stage1_route} | Stage 2 invoked: {resp.stage2_invoked} | Latency: {resp.latency_ms:.0f}ms | Sources: {len(resp.sources)}")
    print(f"Answer snippet: {resp.answer[:200]}")

if __name__ == "__main__":
    main()
