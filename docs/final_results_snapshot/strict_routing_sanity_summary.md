# Strict Routing Sanity Summary

- Eval file: `qa_pipeline\data\legal_strict\test.json`
- Total samples: `600`
- Route accuracy: `0.8933`
- Clarify false positives: `10`
- Stage 2 trigger rate: `0.5583`
- Stage 2 override rate: `0.3403`
- Avg latency ms: `1972.3544133337175`
- Route distribution: `{'dense_retrieval': 275, 'graph_traversal': 164, 'hybrid_reasoning': 151, 'clarify': 10}`

## By Gold Route

| Gold route | Total | Route Acc. | Clarify FP | Pred routes |
|---|---:|---:|---:|---|
| `dense_retrieval` | 300 | 0.883 | 2 | `{'dense_retrieval': 265, 'graph_traversal': 32, 'clarify': 2, 'hybrid_reasoning': 1}` |
| `graph_traversal` | 150 | 0.853 | 5 | `{'graph_traversal': 128, 'clarify': 5, 'hybrid_reasoning': 7, 'dense_retrieval': 10}` |
| `hybrid_reasoning` | 150 | 0.953 | 3 | `{'hybrid_reasoning': 143, 'clarify': 3, 'graph_traversal': 4}` |
