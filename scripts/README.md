# Scripts

This folder contains only reproducible entry points for the thesis experiments.

## Primary Vietnamese Legal QA Workflow

1. Crawl or update raw legal data:

```powershell
python main.py --source all --output data/
```

2. Build the legal graph and vector index:

```powershell
python scripts/build_kg.py --config configs/config.yaml
python scripts/build_vectordb.py --config configs/config.yaml
```

3. Enrich Neo4j with article/content evidence and vector links:

```powershell
python scripts/enrich_neo4j_legal_graph.py --config configs/config.yaml --limit-vector-chunks -1 --batch-size 2000
```

This is incremental and does not clear the database. It adds `LegalArticle`, `VectorChunk`, `LegalConcept`, and evidence relations such as `HAS_ARTICLE`, `HAS_VECTOR_CHUNK`, `CROSS_REFERENCES`, `INTRA_DOC_REFERENCE`, `AMENDS`, `REPEALS`, `GUIDES`, and `REGULATES_CONCEPT`.

4. Check Neo4j graph quality:

```powershell
python scripts/check_neo4j_graph_quality.py --config configs/config.yaml --output eval_results/legal_graph_quality.json
```

5. Train and evaluate the router:

```powershell
python scripts/run_router_training.py --config configs/config.yaml --train_path qa_pipeline/data/final/train.json --dev_path qa_pipeline/data/final/dev.json --test_path qa_pipeline/data/final/test.json
```

6. Run end-to-end baseline comparison:

```powershell
python scripts/run_benchmark_eval.py --config configs/config.yaml --dataset legal --systems all --eval-answer-style
```

7. Interactive demo:

```powershell
python scripts/run_pipeline.py --config configs/config.yaml --verbose
python scripts/run_conversation_demo.py --config configs/config.yaml
```

## Secondary Benchmarks

HotpotQA is English/open-domain and must use `configs/config_hotpot.yaml`.

```powershell
python scripts/download_en_benchmark.py
python scripts/prepare_router_data.py
python scripts/run_router_training.py --config configs/config_hotpot.yaml --train_path data/en_benchmark/router_training/train.json --dev_path data/en_benchmark/router_training/train.json --test_path data/en_benchmark/router_training/train.json
python scripts/run_benchmark_eval.py --config configs/config_hotpot.yaml --dataset hotpot --systems all
```

ViMQA is Vietnamese/open-domain and should be reported as a secondary generalization benchmark.

```powershell
python scripts/download_vimqa.py --output data/vimqa
python scripts/run_router_training.py --config configs/config_vimqa.yaml --train_path data/vimqa/train.json --dev_path data/vimqa/validation.json --test_path data/vimqa/test.json
python scripts/run_benchmark_eval.py --config configs/config_vimqa.yaml --dataset vimqa --systems all
```

## Data Preparation Utilities

- `clean_and_merge_hf.py`, `clean_phapdien.py`, `resolve_relationships.py`, `finalize_corpus.py`: build the Vietnamese legal corpus from crawled sources.
- `run_qa_generation.py`: generate or verify Vietnamese legal QA data.
- `analyze_qa_dataset.py`: report routing-label distribution and dataset balance.
- `ablation_study.py`: routing-only ablation over thresholds/features.
