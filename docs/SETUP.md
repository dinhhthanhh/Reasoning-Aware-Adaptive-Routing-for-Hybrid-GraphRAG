# Setup Guide

This project needs Python, Neo4j, a Chroma vector store, and optionally an
OpenAI-compatible LLM endpoint for Stage 2 and answer generation.

## Python Environment

Recommended Python: `3.10+`.

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Conda alternative:

```bash
conda env create -f environment.yml
conda activate hybrid-graphrag
```

## Environment Variables

Copy the placeholder file and fill in local values:

```bash
copy .env.example .env
copy configs\config.yaml.example configs\config.yaml
```

Required or commonly used variables:

```text
OPENAI_API_KEY=<your-key-or-not-required>
OPENAI_BASE_URL=<openai-compatible-base-url>
OPENAI_MODEL=<model-name>
NEO4J_URI=<neo4j-uri>
NEO4J_USER=<neo4j-user>
NEO4J_PASSWORD=<neo4j-password>
```

Never commit `.env`, local `configs/config.yaml`, passwords, API keys, private
endpoints, model weights, vector stores, or Neo4j database files.

## Neo4j

Neo4j 5.x is recommended. The local config should point to the correct
`NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD`.

Smoke check:

```bash
python scripts/smoke_neo4j_graphrag.py
```

Post-migration graph quality artifact:

```text
eval_results/post_migration_graph_quality.json
docs/final_results_snapshot/post_migration_graph_quality.json
```

## Vector Store

The paper experiments used:

```text
data/vector_store/chroma_harrier_oss_0_6b
```

This folder is not committed because it is large/local. If missing, rebuild or
obtain the artifact separately before running full retrieval/generation
experiments.

## Minimal Local Checks

These do not rebuild vector DB/Neo4j and do not rerun the full benchmark:

```bash
python -m compileall router graph llm scripts
python -m pytest tests/ -v
python scripts/demo_conversation_routing.py --config configs/config.yaml
```

If Neo4j, Chroma, or the LLM endpoint is not available, routing-only scripts may
still work when they only load the router checkpoint and cached outputs. Record
environment failures in `docs/reproducibility_audit.md` instead of changing
paper metrics.
