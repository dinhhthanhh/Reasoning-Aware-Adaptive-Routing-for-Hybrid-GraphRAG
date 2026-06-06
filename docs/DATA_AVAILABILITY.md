# Data Availability Statement

For the publication of the **Reasoning-aware Adaptive Routing for Hybrid GraphRAG** project, we enforce a strict policy regarding raw datasets, generated indexes, and experimental artifacts to maintain a clean and lightweight repository.

## Excluded Assets

The following assets are **not committed** to this repository:
- **Raw Legal Crawl Data**: Extracted documents from external sources (e.g., HuggingFace datasets, legal portals) are not hosted here due to size and licensing constraints.
- **Generated Indexes**: The constructed Knowledge Graph (Neo4j) and Vector Indexes (FAISS/Chroma) are omitted.
- **Evaluation Logs & Artifacts**: Raw evaluation benchmark results, LLM generations, routing logs, and intermediate reports are ignored by default via `.gitignore` to prevent repository bloat.
- **Private Documentation**: Internal progress notes, thesis materials, and private architectural documents are excluded.

## Reproduction Steps

To fully reproduce the project, users must reconstruct the environment locally:
1. **Prepare the Legal Corpus**: Use the provided scripts in the `crawlers/` and `pipeline/` directories to fetch and process the required raw documents.
2. **Build the Indexes**: Use `scripts/build_kg.py` and `scripts/build_vectordb.py` to generate the Neo4j Knowledge Graph and Vector Database.
3. **Configure the Environment**: Reproduction requires an active Neo4j instance and a valid LLM endpoint (e.g., OpenAI or a local vLLM server). 

Safe, sanitized configuration templates are provided in the repository:
- `configs/config.yaml.example`
- `configs/config_en.yaml.example`
- `configs/config_hotpot.yaml.example`
- `configs/config_vimqa.yaml.example`

Please copy these `.example` files and populate them with your own credentials and endpoints before running the system.
