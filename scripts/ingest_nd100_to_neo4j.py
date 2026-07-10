import json
import logging
from pathlib import Path
import yaml
from tqdm import tqdm

from graph.neo4j_client import Neo4jClient
from ner.factory import get_ner_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_nd100_to_neo4j():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    client = Neo4jClient(config["neo4j"])
    ner_model = get_ner_model(config["ner"])
    
    docs = []
    # Extract ND100 from phapdien_processed.jsonl
    with open("data/processed/phapdien_processed.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if "Nghị định 100/2019/NĐ-CP" in line or "100/2019/NĐ-CP" in line:
                docs.append(json.loads(line))
                
    logger.info(f"Found {len(docs)} chunks for ND100. Extracting entities and ingesting...")
    
    nodes = []
    relations = []
    
    for doc in tqdm(docs):
        content = doc.get("content", doc.get("chunk_text", ""))
        doc_id = doc.get("doc_id", "Unknown")
        title = doc.get("title", f"Document {doc_id}")
        
        # Add LegalArticle node
        article_name = f"{title}_{doc_id}"
        nodes.append({
            "name": article_name,
            "type": "LegalArticle",
            "properties": {
                "title": title,
                "doc_id": doc_id,
                "article_id": doc_id,
                "content": content
            }
        })
        
        # Add LegalDoc node
        nodes.append({
            "name": "Nghị định 100/2019/NĐ-CP",
            "type": "LegalDoc",
            "properties": {
                "title": "Nghị định 100/2019/NĐ-CP",
                "doc_id": "100/2019/NĐ-CP",
                "type": "Nghị định",
                "authority": "Chính phủ"
            }
        })
        
        relations.append({
            "source": article_name,
            "target": "Nghị định 100/2019/NĐ-CP",
            "relation_type": "PART_OF",
            "properties": {}
        })
        
        # Extract entities
        try:
            entities_list = ner_model.extract([content])
            entities = entities_list[0] if entities_list else []
            for ent in entities:
                ent_name = ent.text.strip()
                if not ent_name: continue
                
                nodes.append({
                    "name": ent_name,
                    "type": ent.label,
                    "properties": {}
                })
                
                relations.append({
                    "source": ent_name,
                    "target": article_name,
                    "relation_type": "MENTIONED_IN",
                    "properties": {}
                })
        except Exception as e:
            logger.warning(f"NER failed for chunk {doc_id}: {e}")
            
    # Batch insert
    client.batch_insert_nodes(nodes)
    client.batch_insert_relations(relations)
    logger.info("Ingestion complete!")
    client.close()

if __name__ == "__main__":
    ingest_nd100_to_neo4j()
