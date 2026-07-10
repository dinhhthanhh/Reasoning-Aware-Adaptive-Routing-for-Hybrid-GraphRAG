import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from neo4j import GraphDatabase, exceptions
import tqdm
from ner.factory import get_ner_model


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

ALLOWED_LABELS = {"LegalDoc", "HF", "PD"}

class Neo4jIngestor:
    def __init__(self, config, ner_config=None):
        try:
            self.driver = GraphDatabase.driver(
                config["uri"], 
                auth=(config["user"], config["password"])
            )
            self.database = config.get("database", "neo4j")
            self.driver.verify_connectivity()
            
            # Initialize NER for entity extraction
            self.ner_model = get_ner_model(ner_config) if ner_config else None
            
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j driver: {e}")
            raise

    def close(self):
        self.driver.close()

    def create_constraints(self):
        with self.driver.session(database=self.database) as session:
            try:
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:LegalDoc) REQUIRE d.doc_id IS UNIQUE")
                session.run("CREATE INDEX IF NOT EXISTS FOR (d:LegalDoc) ON (d.title)")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
                session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)")
                
                # New indices for the source-specific labels
                session.run("CREATE INDEX IF NOT EXISTS FOR (d:HF) ON (d.doc_id)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (d:PD) ON (d.doc_id)")
                logger.info("Constraints and indices initialized.")
            except exceptions.Neo4jError as e:
                logger.error(f"Error creating constraints: {e}")


    def _ingest_docs_batch(self, tx, batch, label):
        if label not in ALLOWED_LABELS:
            raise ValueError(f"Invalid label requested: {label}")

        # Apply secondary label based on source
        query = f"""
        UNWIND $batch as row
        MERGE (d:{label} {{doc_id: row.doc_id}})
        SET d.title = row.title,
            d.type = row.type,
            d.authority = row.authority,
            d.source = row.source,
            d.issue_date = row.issue_date
        WITH d, row
        CALL apoc.create.addLabels([d], [row.sub_label]) YIELD node
        RETURN count(*)
        """
        # Note: If APOC is not available, we use a simpler approach:
        query_no_apoc = f"""
        UNWIND $batch as row
        MERGE (d:{label} {{doc_id: row.doc_id}})
        SET d.title = row.title,
            d.type = row.type,
            d.authority = row.authority,
            d.source = row.source,
            d.issue_date = row.issue_date
        FOREACH (x in CASE WHEN row.sub_label = "HF" THEN [1] ELSE [] END | SET d:HF)
        FOREACH (x in CASE WHEN row.sub_label = "PD" THEN [1] ELSE [] END | SET d:PD)
        """
        tx.run(query_no_apoc, batch=batch)

    def _ingest_entities_batch(self, tx, entities, doc_id):
        query = """
        UNWIND $entities as ent
        MERGE (e:Entity {name: ent.text})
        SET e.type = ent.label
        WITH e
        MATCH (d:LegalDoc {doc_id: $doc_id})
        MERGE (e)-[:MENTIONED_IN]->(d)
        """
        tx.run(query, entities=entities, doc_id=doc_id)
        
        # Link entities in the same document
        if len(entities) > 1:
            query_link = """
            UNWIND $entities as ent1
            UNWIND $entities as ent2
            WITH ent1, ent2 WHERE ent1.text < ent2.text
            MATCH (e1:Entity {name: ent1.text})
            MATCH (e2:Entity {name: ent2.text})
            MERGE (e1)-[:CO_OCCURRED]-(e2)
            """
            tx.run(query_link, entities=entities)

    def ingest_documents(self, file_path: Path, label="LegalDoc", sub_label="HF"):
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return
        
        logger.info(f"Ingesting documents and entities from {file_path} as :{label}...")
        
        batch = []
        batch_size = 500  # Smaller batch for NER safety
        
        with open(file_path, "r", encoding="utf-8") as f:
            with self.driver.session(database=self.database) as session:
                for line in tqdm.tqdm(f, desc=f"Loading {label}"):
                    try:
                        line = line.strip()
                        if not line: continue
                        
                        item = json.loads(line)
                        doc_id = item.get("doc_id")
                        if doc_id is None: continue
                        content = item.get("content", item.get("content_markdown", item.get("chunk_text", "")))
                        
                        # Ingest Document Node
                        doc_data = {
                            "doc_id": str(doc_id),
                            "title": str(item.get("title", "")),
                            "type": str(item.get("type", "")),
                            "authority": str(item.get("authority", "")),
                            "source": str(item.get("source", "HuggingFace")),
                            "issue_date": str(item.get("issue_date", "")),
                            "sub_label": sub_label
                        }
                        batch.append(doc_data)
                        
                        # Process batch
                        if len(batch) >= batch_size:
                            session.execute_write(self._ingest_docs_batch, batch, label)
                            batch = []
                            
                        # Entity Extraction (if model available)
                        if self.ner_model and content:
                            entities_list = self.ner_model.extract([content])
                            entities = entities_list[0] if entities_list else []
                            if entities:
                                # Ensure doc node exists before linking
                                session.execute_write(self._ingest_docs_batch, [doc_data], label)
                                
                                # Ingest Entities and Links
                                ent_data = [{"text": e.text, "label": e.label} for e in entities]
                                session.execute_write(self._ingest_entities_batch, ent_data, str(doc_id))

                    except Exception as e:
                        logger.error(f"Error in document ingestion: {e}")
                        continue
                
                if batch:
                    session.execute_write(self._ingest_docs_batch, batch, label)


    def _ingest_rels_batch(self, tx, batch):
        # Using MATCH for nodes to avoid ghost nodes. 
        # Orphaned relationships are skipped.
        query = """
        UNWIND $batch as row
        MATCH (src:LegalDoc {doc_id: row.source})
        MATCH (tgt:LegalDoc {doc_id: row.target})
        MERGE (src)-[r:REFERENCES]->(tgt)
        SET r.type = row.type
        """
        tx.run(query, batch=batch)

    def ingest_relationships(self, file_path: Path):
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return
        
        logger.info(f"Ingesting relationships from {file_path}...")
        
        batch = []
        batch_size = 1000
        with open(file_path, "r", encoding="utf-8") as f:
            with self.driver.session(database=self.database) as session:
                for line in tqdm.tqdm(f, desc="Linking nodes"):
                    try:
                        line = line.strip()
                        if not line: continue
                        
                        rel = json.loads(line)
                        src = rel.get("source")
                        tgt = rel.get("target")
                        
                        if src is None or tgt is None:
                            continue
                            
                        batch.append({
                            "source": str(src),
                            "target": str(tgt),
                            "type": rel.get("type", "Unknown")
                        })
                        
                        if len(batch) >= batch_size:
                            session.execute_write(self._ingest_rels_batch, batch)
                            batch = []
                    except json.JSONDecodeError as e:
                        logger.warning(f"Malformed JSON in rels: {e}")
                    except Exception as e:
                        logger.error(f"Error in relationship ingestion: {e}")
                        continue
                        
                if batch:
                    session.execute_write(self._ingest_rels_batch, batch)

    def clear_database(self):
        """Clears the database in batches to avoid high memory usage."""
        logger.warning("Clearing database in batches...")
        with self.driver.session(database=self.database) as session:
            while True:
                result = session.run(
                    "MATCH (n) WITH n LIMIT 1000 DETACH DELETE n RETURN count(*) as deleted"
                )
                deleted = result.single()["deleted"]
                if deleted > 0:
                    logger.info(f"Deleted {deleted} nodes...")
                if deleted == 0:
                    break

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    
    config_path = Path(args.config)
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        is_en = "config_en" in str(args.config)
        data_dir = config.get("data", {}).get("processed_dir", "data/processed")
        
        # Pass NER config for entity extraction for both English and Vietnamese
        ner_config = config.get("ner")
        ingestor = Neo4jIngestor(config["neo4j"], ner_config=ner_config)

        try:
            # Clear the DB to ensure a clean start with the fixed ingestion logic
            ingestor.clear_database() 
            
            ingestor.create_constraints()
            
            if is_en:
                # English benchmark datasets
                ingestor.ingest_documents(Path(data_dir) / "hotpot_full.jsonl", sub_label="HotpotQA")
            else:
                # Ingest Core Laws, HF, Phap Dien 
                ingestor.ingest_documents(Path(data_dir) / "core_laws_processed.jsonl", sub_label="CoreLaws")
                ingestor.ingest_documents(Path(data_dir) / "hf_processed.jsonl", sub_label="HF")
                ingestor.ingest_documents(Path(data_dir) / "phapdien_processed.jsonl", sub_label="PD")
            
                # Relationships
                ingestor.ingest_relationships(Path(data_dir) / "relationships_final.jsonl")
            
            logger.info("Graph ingestion successfully completed.")
        finally:
            ingestor.close()
    except Exception as e:
        logger.error(f"Critical error: {e}")

if __name__ == "__main__":
    main()
