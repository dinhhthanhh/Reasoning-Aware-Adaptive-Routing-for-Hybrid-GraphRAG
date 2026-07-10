"""Build Layer 2 Semantic Knowledge Graph for Phap Dien.

Targeting exclusively the `phapdien` database. Extracts semantic
relations from `LegalArticle` nodes with resilient LLM handling,
canonicalization, bidirectional provenance, and safe constraints.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from loguru import logger
from neo4j import exceptions

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from graph.neo4j_client import Neo4jClient
from llm.openai_client import OpenAIClient

CONFIG = ROOT / "configs/config.yaml"
TARGET_DATABASE = "phapdien"

SEMANTIC_LABELS = [
    "LEGAL_CONCEPT", "ACTOR", "AUTHORITY", "RIGHT", "OBLIGATION", 
    "CONDITION", "PROCEDURE", "PENALTY"
]

VALID_RELS = {
    "REGULATES", "GOVERNS", "APPLIES_TO", "OBLIGATES", "PERMITS", "PROHIBITS", 
    "REQUIRES", "HAS_CONDITION", "HAS_EXCEPTION", "ELIGIBLE_FOR", "NOT_ELIGIBLE_FOR", 
    "RESPONSIBLE_FOR", "AUTHORIZES", "SUPERVISES", "ENFORCES", "ISSUED_BY", 
    "GUIDED_BY", "IMPLEMENTS", "REFERS_TO", "AMENDS", "REPLACES", "ABOLISHES", 
    "PART_OF_PROCEDURE", "NEXT_STEP", "RELATED_CONCEPT"
}

try:
    with open(ROOT / "data" / "alias_map.json", "r", encoding="utf-8") as f:
        ALIAS_MAP = json.load(f)
except FileNotFoundError:
    ALIAS_MAP = {}


class SemanticStats:
    def __init__(self):
        self.articles_processed = 0
        self.triples_generated = 0
        self.triples_kept = 0
        self.triples_filtered = 0
        
        self.concept_counts = Counter()
        self.actor_counts = Counter()
        self.right_counts = Counter()
        self.obligation_counts = Counter()
        self.condition_counts = Counter()
        self.relation_counts = Counter()
        
    def add_kept_triple(self, sub_label, sub_name, rel, obj_label, obj_name):
        self.relation_counts[rel] += 1
        
        for label, name in [(sub_label, sub_name), (obj_label, obj_name)]:
            if label == "LEGAL_CONCEPT": self.concept_counts[name] += 1
            elif label == "ACTOR": self.actor_counts[name] += 1
            elif label == "RIGHT": self.right_counts[name] += 1
            elif label == "OBLIGATION": self.obligation_counts[name] += 1
            elif label == "CONDITION": self.condition_counts[name] += 1

    def print_report(self):
        print("\n" + "="*50)
        print("SEMANTIC KG STATS")
        print("="*50)
        print(f"Articles Processed: {self.articles_processed}")
        print(f"Triples Generated:  {self.triples_generated}")
        print(f"Triples Kept:       {self.triples_kept}")
        print(f"Triples Filtered:   {self.triples_filtered}")
        
        print("\nNode Counts:")
        print(f"  LEGAL_CONCEPT: {len(self.concept_counts)} unique")
        print(f"  ACTOR:         {len(self.actor_counts)} unique")
        print(f"  RIGHT:         {len(self.right_counts)} unique")
        
        print("\nRelation Counts (Top 5):")
        for k, v in self.relation_counts.most_common(5):
            print(f"  {k}: {v}")
        print("="*50 + "\n")

    def export(self, path: Path):
        data = {
            "articles_processed": self.articles_processed,
            "triples_generated": self.triples_generated,
            "triples_kept": self.triples_kept,
            "triples_filtered": self.triples_filtered,
            "unique_concepts": len(self.concept_counts),
            "unique_actors": len(self.actor_counts),
            "unique_rights": len(self.right_counts),
            "unique_obligations": len(self.obligation_counts),
            "unique_conditions": len(self.condition_counts),
            "top_relations": dict(self.relation_counts.most_common(20))
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def create_semantic_schema(client: Neo4jClient):
    """Create unique constraints for fast and safe semantic MERGE operations."""
    logger.info(f"Creating semantic schema constraints in database '{client.database}'...")
    statements = [f"CREATE CONSTRAINT {l.lower()}_unique IF NOT EXISTS FOR (n:{l}) REQUIRE n.name IS UNIQUE" for l in SEMANTIC_LABELS]
    for stmt in statements:
        try:
            client.query(stmt)
        except Exception as e:
            logger.debug(f"Schema stmt error/skipped: {e}")


def get_unprocessed_batch(client: Neo4jClient, limit: int = 100) -> list[dict]:
    """Fetch LegalArticles that haven't been semantically processed yet."""
    cypher = """
    MATCH (a:LegalArticle)
    WHERE a.semantic_processed IS NULL AND a.content IS NOT NULL
    RETURN a.article_id AS article_id, a.content AS content
    ORDER BY a.article_id
    LIMIT $limit
    """
    return client.query(cypher, {"limit": limit})


def extract_semantics_with_retry(llm_client: OpenAIClient, content: str, max_retries: int = 10) -> list[dict]:
    """Extract semantics using LLM with exponential backoff and strict rules."""
    prompt = """
    Bạn là chuyên gia về Đồ thị Tri thức Pháp luật.
    Trích xuất thực thể và quan hệ ngữ nghĩa từ đoạn văn bản sau.
    
    Loại thực thể (Node Labels):
    LEGAL_CONCEPT, ACTOR, AUTHORITY, OBLIGATION, RIGHT, PROCEDURE, CONDITION, PENALTY
    
    Loại quan hệ (Relations):
    REGULATES, APPLIES_TO, OBLIGATES, PERMITS, PROHIBITS, RESPONSIBLE_FOR, 
    ELIGIBLE_FOR, HAS_CONDITION, HAS_EXCEPTION, AUTHORIZES, ENFORCES, SUPERVISES, 
    PART_OF_PROCEDURE, NEXT_STEP, AMENDS, REPLACES, REFERS_TO
    
    QUY TẮC CẤM KỴ (CHỐNG RÁC ĐỒ THỊ):
    Tuyệt đối KHÔNG TẠO các thực thể (concept, actor, condition) chung chung như: "quy định", "nhà nước", "cơ quan", "tổ chức", "cá nhân", "đối tượng", "trách nhiệm", "quyền", "nghĩa vụ", "điều kiện", "thủ tục", "hồ sơ", "trường hợp" NẾU CHÚNG KHÔNG ĐI KÈM MỘT CỤM TỪ PHÁP LÝ CỤ THỂ.
    - BAD: LEGAL_CONCEPT("quy định"), ACTOR("cơ quan"), CONDITION("điều kiện"), PROCEDURE("thủ tục")
    - GOOD: LEGAL_CONCEPT("quy định về miễn giảm tiền thuê đất"), ACTOR("Cơ quan nhà nước có thẩm quyền"), CONDITION("điều kiện được miễn tiền thuê đất")
    
    Định dạng JSON (BẮT BUỘC, CHỈ XUẤT MẢNG 2 CHIỀU NGẮN GỌN):
    [
      ["ACTOR", "ủy ban nhân dân cấp tỉnh", "RESPONSIBLE_FOR", "OBLIGATION", "công bố giá", 0.95]
    ]
    (Thứ tự: subject_label, subject_name, relation, object_label, object_name, confidence).
    Nếu quan hệ xuất phát từ chính Điều luật này tới thực thể, dùng subject_label: "LegalArticle" và subject_name: "ARTICLE_NODE". Thuộc tính confidence (0.0-1.0) là BẮT BUỘC.
    
    Văn bản:
    """ + content
    content_len = len(content)
    if content_len < 500:
        dynamic_max_tokens = 1000
    elif content_len < 1500:
        dynamic_max_tokens = 2000
    elif content_len < 3000:
        dynamic_max_tokens = 3000
    else:
        dynamic_max_tokens = 4000

    for attempt in range(1, max_retries + 1):
        try:
            response = llm_client.generate(prompt=prompt, system_prompt="You are a JSON exporter. Output raw JSON array only.", temperature=0.1, max_tokens=dynamic_max_tokens)
            
            raw_json = None
            try:
                raw_json = json.loads(response)
            except json.JSONDecodeError:
                match = re.search(r'\[.*\]', response, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    try:
                        raw_json = json.loads(json_str)
                    except json.JSONDecodeError:
                        last_bracket = json_str.rfind(']')
                        if last_bracket != -1:
                            salvaged = json_str[:last_bracket+1] + ']'
                            try:
                                raw_json = json.loads(salvaged)
                            except json.JSONDecodeError:
                                pass
            
            if raw_json is not None:
                parsed_triples = []
                for row in raw_json:
                    if isinstance(row, list) and len(row) >= 5:
                        conf = 1.0
                        if len(row) > 5:
                            try:
                                conf = float(row[5])
                            except (ValueError, TypeError):
                                conf = 1.0
                                
                        parsed_triples.append({
                            "subject_label": str(row[0]),
                            "subject_name": str(row[1]),
                            "relation": str(row[2]),
                            "object_label": str(row[3]),
                            "object_name": str(row[4]),
                            "confidence": conf
                        })
                    elif isinstance(row, dict):
                        parsed_triples.append(row)
                return parsed_triples
                
            raise ValueError("No JSON array found in response")
        except Exception as e:
            wait_time = attempt * 3
            logger.warning(f"LLM extraction failed (attempt {attempt}/{max_retries}): {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            
    logger.error("Max retries reached for semantic extraction.")
    return []


def canonicalize(text: str) -> str:
    """Normalize semantic node names via alias map."""
    if not text: return ""
    text = str(text).strip().lower()
    text = re.sub(r'\s+', ' ', text)
    return ALIAS_MAP.get(text, text)


def _ingest_semantics_tx(tx, batch: list[dict], article_ids: list[str]):
    """Ingest semantic triples securely inside a transaction with bidirectional provenance."""
    for row in batch:
        sub_label = row.get("subject_label")
        sub_name = canonicalize(row.get("subject_name"))
        rel = str(row.get("relation", "")).strip().upper()
        obj_label = row.get("object_label")
        obj_name = canonicalize(row.get("object_name"))
        article_id = row.get("article_id")
        conf = row.get("confidence", 1.0)
        
        valid_labels = set(SEMANTIC_LABELS).union({"LegalArticle"})
        if sub_label not in valid_labels or obj_label not in valid_labels or rel not in VALID_RELS:
            continue
        if obj_label == "LegalArticle":
            continue
        if not sub_name or not obj_name:
            continue
            
        if sub_label == "LegalArticle" and sub_name == "article_node":
            cypher = f"""
            MATCH (a:LegalArticle {{article_id: $article_id}})
            MERGE (o:{obj_label} {{name: $obj_name}})
            ON CREATE SET o.aliases = [$raw_obj_name]
            MERGE (a)-[r:{rel}]->(o)
            SET r.confidence = $conf
            MERGE (a)-[:MENTIONS]->(o)
            """
            tx.run(cypher, article_id=article_id, obj_name=obj_name, raw_obj_name=row.get("object_name"), conf=conf)
        else:
            cypher = f"""
            MERGE (s:{sub_label} {{name: $sub_name}})
            ON CREATE SET s.aliases = [$raw_sub_name]
            MERGE (o:{obj_label} {{name: $obj_name}})
            ON CREATE SET o.aliases = [$raw_obj_name]
            MERGE (s)-[r:{rel}]->(o)
            SET r.confidence = $conf
            """
            tx.run(cypher, sub_name=sub_name, raw_sub_name=row.get("subject_name"), obj_name=obj_name, raw_obj_name=row.get("object_name"), conf=conf)
            
            # Bidirectional provenance
            cypher_link = f"""
            MATCH (a:LegalArticle {{article_id: $article_id}})
            MATCH (s:{sub_label} {{name: $sub_name}})
            MATCH (o:{obj_label} {{name: $obj_name}})
            MERGE (a)-[:MENTIONS]->(s)
            MERGE (a)-[:MENTIONS]->(o)
            """
            tx.run(cypher_link, article_id=article_id, sub_name=sub_name, obj_name=obj_name)

    # Checkpointing
    if article_ids:
        tx.run(
            "UNWIND $ids AS aid MATCH (a:LegalArticle {article_id: aid}) SET a.semantic_processed = true",
            ids=article_ids
        )


def ingest_semantics_batch(client: Neo4jClient, batch: list[dict], article_ids: list[str]):
    """Execute ingestion transaction."""
    with client._get_driver().session(database=client.database) as session:
        session.execute_write(_ingest_semantics_tx, batch, article_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--batch-size", type=int, default=10, help="Number of articles per batch")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent threads for LLM extraction")
    parser.add_argument("--max-batches", type=int, default=0, help="Max batches to process (0 = infinite)")
    parser.add_argument("--pilot", action="store_true", help="Run in pilot mode (generates full reports at end)")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Force strict isolation to 'phapdien' database
    neo_cfg = dict(cfg["neo4j"])
    neo_cfg["database"] = TARGET_DATABASE

    client = Neo4jClient(neo_cfg)
    llm_client = OpenAIClient(cfg.get("openai", cfg.get("ollama", {})))
    stats = SemanticStats()
    audit_samples = []

    try:
        if not client.verify_connection():
            logger.error(f"Cannot connect to Neo4j database '{TARGET_DATABASE}'.")
            sys.exit(1)

        create_semantic_schema(client)
        batches_processed = 0

        logger.info(f"Starting Layer 2 Semantic Ingestion on database '{TARGET_DATABASE}'...")

        while True:
            if args.max_batches > 0 and batches_processed >= args.max_batches:
                logger.info(f"Reached max batches ({args.max_batches}). Stopping.")
                break

            articles = get_unprocessed_batch(client, limit=args.batch_size)
            if not articles:
                logger.info("No unprocessed LegalArticles found. Ingestion complete!")
                break

            logger.info(f"Processing batch {batches_processed + 1} ({len(articles)} articles)...")
            
            semantic_batch = []
            processed_article_ids = []
            
            # Use ThreadPoolExecutor to process articles in the batch concurrently
            def process_article(art):
                content = art.get("content", "").strip()
                if not content or len(content) < 20:
                    return art["article_id"], []
                return art["article_id"], extract_semantics_with_retry(llm_client, content)

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                results = list(executor.map(process_article, articles))

            for art_id, triples in results:
                processed_article_ids.append(art_id)
                stats.articles_processed += 1
                
                for t in triples:
                    stats.triples_generated += 1
                    try:
                        conf = float(t.get("confidence", 1.0))
                    except (ValueError, TypeError):
                        conf = 0.0
                    
                    if conf < 0.75:
                        stats.triples_filtered += 1
                        continue
                        
                    t["article_id"] = art_id
                    semantic_batch.append(t)
                    stats.triples_kept += 1
                    stats.add_kept_triple(
                        t.get("subject_label"), canonicalize(t.get("subject_name")),
                        t.get("relation"),
                        t.get("object_label"), canonicalize(t.get("object_name"))
                    )
                    
                    # Reservoir sampling for audit
                    if len(audit_samples) < 20:
                        audit_samples.append({
                            "article_id": art_id, 
                            "article_text": "Sampled text", 
                            "relation": t.get("relation"),
                            "confidence": conf,
                            "triple": t
                        })
                    else:
                        j = random.randint(0, stats.triples_kept - 1)
                        if j < 20:
                            audit_samples[j] = {
                                "article_id": art_id, 
                                "article_text": "Sampled text", 
                                "relation": t.get("relation"),
                                "confidence": conf,
                                "triple": t
                            }
                
            if processed_article_ids:
                ingest_semantics_batch(client, semantic_batch, processed_article_ids)
                
            batches_processed += 1
            stats.print_report()
            
        # Export stats & audit if pilot
        if args.pilot:
            logger.info("Generating post-pilot audit reports...")
            
            logs_dir = ROOT / "eval_logs"
            logs_dir.mkdir(exist_ok=True)
            
            stats.export(logs_dir / "pilot_report.json")
            
            with open(logs_dir / "top_relations.json", "w", encoding="utf-8") as f:
                json.dump(dict(stats.relation_counts.most_common()), f, ensure_ascii=False, indent=2)
                
            with client._get_driver().session(database=client.database) as session:
                degree_query = """
                MATCH (c:LEGAL_CONCEPT)
                WITH c, count{(c)--()} as degree
                ORDER BY degree DESC
                LIMIT 50
                RETURN c.name AS concept, degree
                """
                top_concepts_degree = session.run(degree_query).data()
                
                label_query = """
                MATCH (n)
                WHERE ANY(l IN labels(n) WHERE l IN $semantic_labels)
                RETURN labels(n)[0] AS label, count(n) AS count
                """
                label_dist_records = session.run(label_query, semantic_labels=SEMANTIC_LABELS).data()
                label_distribution = {r["label"]: r["count"] for r in label_dist_records}
                
                readiness_query = """
                MATCH (s)-[r]->(o)
                WHERE labels(s)[0] IN $semantic_labels
                RETURN
                labels(s)[0] as subject_type,
                s.name as subject_name,
                type(r) as relation,
                labels(o)[0] as object_type,
                o.name as object_name
                LIMIT 50
                """
                readiness_samples = session.run(readiness_query, semantic_labels=SEMANTIC_LABELS).data()
                
            with open(logs_dir / "top_concepts.json", "w", encoding="utf-8") as f:
                json.dump(top_concepts_degree, f, ensure_ascii=False, indent=2)
                
            with open(logs_dir / "top_actors.json", "w", encoding="utf-8") as f:
                json.dump(dict(stats.actor_counts.most_common(50)), f, ensure_ascii=False, indent=2)

            with open(logs_dir / "semantic_audit_samples.json", "w", encoding="utf-8") as f:
                json.dump(audit_samples, f, ensure_ascii=False, indent=2)
                
            with open(logs_dir / "semantic_label_distribution.json", "w", encoding="utf-8") as f:
                json.dump(label_distribution, f, ensure_ascii=False, indent=2)
                
            with open(logs_dir / "retrieval_readiness_samples.json", "w", encoding="utf-8") as f:
                json.dump(readiness_samples, f, ensure_ascii=False, indent=2)
                
            logger.info(f"7 Pilot Audit reports exported to {logs_dir}/")

    except KeyboardInterrupt:
        logger.info("Ingestion interrupted by user. Safe to resume later (Checkpointing is active).")
    except Exception as e:
        logger.error(f"Critical error during ingestion: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    main()
