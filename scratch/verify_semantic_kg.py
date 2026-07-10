from neo4j import GraphDatabase
import os
import yaml

with open('configs/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
neo_cfg = config['neo4j']
uri = neo_cfg.get('uri', 'bolt://localhost:7687')
user = neo_cfg.get('user', 'neo4j')
password = neo_cfg.get('password', 'password')

driver = GraphDatabase.driver(uri, auth=(user, password))

print("=== SCHEMA: phapdien ===")
try:
    with driver.session(database="phapdien") as session:
        result = session.run("CALL db.labels()")
        labels = [r[0] for r in result]
        
        print("\nNode Counts:")
        for label in labels:
            if label not in ["LegalDoc", "LegalArticle", "PD"]:
                res = session.run(f"MATCH (n:`{label}`) RETURN count(n) as c")
                print(f"  {label}: {res.single()[0]}")
            
        result = session.run("CALL db.relationshipTypes()")
        rel_types = [r[0] for r in result]
        
        print("\nRelation Counts:")
        for rel in rel_types:
            if rel != "HAS_ARTICLE":
                res = session.run(f"MATCH ()-[r:`{rel}`]->() RETURN count(r) as c")
                print(f"  {rel}: {res.single()[0]}")
                
        print("\n=== SAMPLE TRAVERSAL ===")
        sample_query = """
        MATCH (a:LegalArticle:PD)-[r1]->(concept)
        WHERE NOT labels(concept) = ['LegalArticle'] AND NOT labels(concept) = ['PD']
        RETURN a.article_id AS article, type(r1) AS relation, labels(concept)[0] AS type, concept.name AS name
        LIMIT 5
        """
        res = session.run(sample_query)
        for r in res:
            print(f"{r['article']} -[{r['relation']}]-> ({r['type']}: {r['name']})")
except Exception as e:
    print(f"Failed to inspect schema: {e}")

driver.close()
