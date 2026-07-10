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

print("=== DATABASES ===")
try:
    with driver.session(database="system") as session:
        result = session.run("SHOW DATABASES")
        for record in result:
            print(f"- {record['name']} (status: {record.get('currentStatus', 'unknown')})")
except Exception as e:
    print(f"Failed to show databases (likely Community Edition): {e}")

print("\n=== SCHEMA: phapdien ===")
try:
    with driver.session(database="phapdien") as session:
        result = session.run("CALL db.labels()")
        labels = [r[0] for r in result]
        print(f"Labels: {labels}")
        
        print("\nNode Counts:")
        for label in labels:
            res = session.run(f"MATCH (n:`{label}`) RETURN count(n) as c")
            print(f"  {label}: {res.single()[0]}")
            
        result = session.run("CALL db.relationshipTypes()")
        rel_types = [r[0] for r in result]
        print(f"\nRelationship Types: {rel_types}")
        
        print("\nRelation Counts:")
        for rel in rel_types:
            res = session.run(f"MATCH ()-[r:`{rel}`]->() RETURN count(r) as c")
            print(f"  {rel}: {res.single()[0]}")
except Exception as e:
    print(f"Failed to inspect schema: {e}")

driver.close()
