import yaml
from graph.neo4j_client import Neo4jClient

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

client = Neo4jClient(config["neo4j"])
res = client.query("CALL db.schema.visualization()")
print(res)
