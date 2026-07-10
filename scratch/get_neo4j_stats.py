from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "dinhhthanhh"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def run_query(query):
    with driver.session() as session:
        result = session.run(query)
        return [record.data() for record in result]

# Edge types
edge_types = run_query("MATCH ()-[r]->() RETURN type(r) as type, count(r) as count ORDER BY count DESC")
print("Edge Types:")
for et in edge_types:
    print(f"  {et['type']}: {et['count']}")

# Avg degree
degree = run_query("MATCH (n) RETURN avg(COUNT { (n)--() }) as avg_degree")[0]['avg_degree']
print(f"\nAverage Degree: {degree:.2f}")

# Isolated nodes
isolated = run_query("MATCH (n) WHERE NOT (n)--() RETURN count(n) as count")[0]['count']
print(f"Isolated Nodes: {isolated}")

driver.close()
