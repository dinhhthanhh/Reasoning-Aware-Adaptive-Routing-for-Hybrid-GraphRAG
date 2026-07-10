from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "dinhhthanhh"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def run_query(query):
    with driver.session() as session:
        result = session.run(query)
        return [record.data() for record in result]

print("LegalDoc:", run_query("MATCH (n:LegalDoc) RETURN count(n) as count")[0]['count'])
print("LegalArticle:", run_query("MATCH (n:LegalArticle) RETURN count(n) as count")[0]['count'])
print("LegalConcept:", run_query("MATCH (n:LegalConcept) RETURN count(n) as count")[0]['count'])

# check if there are other edge types at all
print("Edge Types count:")
print(run_query("MATCH ()-[r]->() RETURN type(r) as type, count(r) as count"))

driver.close()
