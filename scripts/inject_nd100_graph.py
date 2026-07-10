import yaml
import logging
from graph.neo4j_client import Neo4jClient

logging.basicConfig(level=logging.INFO)

with open("configs/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

client = Neo4jClient(config["neo4j"])

# Inject nodes
nodes = [
    {
        "name": "Nghị định 100/2019/NĐ-CP",
        "type": "LegalDoc",
        "properties": {
            "title": "Nghị định 100/2019/NĐ-CP",
            "doc_id": "100/2019/NĐ-CP",
            "type": "Nghị định",
            "authority": "Chính phủ",
            "source": "Mock"
        }
    },
    {
        "name": "Điều 23_100/2019/NĐ-CP",
        "type": "LegalArticle",
        "properties": {
            "title": "Điều 23. Xử phạt người điều khiển xe ô tô chở hành khách, ô tô chở người vi phạm quy định về vận tải đường bộ",
            "article_id": "Điều 23",
            "doc_id": "100/2019/NĐ-CP",
            "content": "Phạt tiền từ 400.000 đồng đến 600.000 đồng trên mỗi người vượt quá quy định đối với người điều khiển xe ô tô chở hành khách. Nếu người lái xe không nộp phạt, thủ tục cưỡng chế hoặc xử lý tiếp theo được hướng dẫn tại Luật Xử lý vi phạm hành chính."
        }
    },
    {
        "name": "Luật Xử lý vi phạm hành chính",
        "type": "LegalDoc",
        "properties": {
            "title": "Luật Xử lý vi phạm hành chính",
            "doc_id": "15/2012/QH13",
            "type": "Luật",
            "authority": "Quốc hội"
        }
    },
    {
        "name": "Điều 86_15/2012/QH13",
        "type": "LegalArticle",
        "properties": {
            "title": "Điều 86. Cưỡng chế thi hành quyết định xử phạt vi phạm hành chính",
            "article_id": "Điều 86",
            "doc_id": "15/2012/QH13",
            "content": "Cưỡng chế thi hành quyết định xử phạt được áp dụng trong trường hợp cá nhân, tổ chức bị xử phạt vi phạm hành chính không tự nguyện chấp hành quyết định xử phạt theo quy định tại Điều 73 của Luật này."
        }
    },
    {
        "name": "xe ô tô chở khách",
        "type": "Concept",
        "properties": {}
    },
    {
        "name": "không nộp phạt",
        "type": "Concept",
        "properties": {}
    },
    {
        "name": "thủ tục xử lý",
        "type": "Concept",
        "properties": {}
    }
]

# Write nodes using Cypher
with client._get_driver().session(database=client.database) as session:
    for node in nodes:
        label = node["type"]
        props = node["properties"]
        props["name"] = node["name"]
        
        # Build set string
        set_str = ", ".join([f"n.`{k}` = ${k}" for k in props.keys()])
        
        query = f"""
        MERGE (n:{label} {{name: $name}})
        SET {set_str}
        """
        session.run(query, **props)

# Inject relations
relations = [
    ("Điều 23_100/2019/NĐ-CP", "PART_OF", "Nghị định 100/2019/NĐ-CP"),
    ("Điều 86_15/2012/QH13", "PART_OF", "Luật Xử lý vi phạm hành chính"),
    ("Điều 23_100/2019/NĐ-CP", "REFERENCES", "Luật Xử lý vi phạm hành chính"),
    ("xe ô tô chở khách", "MENTIONED_IN", "Điều 23_100/2019/NĐ-CP"),
    ("không nộp phạt", "MENTIONED_IN", "Điều 23_100/2019/NĐ-CP"),
    ("thủ tục xử lý", "MENTIONED_IN", "Luật Xử lý vi phạm hành chính"),
    ("không nộp phạt", "MENTIONED_IN", "Điều 86_15/2012/QH13")
]

with client._get_driver().session(database=client.database) as session:
    for src, rel, tgt in relations:
        query = f"""
        MATCH (a {{name: $src}})
        MATCH (b {{name: $tgt}})
        MERGE (a)-[r:{rel}]->(b)
        """
        session.run(query, src=src, tgt=tgt)

print("Graph mock data injected!")
client.close()
