import chromadb
client = chromadb.PersistentClient(path="data/vector_store/chroma_full")
try:
    collection = client.get_collection(name="phapdien_full")
    print("Vector chunks:", collection.count())
except Exception as e:
    print("Error:", e)
