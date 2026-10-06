import chromadb
client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_collection("research_papers")
print("Total chunks stored:", collection.count())
results = collection.query(query_texts=["what is RAGAS"], n_results=2)
print()
for i, doc in enumerate(results["documents"][0]):
    print(f"--- Result {i+1} ---")
    print(doc[:200])
    print()