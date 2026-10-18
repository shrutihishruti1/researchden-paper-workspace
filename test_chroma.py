import chromadb
client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_collection("research_papers")
print("Total chunks stored:", collection.count())
results = collection.query(query_texts=["What is RAGAS and what does it evaluate?"], n_results=3)
print()
for i, doc in enumerate(results["documents"][0]):
    print(f"--- Result {i+1} ---")
    print(doc[:200])
    print()