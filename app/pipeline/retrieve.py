# app/pipeline/retrieve.py
# PRIMARY OWNER: Talib
# FALLBACK AUTHOR: Shruti (built independently to agreed interface contract)
# STATUS: Fallback active — replace with Talib's version when delivered

import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def retrieve(question: str, n_results: int = 3,
             collection_name: str = "research_papers") -> list[str]:
    client = chromadb.PersistentClient(path="data/chroma_db")
    collection = client.get_collection(name=collection_name)
    model = _get_model()
    query_vector = model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results
    )
    return results["documents"][0]

if __name__ == "__main__":
    test_questions = [
        "What is RAGAS?",
        "How is faithfulness measured?",
        "What datasets were used for evaluation?"
    ]
    for question in test_questions:
        print(f"\nQuestion: {question}")
        print("-" * 50)
        chunks = retrieve(question, n_results=2)
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}: {chunk[:200]}")