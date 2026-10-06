# app/pipeline/embed.py
# PRIMARY OWNER: Shruti
# STATUS: Active — core pipeline component

import chromadb
from sentence_transformers import SentenceTransformer

def embed_and_store(chunks: list[str], collection_name: str = "research_papers") -> chromadb.Collection:
    """
    Converts text chunks into vector embeddings and stores them in ChromaDB.

    Args:
        chunks:          List of text strings from chunk_text()
        collection_name: Name of the ChromaDB collection (default: "research_papers")
    Returns:
        ChromaDB Collection object — ready for retrieval queries
    """

    # Load the sentence-transformer embedding model

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Convert every chunk into a vector embedding

    embeddings = model.encode(chunks, show_progress_bar=True)

    # Create a persistent ChromaDB client

    client = chromadb.PersistentClient(path="data/chroma_db")

    # Create or retrieve a named collection inside ChromaDB
  
    collection = client.get_or_create_collection(name=collection_name)

    # Build unique IDs for each chunk
 
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    # Add chunks + embeddings into the collection
   
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=chunks
    )

    return collection


if __name__ == "__main__":
    from app.pipeline.ingest import ingest_pdf
    from app.pipeline.chunk import chunk_text

    file_path = "data/Es et al. - 2025 - Ragas Automated Evaluation of Retrieval Augmented Generation.pdf"

    print("Step 1: Ingesting PDF...")
    full_text = ingest_pdf(file_path)

    print("Step 2: Chunking text...")
    chunks = chunk_text(full_text)
    print(f"  → {len(chunks)} chunks ready")

    print("Step 3: Embedding and storing in ChromaDB...")
    collection = embed_and_store(chunks)

    print(f"\n✅ Done. ChromaDB now holds {collection.count()} embedded chunks.")
    print("Stored at: data/chroma_db/")