import streamlit as st
import chromadb
from app.pipeline import process_pdf, vector_store_payload
from app.pipeline.retrieve import retrieve
from app.pipeline.generate import generate_answer
from app.evaluation.ragas_eval import evaluate_faithfulness, evaluate_answer_relevancy
from app.pipeline.embed import EmbeddingService


if "embedding_service" not in st.session_state:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.session_state.embedding_service = EmbeddingService(device=device)

# Persist the ingestion success message across reruns
if "ingestion_success" not in st.session_state:
    st.session_state.ingestion_success = None

st.set_page_config(page_title="ResearchDen RAG", page_icon="📚", layout="centered")
st.title("ResearchDen Research Assistant")
st.markdown("Ask a question about the uploaded research papers.")

# ---------------------------------------------------------------------------
# SECTION 1: PDF Upload (Jamal's ingestion pipeline)
# ---------------------------------------------------------------------------
st.subheader("📄 Upload a Research Paper")
uploaded_file = st.file_uploader(
    "Add a PDF to the knowledge base",
    type=["pdf"],
    help="Uses Jamal's ingestion pipeline: extracts text, detects sections, chunks, and embeds into ChromaDB.",
)

if uploaded_file is not None:
    # Clear the previous success message when a new file is selected
    if st.session_state.ingestion_success and uploaded_file.name not in st.session_state.ingestion_success:
        st.session_state.ingestion_success = None

    if st.button("Ingest PDF"):
        with st.spinner(f"Processing {uploaded_file.name} — extracting, chunking, embedding..."):
            # Step 1: Run Jamal's ingestion pipeline
            result = process_pdf(
                uploaded_file.getvalue(),
                source_filename=uploaded_file.name,
                service=st.session_state.embedding_service,
            )
            # Step 2: Convert to ChromaDB-ready arrays
            payload = vector_store_payload(result)
            # Step 3: Upsert into the persistent ChromaDB collection
            client = chromadb.PersistentClient(path="data/chroma_db")
            collection = client.get_or_create_collection(name="research_papers")
            collection.upsert(
                ids=payload["ids"],
                documents=payload["documents"],
                embeddings=payload["embeddings"],
                metadatas=payload["metadatas"],
            )
            # Store result in session_state so it survives the Streamlit rerun
            st.session_state.ingestion_success = (
                f"✅ **{uploaded_file.name}** ingested — "
                f"{result['page_count']} pages, {len(result['chunks'])} chunks added to knowledge base."
            )

# Show the success message outside the upload block so it persists after rerun
if st.session_state.ingestion_success:
    st.success(st.session_state.ingestion_success)

st.divider()

# ---------------------------------------------------------------------------
# SECTION 2: Question Answering (Shruti's retrieve + generate pipeline)
# ---------------------------------------------------------------------------
st.subheader("💬 Ask a Question")
question = st.text_input(
    "Your question",
    placeholder="e.g. What is RAGAS and what does it evaluate?",
)
run_eval = st.checkbox("Run RAGAS evaluation (slow — ~15 min on CPU)", value=False)

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving relevant passages..."):
        chunks = retrieve(question, n_results=3)

    st.subheader("Retrieved Passages")
    for i, chunk in enumerate(chunks):
        with st.expander(f"Passage {i + 1}"):
            st.write(chunk)

    with st.spinner("Generating answer..."):
        answer = generate_answer(question, chunks)

    st.subheader("Answer")
    st.write(answer)

    if run_eval:
        st.subheader("RAGAS Evaluation")
        with st.spinner("Running faithfulness evaluation (~10–15 min)..."):
            f_score = evaluate_faithfulness(question, answer, chunks)
        st.metric("Faithfulness", f"{f_score:.2f}")
        with st.spinner("Running answer relevancy evaluation (~2 min)..."):
            r_score = evaluate_answer_relevancy(question, answer, chunks)
        st.metric("Answer Relevancy", f"{r_score:.2f}")