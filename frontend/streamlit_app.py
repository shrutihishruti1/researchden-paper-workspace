# frontend/streamlit_app.py
# PRIMARY OWNER: Shruti
# STATUS: Active — Streamlit web interface for RAG pipeline

import streamlit as st
from app.pipeline.retrieve import retrieve
from app.pipeline.generate import generate_answer
from app.evaluation.ragas_eval import evaluate_faithfulness, evaluate_answer_relevancy

st.set_page_config(page_title="ResearchDen RAG", page_icon="📚", layout="centered")

st.title("ResearchDen Research Assistant")
st.markdown("Ask a question about the uploaded research papers.")

question = st.text_input("Your question", placeholder="e.g. What is RAGAS and what does it evaluate?")

run_eval = st.checkbox("Run RAGAS evaluation (slow — ~15 min on CPU)", value=False)

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving relevant passages..."):
        chunks = retrieve(question, n_results=3)

    st.subheader("Retrieved Passages")
    for i, chunk in enumerate(chunks):
        with st.expander(f"Passage {i+1}"):
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