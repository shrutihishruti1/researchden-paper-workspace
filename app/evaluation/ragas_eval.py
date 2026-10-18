# app/evaluation/ragas_eval.py
# PRIMARY OWNER: Shruti
# STATUS: Active — RAGAS faithfulness + answer relevancy evaluation
# NOTE: Switched LLM judge from Gemini to Groq (llama-3.1-8b-instant)
# Reason: Gemini free tier exhausted (20 req/day); Groq = 14,400/day free

import os
from dotenv import load_dotenv, find_dotenv
from datasets import Dataset
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas import evaluate, RunConfig
#from langchain_groq import ChatGroq
#from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
# --- NEW: Ollama (local, no API key, no quota) ---
from langchain_ollama import ChatOllama


# --- OLD GEMINI IMPORT ---
# from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(find_dotenv())

# WHY strictness=1: RAGAS defaults to generating 3 candidate questions internally.
# Free-tier LLMs (including Groq free) don't support n>1 candidates — causes error.




# --- OLD ---
# from langchain_mistralai import ChatMistralAI

def get_llm():
    return ChatOllama(model="llama3.1:8b", temperature=0)
  
    # --- OLD: Gemini as RAGAS judge ---
    # return ChatGoogleGenerativeAI(model="gemini-3.8-flash")


def get_embeddings():
    # Local HuggingFace model — no API key, no quota, no availability issues.
    # Downloads ~90MB once, then cached locally.
    # WHY not Google embeddings: models/embedding-001 and text-embedding-004
    # both return 404 via the v1beta API endpoint used by langchain-google-genai.
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# def evaluate_faithfulness(question: str, answer: str, contexts: list[str]) -> float:
#     # Scores how much the answer is supported by the retrieved chunks.
#     # RAGAS breaks the answer into individual claims, checks each one
#     #      against the retrieved passages, returns fraction that are supported.
#     # RANGE: 0.0 (fully hallucinated) to 1.0 (fully grounded)

#     # KEY FIX: Must set .llm directly on the metric object.
#     # Passing llm= to evaluate() is IGNORED by RAGAS for these metric objects.
#     ragas_llm = LangchainLLMWrapper(get_llm())
#     faithfulness.llm = ragas_llm

#     sample = {
#         "question": [question],
#         "answer": [answer],
#         "contexts": [contexts],
#         "ground_truth": [""]   # not needed for faithfulness but RAGAS expects the key
#     }
#     dataset = Dataset.from_dict(sample)
#     result = evaluate(dataset, metrics=[faithfulness], run_config=RunConfig(timeout=1200))
#     return float(result["faithfulness"][0])

def evaluate_faithfulness(question: str, answer: str, contexts: list[str]) -> float:
    ragas_llm = LangchainLLMWrapper(get_llm())
    metric = Faithfulness(llm=ragas_llm)
    sample = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [""]
    }
    dataset = Dataset.from_dict(sample)
    result = evaluate(dataset, metrics=[metric], run_config=RunConfig(timeout=1200))
    return float(result["faithfulness"][0])

# def evaluate_answer_relevancy(question: str, answer: str, contexts: list[str]) -> float:
#     # Scores how directly the answer addresses the question asked.
#     # RAGAS generates a reverse-engineered question from the answer,then measures embedding similarity between that and the original question.
#     # RANGE: 0.0 (off-topic) to 1.0 (directly on-topic)


#     ragas_llm = LangchainLLMWrapper(get_llm())
#     ragas_emb = LangchainEmbeddingsWrapper(get_embeddings())
#     answer_relevancy.llm = ragas_llm
#     answer_relevancy.embeddings = ragas_emb

#     sample = {
#         "question": [question],
#         "answer": [answer],
#         "contexts": [contexts],
#         "ground_truth": [""]
#     }
#     dataset = Dataset.from_dict(sample)
#     #old
#     result = evaluate(dataset, metrics=[answer_relevancy], run_config=RunConfig(timeout=1200))
#     return float(result["answer_relevancy"][0])

def evaluate_answer_relevancy(question: str, answer: str, contexts: list[str]) -> float:
    ragas_llm = LangchainLLMWrapper(get_llm())
    ragas_emb = LangchainEmbeddingsWrapper(get_embeddings())
    metric = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb)
    metric.strictness = 1
    sample = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [""]
    }
    dataset = Dataset.from_dict(sample)
    result = evaluate(dataset, metrics=[metric], run_config=RunConfig(timeout=1200))
    return float(result["answer_relevancy"][0])

if __name__ == "__main__":
    from app.pipeline.retrieve import retrieve
    from app.pipeline.generate import generate_answer

    # --- OLD TEST DATA (amygdala — kept for reference) ---
    # score1 = evaluate_faithfulness(question="What does the amygdala do?", ...)

    test_questions = [
        "What is RAGAS and what does it evaluate?",
        # "How is faithfulness measured in RAGAS?",     # commented to save quota
        # "What datasets were used to evaluate RAGAS?"  # commented to save quota
    ]

    print("=" * 60)
    print("RAGAS PIPELINE EVALUATION")
    print("=" * 60)

    for question in test_questions:
        print(f"\nQuestion: {question}")
        print("-" * 60)

        # Step 1: Retrieve — ChromaDB finds the 3 most relevant chunks
        chunks = retrieve(question, n_results=3)
        print(f"Retrieved {len(chunks)} chunks")

        # Step 2: Generate — Groq reads those chunks and writes an answer
        # --- OLD: Gemini reads those chunks ---
        answer = generate_answer(question, chunks)
        print(f"Answer: {answer[:200]}...")

        # Step 3: Evaluate — RAGAS judges both scores
        # Each evaluate() call makes ~4-6 internal Groq API calls.
        # With 14,400/day free quota, this is no longer a bottleneck.
        f_score = evaluate_faithfulness(question, answer, chunks)
        print(f"Faithfulness:     {f_score:.2f}")

        r_score = evaluate_answer_relevancy(question, answer, chunks)
        print(f"Answer Relevancy: {r_score:.2f}")

    print("\n" + "=" * 60)
    print("Evaluation complete.")