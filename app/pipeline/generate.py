# app/pipeline/generate.py
# PRIMARY OWNER: Shruti
# STATUS: Active — LLM generation step of RAG pipeline
# NOTE: Switched from Gemini (google-genai) to Groq (llama-3.1-8b-instant)
#       Reason: Gemini free tier = 20 requests/day; Groq free tier = 14,400/day

import os
from dotenv import load_dotenv
#from langchain_groq import ChatGroq
#from langchain_mistralai import ChatMistralAI

# --- REMOVED (Gemini dependencies, no longer needed) ---
# from google import genai
# import time
# --- NEW: Ollama (local, no API key, no quota) ---
from langchain_ollama import ChatOllama

# --- OLD: Mistral ---
# from langchain_mistralai import ChatMistralAI

load_dotenv()

#llm = ChatOllama(model="llama3.2")
llm = ChatOllama(model="llama3.1:8b", temperature=0)

# --- OLD ---
# llm = ChatMistralAI(model="mistral-small-latest", api_key=os.getenv("MISTRAL_API_KEY"))


# --- OLD GEMINI SETUP ---
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# client = genai.Client(api_key=GOOGLE_API_KEY)
# MODEL_NAME = "gemini-3.8-flash"


def generate_answer(question: str, context_chunks: list[str]) -> str:
    # WHAT: Builds a prompt from retrieved chunks, sends it to Groq, returns the answer.
    # WHY: "Use ONLY the passages below" forces the model to stay grounded in
    #      our retrieved text — this is what faithfulness measures.

    context_text = ""
    for i, chunk in enumerate(context_chunks):
        context_text += f"Passage {i+1}:\n{chunk}\n\n"

    prompt = f"""You are a research assistant answering questions about academic papers.
Use ONLY the passages below to answer the question.
Do not use any outside knowledge.
If the answer is not in the passages, say "The provided passages do not contain enough information to answer this question."

{context_text}
Question: {question}

Answer:"""

    # --- NEW: Groq via LangChain ---
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"  Ollama error: {e}")
        return "LLM is temporarily unavailable. Please try again in a moment."

    # --- OLD: Gemini with retry logic ---
    # for attempt in range(3):
    #     try:
    #         response = client.models.generate_content(
    #             model=MODEL_NAME,
    #             contents=prompt
    #         )
    #         return response.text
    #     except Exception as e:
    #         if attempt < 2:
    #             print(f"  Attempt {attempt+1} failed: {e}. Retrying in 3s...")
    #             time.sleep(3)
    #         else:
    #             print(f"  All retries exhausted. Last error: {e}")
    #             return "Gemini is temporarily unavailable. Please try again in a moment."


if __name__ == "__main__":
    from app.pipeline.retrieve import retrieve

    test_question = "What is RAGAS and what does it evaluate?"

    print(f"Question: {test_question}")
    print("-" * 60)
    print("Retrieving relevant chunks...")
    chunks = retrieve(test_question, n_results=3)
    print(f"Retrieved {len(chunks)} chunks\n")

    # --- NEW ---
    print("Asking Ollama...")
    # --- OLD ---
    # print("Asking Gemini...")

    answer = generate_answer(test_question, chunks)
    print(f"\nOllama's Answer:\n{answer}")
    # --- OLD ---
    # print(f"\nGemini's Answer:\n{answer}")