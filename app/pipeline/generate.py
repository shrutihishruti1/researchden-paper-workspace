# app/pipeline/generate.py
# PRIMARY OWNER: Shruti
# STATUS: Active — LLM generation step of RAG pipeline

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=GOOGLE_API_KEY)

MODEL_NAME = "gemini-3.8-flash"


def generate_answer(question: str, context_chunks: list[str]) -> str:
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

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    return response.text


if __name__ == "__main__":
    from app.pipeline.retrieve import retrieve

    test_question = "What is RAGAS and what does it evaluate?"

    print(f"Question: {test_question}")
    print("-" * 60)
    print("Retrieving relevant chunks...")
    chunks = retrieve(test_question, n_results=3)
    print(f"Retrieved {len(chunks)} chunks\n")
    print("Asking Gemini...")
    answer = generate_answer(test_question, chunks)
    print(f"\nGemini's Answer:\n{answer}")