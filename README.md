# Research Paper Workspace

An AI-powered tool for uploading academic PDFs, asking natural-language questions, and receiving grounded answers scored for faithfulness to the source material.

**Built for ResearchDen Fall 2026 Technical Project**  
Team: Shruti, Jamal Hossain, Talib | Supervisor: Sultanul Ovi

---

## What It Does

Upload a research paper, ask a question, and get an answer that is:

- Retrieved from the paper's actual content (not LLM memory)
- Scored for **Faithfulness** — how grounded the answer is in the source
- Scored for **Answer Relevancy** — how well it addresses what was asked

---

## Features

- Upload a PDF and ask natural-language questions about it
- Dense semantic retrieval via ChromaDB vector search
- Answer generation using a locally hosted LLM (no external API required)
- RAGAS evaluation: Faithfulness and Answer Relevancy scores on demand
- Streamlit web interface — no command-line knowledge required

---

## Tech Stack

| Layer                         | Technology                               |
| ----------------------------- | ---------------------------------------- |
| PDF parsing                   | PyMuPDF                                  |
| Text chunking                 | LangChain RecursiveCharacterTextSplitter |
| Embeddings                    | sentence-transformers (all-MiniLM-L6-v2) |
| Vector store                  | ChromaDB                                 |
| LLM (generation + evaluation) | Ollama — llama3.1:8b (local)             |
| Evaluation framework          | RAGAS (faithfulness + answer relevancy)  |
| Frontend                      | Streamlit                                |

---

## Project Structure

```
researchden-paper-workspace/
├── app/
│   ├── main.py                  # App entry point
│   ├── pipeline/
│   │   ├── __init__.py          # process_pdf + vector_store_payload (Jamal)
│   │   ├── ingest.py            # PyMuPDF text extraction (Jamal)
│   │   ├── chunk.py             # Text chunking (Jamal)
│   │   ├── embed.py             # Sentence transformer embeddings (Jamal)
│   │   ├── numeric.py           # Numeric content extraction (Jamal)
│   │   ├── tables.py            # Table extraction (Jamal)
│   │   ├── retrieve.py          # ChromaDB vector retrieval (Shruti)
│   │   └── generate.py          # Ollama answer generation (Shruti)
│   └── evaluation/
│       └── ragas_eval.py        # RAGAS faithfulness + relevancy (Shruti)
├── frontend/
│   └── streamlit_app.py         # Streamlit UI (Shruti)
├── data/
│   └── chroma_db/               # Persistent ChromaDB collection
├── tests/
│   └── test_evaluation.py       # RAGAS component tests (Shruti)
├── requirements.txt
└── README.md
```

---

## Team Contributions

| Contributor   | Component                                                                  |
| ------------- | -------------------------------------------------------------------------- |
| Shruti        | Retrieval, generation, RAGAS evaluation, Streamlit UI, technical report    |
| Jamal Hossain | PDF ingestion pipeline (ingest, chunk, embed, numeric, tables)             |
| Talib         | Hybrid retrieval design — BM25 + RRF (proposed, PR #3, not yet integrated) |

---

## How to Run Locally

**Prerequisites:** [Ollama](https://ollama.com) installed and running with `llama3.1:8b` pulled.

```bash
# 1. Clone the repo
git clone https://github.com/shrutihishruti1/researchden-paper-workspace.git
cd researchden-paper-workspace

# 2. Create and activate virtual environment
python -m venv research
research\Scripts\activate        # Windows
# source research/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the LLM (first time only)
ollama pull llama3.1:8b

# 5. Start the Streamlit app
python -m streamlit run frontend/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Evaluation Results

Evaluated on the question: _"What is RAGAS and what does it evaluate?"_

| Metric           | Score |
| ---------------- | ----- |
| Faithfulness     | 0.62  |
| Answer Relevancy | 0.78  |

Scores are stable across runs with `temperature=0` on both the generation and judge LLMs.

---

## Status

✅ Submitted — ResearchDen Fall 2026 Internship, October 2026
