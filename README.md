# pdf-rag-pinecone

A minimal **PDF → Pinecone RAG pipeline** for semantic search.  
This project allows you to ingest PDFs, create embeddings using OpenAI models, store them in Pinecone, and perform Retrieval-Augmented Generation (RAG) queries.

---

## Requirements

- Python **3.12.x**
- Pinecone account and API key
- OpenAI API key

---

## Setup

1. **Create a virtual environment**:

```bash
python -m venv .venv
source .venv/bin/activate
.venv\Scripts\activate
pip install -r requirements.txt

OR install manually after activating the virtual environment:
pip install pinecone openai langchain-community pypdf python-dotenv


