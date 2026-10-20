# RAG PDF Chatbot

Ask questions about a PDF and get answers grounded in the document using Retrieval-Augmented Generation (RAG).

## Overview

This project lets you upload a text-based PDF, ask questions about its contents, and view an answer alongside the retrieved passages and their page numbers.

It uses a local embedding model and FAISS for retrieval, then sends the retrieved context to an LLM through OpenRouter.

## Features

- Upload PDF files through a Streamlit interface.
- Extract text while preserving page numbers with `pypdf`.
- Split document text into overlapping chunks.
- Create text embeddings with Sentence Transformers (`all-MiniLM-L6-v2`).
- Retrieve relevant chunks with FAISS similarity search.
- Generate context-grounded answers through the OpenRouter API.
- Display retrieved source passages, page numbers, and similarity scores.
- Handle empty/unreadable PDFs and common processing/API errors.

## Architecture

```text
PDF Upload
    |
    v
Text Extraction (pypdf)
    |
    v
Chunking with Overlap
    |
    v
Embeddings (Sentence Transformers)
    |
    v
Vector Index (FAISS)
    |
User Question
    |
    v
Question Embedding + Similarity Search
    |
    v
Top Relevant PDF Chunks
    |
    v
LLM Answer Generation (OpenRouter)
    |
    v
Answer + Retrieved Sources in Streamlit
```

## Tech Stack

- **Python** — application logic
- **Streamlit** — user interface
- **pypdf** — PDF text extraction
- **Sentence Transformers** — text embeddings
- **FAISS** — vector similarity search
- **OpenAI Python SDK** — OpenAI-compatible client for OpenRouter
- **OpenRouter** — access to supported language models
- **python-dotenv** — local environment variable loading

## Project Structure

```text
RAG PDF chatbot/
├── app.py
├── requirements.txt
├── .env                 # Local secret; do not commit
├── .gitignore
└── README.md
```

## Prerequisites

- Python 3.10 or a compatible Python version supported by your installed dependencies
- A GitHub account
- An OpenRouter API key
- Internet access for the API and for the first download of the embedding model

## Setup and Run

### 1. Clone the repository

After you push the project to GitHub, clone it with:

```bash
git clone https://github.com/Hemraj0808/RAG-PDF-Chatbot.git
cd RAG-PDF-Chatbot
```

Replace the URL with your actual repository URL.

### 2. Create and activate a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configure the API key

Create a `.env` file in the project root:

```text
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Replace the placeholder with your own key. Do not commit `.env` or share your API key.

### 5. Start the application

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal, upload a text-based PDF, enter a question, and select **Get Answer**.

## How RAG Works in This Project

1. **Extract:** Read text from each PDF page and retain its page number.
2. **Chunk:** Split page text into overlapping pieces so relevant passages can be retrieved independently.
3. **Embed:** Convert each chunk into a numeric vector using Sentence Transformers.
4. **Index:** Store vectors in a FAISS inner-product index. The embeddings are normalized, so inner-product ranking corresponds to cosine similarity.
5. **Retrieve:** Embed the user's question and retrieve the top matching chunks.
6. **Generate:** Send the question and retrieved passages to the selected OpenRouter model.
7. **Display:** Show the generated answer and retrieved source passages with page numbers.

## Configuration Notes

The starter application uses the following defaults:

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Chunk size: `800` characters
- Chunk overlap: `120` characters
- Retrieval count: up to `4` chunks
- OpenRouter model: `openrouter/free`

The free-model router and individual free models can have changing availability and rate limits. Check the [OpenRouter models catalogue](https://openrouter.ai/models) if requests fail. API access may require credits depending on the model and account.

## Limitations

- Scanned PDFs may not contain extractable text and may require OCR.
- The index is rebuilt when Streamlit reruns after a PDF is uploaded; caching and persistent indexing could improve performance.
- Retrieval quality depends on the document, chunking, embedding model, and question.
- The LLM may still make mistakes. Verify important answers against the cited pages.
- The current implementation is intended as a learning project, not a hardened production service.

## Security

- Store API keys in `.env`, not in source code.
- Keep `.env` in `.gitignore`.
- Never commit API keys, private documents, or other secrets.
- If a key is accidentally pushed to GitHub, revoke it immediately and create a replacement. Removing it from the latest commit alone may not remove it from Git history.

## Possible Improvements

- Add chat history for follow-up questions.
- Cache extracted text and vector indexes.
- Support multiple PDFs.
- Add OCR for scanned PDFs.
- Add token-usage and latency measurements.
- Add automated tests and more detailed error handling.
- Let the user select an available OpenRouter model.

## License

Choose a license before publishing if you want others to reuse or modify the project. For example, you can add an MIT License by creating a `LICENSE` file and including the appropriate license text.

---

Built as a hands-on learning project to understand Retrieval-Augmented Generation (RAG), embeddings, vector search, and LLM integration.
