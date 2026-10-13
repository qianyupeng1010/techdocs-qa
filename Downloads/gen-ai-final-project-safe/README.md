---
title: Technical Documentation Q&A
emoji: 📖
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
python_version: "3.11"
pinned: false
---

# Technical Documentation Q&A

A conversational RAG system for querying technical documentation PDFs, with built-in RAGAS quality evaluation.

## Features

- Upload API docs, framework manuals, or any technical PDFs
- Multi-turn conversation with history-aware retrieval (follow-up questions work correctly)
- RAGAS evaluation panel per answer: Faithfulness, Answer Relevancy, Context Precision
- Source chunk inspection with page-level citations
- Automatic conversation reset when the document set changes

## Tech Stack

- Python 3.11
- Streamlit
- LangChain (`create_history_aware_retriever` + `create_retrieval_chain`)
- FAISS (in-memory vector store)
- OpenAI embeddings (`text-embedding-3-small`) and chat (`gpt-4o-mini`)
- RAGAS 0.2 for answer quality evaluation

## Run Locally

1. Create a virtual environment:

```bash
uv venv --python 3.11 .venv
```

2. Install dependencies:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=your_key_here
```

4. Run the app:

```bash
.venv/bin/streamlit run app.py
```

## Run with Docker

```bash
docker build -t techdocs-qa .
docker run -p 7860:7860 -e OPENAI_API_KEY=your_key_here techdocs-qa
```

Then open `http://localhost:7860`.

## Deploy to Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space), select **Docker** as the SDK.
2. Push this repository to the Space:

```bash
git remote add space https://huggingface.co/spaces/<your-username>/<your-space-name>
git push space main
```

3. Go to **Settings → Variables and secrets** in your Space and add:
   - `OPENAI_API_KEY` as a secret (never commit this value)
   - Optionally `OPENAI_CHAT_MODEL` and `OPENAI_EMBEDDING_MODEL` to override defaults

The Space builds the Docker image automatically on each push. Build logs are visible in the Space UI.

## How to Use

1. Upload one or more documentation PDFs using the file uploader.
2. Ask a technical question in the chat input.
3. Inspect the retrieved source chunks under each answer.
4. Click **Evaluate this answer** to run RAGAS and see quality scores.

## RAGAS Metrics

| Metric | What it measures |
|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? 1.0 = no hallucination |
| **Answer Relevancy** | Does the answer address the question? 1.0 = fully on-topic |
| **Context Precision** | Are the retrieved chunks relevant? 1.0 = all chunks were useful |
