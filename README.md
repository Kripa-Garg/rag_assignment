# DocSage 🧠

A Retrieval-Augmented Generation (RAG) app I built that lets you upload a PDF or TXT document, ask a question, and get back the most relevant passages - plus an AI-generated answer powered by Cohere.

## Overview

DocSage combines semantic search with generative AI to answer questions grounded in your own documents. Instead of just keyword matching, it understands the *meaning* of your question and finds the most relevant parts of the document - then writes you a clear answer using only that context.

## How it works

1. **Load & chunk** — reads a PDF or TXT file and splits it into overlapping text chunks
2. **Embed** — converts each chunk into a vector embedding using `sentence-transformers` (`all-MiniLM-L6-v2`)
3. **Search** — embeds the user's question and finds the top-k most similar chunks using cosine similarity
4. **Generate** — sends the top chunks + question to Cohere's Command model to produce a natural-language answer

## Project structure

```
rag_assignment/
├── README.md
├── requirements.txt
├── app.py           # Streamlit interface
├── functions.py      # Core functions: load_and_chunk_document, create_embeddings, search_chunks, generate_answer
├── notebook.ipynb     # Development/testing notebook
└── screenshots/       # Screenshots of the app running
```

## Setup

1. Clone this repo:
   ```
   git clone https://github.com/<your-username>/rag_assignment.git
   cd rag_assignment
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the app:
   ```
   streamlit run app.py
   ```

4. Open the app in your browser (Streamlit will print a local URL, usually `http://localhost:8501`)

## Using the app

1. Upload a PDF or TXT file from the sidebar
2. (Optional) Adjust chunk size, overlap, and number of results in the sidebar
3. (Optional) Paste a free [Cohere API key](https://dashboard.cohere.com) in the sidebar to get an AI-generated answer, not just raw passages
4. Type a question in the main text box
5. Click **Search** to see the top matching passages and (if a Cohere key is provided) a generated answer

## Functions (`functions.py`)

- `load_and_chunk_document(file_path, chunk_size=300, overlap=50)` — reads a file and splits it into chunks
- `create_embeddings(chunks, model_name="all-MiniLM-L6-v2")` — creates embeddings for a list of chunks
- `search_chunks(query, chunks, embeddings, k=3)` — finds the top-k most similar chunks to a query
- `generate_answer(query, context, api_key)` — generates a natural-language answer using Cohere, grounded in the retrieved context

## Tech stack

- **Streamlit** — web interface
- **sentence-transformers** — text embeddings (`all-MiniLM-L6-v2`)
- **scikit-learn** — cosine similarity search
- **Cohere** — answer generation (`command-a-03-2025`)
- **PyPDF2 / LangChain text splitters** — document loading and chunking

## Future improvements

- Support for more file types (DOCX, HTML)
- Persistent vector storage instead of re-embedding on every search
- Source highlighting to show exactly where in the document an answer came from
