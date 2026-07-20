"""
functions.py
Custom functions for the RAG assignment:
1. load_and_chunk_document
2. create_embeddings
3. search_chunks
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import PyPDF2
import cohere


def load_and_chunk_document(file_path, chunk_size=300, overlap=50):
    """
    Reads a TXT or PDF file and splits it into overlapping text chunks.
    """
    # Step 1: Read the file's raw text
    if file_path.lower().endswith(".pdf"):
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    else:  # assume .txt
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    # Step 2: Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    chunks = splitter.split_text(text)

    # Step 3: Return the chunks
    return chunks


def create_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    """
    Converts a list of text chunks into embeddings (lists of floats).
    """
    # Step 1: Load the model
    model = SentenceTransformer(model_name)

    # Step 2: Create embeddings for all chunks
    embeddings = model.encode(chunks)

    # Step 3: Return as list of lists
    return embeddings.tolist()


def search_chunks(query, chunks, embeddings, k=3):
    """
    Finds the top-k chunks most similar to the query using cosine similarity.
    """
    # Step 1: Embed the query using the same model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query])

    # Step 2: Calculate cosine similarity between query and all chunk embeddings
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # Step 3: Get indices of the top-k highest similarity scores
    top_k_indices = np.argsort(similarities)[::-1][:k]

    # Step 4: Return the top-k chunks with their similarity scores
    results = [(chunks[i], float(similarities[i])) for i in top_k_indices]
    return results


def generate_answer(query, context, api_key):
    """
    Uses Cohere to generate a natural-language answer to `query`,
    grounded only in the provided `context` (the retrieved chunks).
    """
    # Step 1: Initialize the Cohere client with the given API key
    co = cohere.Client(api_key)

    # Step 2: Build a prompt combining the context and the question
    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't have enough information.

Context:
{context}

Question: {query}

Answer:"""

    # Step 3: Call Cohere's chat model to generate the answer
    response = co.chat(
        model="command-a-03-2025",
        message=prompt
    )

    # Step 4: Return the generated answer text
    return response.text
