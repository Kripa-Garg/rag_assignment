"""
app.py
Simple RAG app with a cleaner layout:
- Sidebar: file upload + settings
- Main area: question input, search, and styled result cards
"""

import streamlit as st
import tempfile
import os
import re
from functions import load_and_chunk_document, create_embeddings, search_chunks, generate_answer

st.set_page_config(page_title="DocSage", page_icon="🧠", layout="wide")

def clean_chunk_text(text):
    """Collapses PDF-extraction line breaks and extra spaces into normal flowing text."""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# --- Custom styling ---
st.markdown("""
<style>
.result-card {
    background-color: #1e2129;
    border-left: 4px solid #6C63FF;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 14px;
}
.score-badge {
    display: inline-block;
    background-color: #6C63FF;
    color: white;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.8em;
    font-weight: 600;
    margin-bottom: 8px;
}
.chunk-text {
    color: #d0d0d0;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar: upload + settings ---
with st.sidebar:
    st.header("📄 Document Setup")
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

    st.divider()
    st.subheader("Settings")
    chunk_size = st.slider("Chunk size", min_value=100, max_value=1000, value=300, step=50)
    overlap = st.slider("Chunk overlap", min_value=0, max_value=200, value=50, step=10)
    top_k = st.slider("Number of results (k)", min_value=1, max_value=10, value=3)

    st.divider()
    st.subheader("AI Answer (optional)")
    cohere_api_key = st.text_input("Cohere API key", type="password", help="Get a free key at dashboard.cohere.com")

    st.divider()
    st.caption("Built for the RAG internship assignment 🚀")

# --- Main area ---
st.title("🧠 DocSage")
st.caption("Ask questions, get the most relevant passages from your document — instantly.")

query = st.text_input("💬 Ask a question about the document", placeholder="e.g. What is this document about?")

search_clicked = st.button("Search", type="primary", use_container_width=False)

if search_clicked:
    if uploaded_file is None:
        st.warning("Please upload a file from the sidebar first.")
    elif not query.strip():
        st.warning("Please type a question first.")
    else:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        progress = st.progress(0, text="Loading and chunking document...")
        chunks = load_and_chunk_document(tmp_path, chunk_size=chunk_size, overlap=overlap)

        progress.progress(40, text="Creating embeddings...")
        embeddings = create_embeddings(chunks)

        progress.progress(75, text="Searching for relevant chunks...")
        results = search_chunks(query, chunks, embeddings, k=top_k)

        progress.progress(100, text="Done!")
        progress.empty()

        st.subheader(f"Top {len(results)} matching passages")
        for i, (chunk, score) in enumerate(results):
            display_text = clean_chunk_text(chunk)
            st.markdown(f"""
            <div class="result-card">
                <div class="score-badge">Match {i+1} · Score {score:.3f}</div>
                <div class="chunk-text">{display_text}</div>
            </div>
            """, unsafe_allow_html=True)

        # --- AI-generated answer using Cohere ---
        if cohere_api_key.strip():
            with st.spinner("Generating answer with Cohere..."):
                context = "\n\n".join(clean_chunk_text(c) for c, _ in results)
                try:
                    answer = generate_answer(query, context, cohere_api_key)
                    st.subheader("🤖 AI-Generated Answer")
                    st.markdown(f"""
                    <div class="result-card" style="border-left-color:#00C896;">
                        <div class="chunk-text">{answer}</div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Couldn't generate an answer: {e}")
        else:
            st.info("💡 Add a Cohere API key in the sidebar to get an AI-generated answer, not just raw passages.")

        os.remove(tmp_path)
