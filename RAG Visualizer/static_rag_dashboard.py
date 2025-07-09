import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

st.set_page_config(page_title="Static RAG Dashboard (FLAN-T5)", layout="wide")
st.title("🔍 Retrieval-Augmented Generation (RAG) Dashboard — Static Query")
st.markdown("**Embeddings:** all-MiniLM-L6-v2 &nbsp;&nbsp;|&nbsp;&nbsp; **LLM:** google/flan-t5-base")

# Example corpus
documents = [
    "RAG enables LLMs to use external knowledge for accurate answers.",
    "Vector databases store document embeddings for fast retrieval.",
    "Cosine similarity measures how close a query is to a document.",
    "Prompt engineering helps LLMs use retrieved context effectively.",
    "Embedding models turn text into numerical vectors.",
    "LLMs can hallucinate, but RAG grounds answers in facts.",
    "Updating embeddings keeps the knowledge base fresh.",
    "Chunking splits documents for more granular retrieval.",
    "RAG reduces the need for expensive retraining of LLMs.",
    "Transparency improves as RAG can cite its sources."
]

# Static query
query = "How does RAG improve LLM accuracy?"

# Cache embedding and retrieval setup
@st.cache_resource
def load_embedder_and_index():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = embedder.encode(documents, convert_to_numpy=True)
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings)
    return embedder, index, doc_embeddings

embedder, index, doc_embeddings = load_embedder_and_index()

# Cache FLAN-T5 model
@st.cache_resource
def load_generator():
    return pipeline("text2text-generation", model="google/flan-t5-base")

generator = load_generator()

# Embed query and retrieve top-k
query_embedding = embedder.encode([query], convert_to_numpy=True)
k = 3
D, I = index.search(query_embedding, k)
retrieved_docs = [documents[i] for i in I[0]]

# Build prompt for FLAN-T5
context = "\n".join(retrieved_docs)
prompt = f"""Context:
{context}

Question: {query}
Answer:"""

# Generate answer
answer = generator(prompt, max_new_tokens=128, do_sample=False)[0]['generated_text'].strip()

# --- VISUALIZATION ---
st.subheader("Query")
st.info(query)

st.subheader("Retrieved Context Chunks")
for idx, doc in enumerate(retrieved_docs):
    st.code(f"{doc}", language="text")

st.subheader("FLAN-T5 Answer")
st.success(answer)

st.subheader("RAG Pipeline Diagram")
st.graphviz_chart("""
digraph RAG {
    Query [shape=rect, style=filled, color="#ffe600"];
    Embedding [shape=rect, style=filled, color="#00ffe7"];
    Retrieval [label="Similarity Search", shape=rect, style=filled, color="#ff00c8"];
    Augmentation [label="Augment Prompt", shape=rect, style=filled, color="#00ffe7"];
    LLM [label="LLM Generation", shape=rect, style=filled, color="#ffe600"];
    Query -> Embedding -> Retrieval -> Augmentation -> LLM;
}
""")

st.subheader("First Principles Explanation")
st.markdown("""
- **Embedding:** The query and documents are converted to vectors.
- **Retrieval:** The system finds the most similar document chunks.
- **Augmentation:** Retrieved context is combined with the query.
- **LLM Generation:** The LLM generates an answer using both the query and the retrieved context.
""")
