"""
SAP RAG Chatbot: Streamlit UI
Run with:
  streamlit run app.py
"""

import os
import numpy as np
import faiss
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

#streamlit page config 
st.set_page_config(page_title="SAP Knowledge Bot", page_icon="🤖", layout="centered")
st.title("🤖 SAP Knowledge Bot")
st.caption("Answers grounded in the [SAP Wikipedia article](https://en.wikipedia.org/wiki/SAP)")

#RAG pipeline, cached to only run once
@st.cache_resource(show_spinner="Building knowledge base from Wikipedia…")
def build_knowledge_base():
    url = "https://en.wikipedia.org/wiki/SAP"
    response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    content = soup.find("div", {"id": "mw-content-text"})
    for tag in content.find_all(["table", "sup", "span", "style", "script"]):
        tag.decompose()
    paragraphs = content.find_all("p")
    text = "\n".join(p.get_text() for p in paragraphs if p.get_text().strip())

    words = text.split()
    #for type of chunking, we use a sliding window chunking: fixed-size chunking (300) with overlap (50)
    chunk_size, overlap = 300, 50
    chunks, start = [], 0
    while start < len(words):
        chunk = " ".join(words[start: start + chunk_size])
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=False, batch_size=64)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return chunks, index, model

# distance metric used to compare vectors during search is cosine similarity 
def retrieve(query, index, chunks, model, top_k=3):
    vec = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vec)
    _, indices = index.search(vec, top_k)
    return [chunks[i] for i in indices[0]]

# max_tokens=300 limits the response length (can be increased for more detail)
# temperature=0 makes answers deterministic and factual (no creativity)
# both are intentional choices for a grounded RAG system
def generate_answer(query, context_chunks):
    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
    context = "\n\n---\n\n".join(context_chunks)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=300,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question using ONLY "
                    "the context provided. If the answer is not in the context, say: "
                    "'I could not find that in the SAP Wikipedia article.'"
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )
    return response.choices[0].message.content.strip()


# loadin' knowledge base
chunks, index, embed_model = build_knowledge_base()

# history of hcat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I can answer questions about SAP based on the Wikipedia article. Try asking me about SAP's history, products, or founders!",
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📄 Source chunks used for grounding"):
                for i, chunk in enumerate(msg["sources"], 1):
                    st.markdown(f"**[{i}]** {chunk[:300]}…")


if query := st.chat_input("Ask something about SAP…"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating answer…"):
            retrieved = retrieve(query, index, chunks, embed_model, top_k=3)
            answer = generate_answer(query, retrieved)

        st.markdown(answer)
        with st.expander("📄 Source chunks used for grounding"):
            for i, chunk in enumerate(retrieved, 1):
                st.markdown(f"**[{i}]** {chunk[:300]}…")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": retrieved}
    )