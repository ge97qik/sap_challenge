"""
SAP RAG Chatbot: Streamlit UI
Run with:
    streamlit run sap_chatbot.py
"""

import os
import numpy as np
import faiss
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder 
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# streamlit app configurations 
st.set_page_config(page_title="SAP Knowledge Bot", page_icon="🤖", layout="centered")
st.title("🤖 SAP Knowledge Bot")
st.caption("Answers grounded in the [SAP Wikipedia article](https://en.wikipedia.org/wiki/SAP)")

# exit commands for app, rue based 
EXIT_COMMANDS = {"exit", "quit", "stop", "bye", "goodbye"}

#rag pipeline, cashed to only run once. 
@st.cache_resource(show_spinner="Building knowledge base from Wikipedia…")
def build_knowledge_base():
    url = "https://en.wikipedia.org/wiki/SAP"
    #fetchin' the page
    response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()

    #cleaning the html using beatifulSoup
    soup = BeautifulSoup(response.text, "html.parser")
    content = soup.find("div", {"id": "mw-content-text"})
    for tag in content.find_all(["table", "sup", "span", "style", "script"]):
        tag.decompose()

    paragraphs = content.find_all("p")
    text = "\n".join(p.get_text() for p in paragraphs if p.get_text().strip())

    # sliding window chunking approach -> fixed-size of 300 words and with an overlap of 50 words
    words = text.split()
    chunk_size, overlap = 300, 50
    chunks, start = [], 0
    while start < len(words):
        chunk = " ".join(words[start: start + chunk_size])
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    #embedding
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=False, batch_size=64)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return chunks, index, model


# for retrieval --> distance metric: cosine similarity (IndexFlatIP + L2-normalized vectors)
def retrieve(query, index, chunks, model, top_k=3):
    vec = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vec)
    _, indices = index.search(vec, top_k)
    return [chunks[i] for i in indices[0]]


# Reranker as an option for a more optimized retriveal generation, . 
# The reranker is just a second filter that re-orders the top 10 candidates more precisely before they go to the LLM.
# limitation of the current approach: cosine similarity retrieves by vector closeness, not true relevance.
# A cross-encoder reranker scores each (query, chunk) pair directly which is much more precise. 
# To enable this additional step: uncomment everything below marked RERANKER, and flip the toggle to True.
# Retrieve more candidates (top_k=10) then rerank down to top_k=3

# USE_RERANKER = True                                                 # RERANKER
# @st.cache_resource(show_spinner="Loading reranker model…")         # RERANKER
# def load_reranker():                                                # RERANKER
#     return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")   # RERANKER

# def rerank(query, chunks, top_k=3):                                # RERANKER
#     reranker = load_reranker()                                     # RERANKER
#     scores = reranker.predict([(query, chunk) for chunk in chunks])# RERANKER
#     ranked = sorted(zip(scores, chunks), reverse=True)             # RERANKER
#     return [chunk for _, chunk in ranked[:top_k]]                  # RERANKER

USE_RERANKER = False  # comment for reranker


# additonal refinement: adding one small llm to route classify requests that related from sap from not 
def is_sap_related(query: str, client: OpenAI) -> bool:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=5,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a classifier. Decide if the user's question is related to "
                    "SAP (the company), its products, history, or enterprise software. "
                    "Reply with exactly one word: YES or NO."
                ),
            },
            {"role": "user", "content": query},
        ],
    )
    answer = response.choices[0].message.content.strip().upper()
    return answer.startswith("YES")


# answer generation part ,  
# inference hyperparamters tuned for this specific usecase: 
# temperature=0 → deterministic, factual answers (no creativity) 
# max_tokens=300 → keeps answers concise (can be increased for more detail)
def generate_answer(query, context_chunks, client: OpenAI):
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



chunks, index, embed_model = build_knowledge_base()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

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
        if "sources" in msg and msg["sources"]:
            with st.expander("📄 Source chunks used for grounding"):
                for i, chunk in enumerate(msg["sources"], 1):
                    st.markdown(f"**[{i}]** {chunk[:300]}…")


if query := st.chat_input("Ask something about SAP…"):

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 1. exit check
    if query.strip().lower() in EXIT_COMMANDS:
        farewell = "Goodbye! Feel free to come back anytime. 👋"
        st.session_state.messages.append({"role": "assistant", "content": farewell})
        with st.chat_message("assistant"):
            st.markdown(farewell)
        st.stop()

    # 2. intent check
    elif not is_sap_related(query, client):
        answer = "I can only answer questions about SAP. Try asking about SAP's history, products, founders, or financials."
        retrieved = []

    # 3. normal RAG flow
    else:
        with st.spinner("Retrieving and generating answer…"):
            # retrieve more candidates if reranker is on, otherwise straight to top-3
            candidate_k = 10 if USE_RERANKER else 3
            retrieved = retrieve(query, index, chunks, embed_model, top_k=candidate_k)

            # to use reranker >> uncomment the two lines below (and flip USE_RERANKER = True above)
            # if USE_RERANKER:
            #     retrieved = rerank(query, retrieved, top_k=3)

            answer = generate_answer(query, retrieved, client)

    # display of answer
    with st.chat_message("assistant"):
        st.markdown(answer)
        if retrieved:
            with st.expander("📄 Source chunks used for grounding"):
                for i, chunk in enumerate(retrieved, 1):
                    st.markdown(f"**[{i}]** {chunk[:300]}…")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": retrieved if "retrieved" in dir() else [],
    })