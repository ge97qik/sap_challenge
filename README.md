# SAP Knowledge Bot – RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about SAP using the [SAP Wikipedia page](https://en.wikipedia.org/wiki/SAP) as its only knowledge source. Built without heavy frameworks —> just four explicit steps so the architecture is transparent and easy to explain.

---

## How it works

```
Wikipedia page
     │
     ▼
[1] Fetch & clean      requests + BeautifulSoup → raw text
     │
     ▼
[2] Chunk              sliding window (300 words, 50-word overlap)
     │
     ▼
[3] Embed & index      sentence-transformers (all-MiniLM-L6-v2) → FAISS
     │
     ▼
[4] Query loop
     │  embed question → cosine similarity search → top-3 chunks
     └─ LLM reads only those chunks → grounded answer + source chunks shown
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/ge97qik/sap_challenge.git
cd sap_challenge
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your API key

Create a `.env` file in the project folder:

```
GROQ_API_KEY=your-groq-key-here
```
Get a key at [console.groq.com](https://console.groq.com) 

### 5. Run the chatbot

```bash
streamlit run sap_chatbot.py
```

The app opens automatically the browser. The knowledge base is built from Wikipedia on first run (~15 seconds), then cached for the rest of the session.

---

## Example prompts

**Prompt 1**
> When was SAP founded and by whom?

**Answer:** SAP was founded on April 1, 1972, by five former IBM employees: Dietmar Hopp, Hasso Plattner, Claus Wellenreuther, Klaus Tschira, and Hans-Werner Hector.

---

**Prompt 2**
> What is SAP?

**Answer:** SAP is a German multinational software company based in Walldorf, Baden-Württemberg, that is the world's largest vendor of enterprise software.

---

## Design decisions

| Choice | Reason |
|---|---|
| `sentence-transformers` | Free, runs locally, no extra API cost |
| `faiss-cpu` with cosine similarity | Exact search, zero config, sufficient for ~300 chunks |
| Sliding window chunking (300 words, 50 overlap) | Preserves context across chunk boundaries |
| `temperature=0` | Deterministic, factual answers — right for grounded RAG |
| `max_tokens=300` | Keeps answers concise, can be increased for more detail |
| Groq + LLaMA 3.1 | Free API, fast inference, open-source model |
| No LangChain for this usecase | Keeps the pipeline transparent and explainable step by step |

---

## Dependencies

| Package | Purpose |
|---|---|
| `requests` + `beautifulsoup4` | Scrape and clean Wikipedia |
| `sentence-transformers` | Local embedding model |
| `faiss-cpu` | Vector index for similarity search |
| `openai` | OpenAI-compatible client (used with Groq) |
| `streamlit` | Chat UI |
| `python-dotenv` | Load API key from `.env` file |
| `numpy` | Vector operations |