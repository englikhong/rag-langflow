# RAG Playground — Langflow Backend

A two-phase Retrieval-Augmented Generation (RAG) application with a side-by-side comparison UI.

**Phase 1 — Classic RAG** | **Phase 2 — Agentic RAG**

## Architecture

```
Streamlit UI
    ├── Sidebar: Langflow connection + flow selection
    ├── Shared query bar
    ├── Left panel: Classic RAG (Query → Retrieve → Generate → Answer trace)
    └── Right panel: Agentic RAG (Thought → Action → Observation steps)
         ↕                              ↕
    Langflow Desktop (localhost:7860) — REST API
```

Langflow Desktop handles all RAG pipeline logic. Streamlit calls its API and visualises the results.

## Prerequisites

- Python 3.11+
- [Langflow Desktop](https://www.langflow.org/) installed and running on `http://localhost:7860`
- Ollama running locally (for the fallback LangChain pipeline)

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
streamlit run app.py
```

1. Enter your Langflow URL in the sidebar and click **Test Connection**
2. Select your Classic RAG and Agentic RAG flows from the dropdowns
3. Type a question and click **▶ Run**

## Langflow Flow Design

### Classic RAG Flow (minimum components)
- **File Loader** or **URL Loader** → **Recursive Character Text Splitter** → **Chroma** (vector store)
- **Chroma** (retriever) → **Prompt** → **Ollama / OpenAI LLM** → **Chat Output**

### Agentic RAG Flow (minimum components)
- **Tool-calling Agent** with a **Retriever Tool** wrapping the Chroma vector store
- **Chat Output**

## Fallback

If Langflow is unavailable, the Classic RAG panel offers a fallback that runs the local LangChain pipeline (`query.py` + `ingest.py`) directly against the Chroma vector store.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI — two-panel layout, sidebar, trace rendering |
| `langflow_client.py` | Langflow REST API client + response parsers |
| `config.py` | Configuration (Langflow URL, chunk size, Chroma path) |
| `ingest.py` | Local document ingestion pipeline (fallback) |
| `query.py` | Local RAG query pipeline (fallback) |
