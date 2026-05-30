"""Streamlit app — Classic RAG vs Agentic RAG comparison, backed by Langflow."""
from __future__ import annotations

import concurrent.futures
import os
import tempfile

import streamlit as st

import config
from ingest import ingest
from langflow_client import ClassicRAGTrace, AgenticRAGTrace, LangflowClient
from query import ask

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="RAG Playground", page_icon="🔍", layout="wide")

# ── Session state defaults ────────────────────────────────────────────────────

_DEFAULTS: dict = {
    "connection_ok": False,
    "flows": [],
    "classic_flow_id": "",
    "agentic_flow_id": "",
    "classic_trace": None,
    "agentic_trace": None,
    "langflow_url": config.LANGFLOW_BASE_URL,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Panel render functions ────────────────────────────────────────────────────

def render_classic_panel(trace: ClassicRAGTrace | None, current_query: str = "") -> None:
    st.subheader("⚙️ Classic RAG")

    if trace is None:
        st.info("Run a query to see the pipeline trace here.")
        return

    if trace.error:
        st.error(f"**Error:** {trace.error}")
        with st.expander("Fallback: run via local LangChain pipeline"):
            q = trace.query or current_query
            if q and st.button("Run with LangChain fallback", key="fallback_classic"):
                try:
                    result = ask(q, top_k=config.TOP_K)
                    st.markdown("**Answer:**")
                    st.write(result["answer"])
                    st.caption(f"{len(result['sources'])} source chunk(s) retrieved locally.")
                except Exception as e:
                    st.error(f"Fallback also failed: {e}")
        return

    # Step 1 — Query
    with st.expander("**1 — Query**", expanded=True):
        st.write(trace.query)

    # Step 2 — Retrieve
    n = len(trace.retrieved_chunks)
    with st.expander(f"**2 — Retrieve** ({n} chunk{'s' if n != 1 else ''})", expanded=True):
        if not trace.retrieved_chunks:
            st.caption("No chunks returned by this flow.")
        for i, chunk in enumerate(trace.retrieved_chunks, 1):
            st.markdown(f"**[{i}] {chunk.source}**")
            if chunk.score is not None:
                st.caption(f"Similarity score: {chunk.score:.4f}")
            st.text(chunk.content[:400])
            if i < len(trace.retrieved_chunks):
                st.divider()

    # Step 3 — Generate
    with st.expander("**3 — Generate**", expanded=False):
        if trace.model_used:
            st.caption(f"Model: `{trace.model_used}`")
        if trace.prompt_template:
            st.code(trace.prompt_template, language="text")
        else:
            st.caption(
                "Prompt template not exposed by this flow. "
                "Add a Prompt component with an output in Langflow to see it here."
            )

    # Step 4 — Answer
    with st.expander("**4 — Answer**", expanded=True):
        st.markdown(trace.answer if trace.answer else "_No answer returned._")


def render_agentic_panel(trace: AgenticRAGTrace | None) -> None:
    st.subheader("🤖 Agentic RAG")

    if trace is None:
        st.info("Run a query to see agent reasoning here.")
        st.caption("_Select an Agentic RAG flow in the sidebar to enable this panel._")
        return

    if trace.error:
        st.error(f"**Error:** {trace.error}")
        return

    n_steps = len(trace.steps)
    with st.expander(f"**Agent Reasoning** ({n_steps} step{'s' if n_steps != 1 else ''})", expanded=True):
        if not trace.steps:
            st.caption("No reasoning steps captured.")
        for i, step in enumerate(trace.steps, 1):
            st.markdown(f"**Step {i}**")
            if step.thought:
                st.markdown(f"_Thought:_ {step.thought}")
            if step.action:
                st.markdown(f"_Action:_ `{step.action}`")
            if step.action_input:
                st.markdown(f"_Input:_ {step.action_input}")
            if step.observation:
                st.markdown(f"_Observation:_ {step.observation[:400]}")
            if i < len(trace.steps):
                st.divider()

    with st.expander("**Final Answer**", expanded=True):
        st.markdown(trace.answer if trace.answer else "_No answer returned._")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔧 Configuration")

    st.subheader("Langflow Connection")
    langflow_url = st.text_input(
        "Langflow URL",
        value=st.session_state.langflow_url,
        placeholder="http://localhost:7860",
        key="_url_input",
    )
    st.session_state.langflow_url = langflow_url

    if st.button("Test Connection", use_container_width=True):
        _probe = LangflowClient(base_url=langflow_url, timeout=5)
        ok, msg = _probe.test_connection()
        if ok:
            st.session_state.connection_ok = True
            try:
                st.session_state.flows = LangflowClient(base_url=langflow_url).get_flows()
                st.success(msg)
            except Exception as e:
                st.session_state.flows = []
                st.warning(f"Connected but could not fetch flows: {e}")
        else:
            st.session_state.connection_ok = False
            st.session_state.flows = []
            st.error(msg)

    connected = st.session_state.connection_ok
    st.markdown(f"Status: {'🟢 Connected' if connected else '🔴 Disconnected'}")

    st.divider()
    st.subheader("Flow Selection")

    flows = st.session_state.flows
    flow_map: dict[str, str] = {f["name"]: f["id"] for f in flows}
    flow_names = list(flow_map.keys())
    no_flows = not flow_names

    if no_flows:
        st.caption("Connect to Langflow to load flows.")
        flow_names = ["(none)"]

    classic_name = st.selectbox(
        "Classic RAG Flow",
        flow_names,
        disabled=no_flows,
        key="_classic_select",
    )
    agentic_name = st.selectbox(
        "Agentic RAG Flow",
        flow_names,
        disabled=no_flows,
        key="_agentic_select",
    )

    st.session_state.classic_flow_id = flow_map.get(classic_name, "")
    st.session_state.agentic_flow_id = flow_map.get(agentic_name, "")

    if not no_flows:
        cid = st.session_state.classic_flow_id
        aid = st.session_state.agentic_flow_id
        st.caption(f"Classic: `{cid[:8]}…`" if cid else "Classic: not set")
        st.caption(f"Agentic: `{aid[:8]}…`" if aid else "Agentic: not set")


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("🔍 RAG Playground")
st.caption("Classic RAG vs Agentic RAG — powered by Langflow")

tab_ask, tab_ingest = st.tabs(["Ask", "Ingest"])


# ── Ask tab ───────────────────────────────────────────────────────────────────

with tab_ask:
    query = st.text_area(
        "Your question",
        placeholder="What is the leave policy?",
        height=80,
        key="_query_input",
    )

    run_clicked = st.button("▶ Run", type="primary", disabled=not query.strip())

    if run_clicked and query.strip():
        client = LangflowClient(
            base_url=st.session_state.langflow_url,
            timeout=config.LANGFLOW_REQUEST_TIMEOUT,
        )
        classic_id = st.session_state.classic_flow_id
        agentic_id = st.session_state.agentic_flow_id

        with st.spinner("Running both pipelines…"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                f_classic = executor.submit(client.run_classic_rag, classic_id, query)
                f_agentic = executor.submit(client.run_agentic_rag, agentic_id, query)
                st.session_state.classic_trace = f_classic.result()
                st.session_state.agentic_trace = f_agentic.result()

    st.divider()
    col_classic, col_agentic = st.columns(2, gap="large")

    with col_classic:
        render_classic_panel(st.session_state.classic_trace, query)

    with col_agentic:
        render_agentic_panel(st.session_state.agentic_trace)


# ── Ingest tab ────────────────────────────────────────────────────────────────

with tab_ingest:
    st.header("Ingest Documents")
    st.caption("Upload documents, tune chunk size, then click Ingest.")

    uploaded_files = st.file_uploader(
        "Upload files (.txt, .pdf, .md)",
        accept_multiple_files=True,
        type=["txt", "pdf", "md"],
    )

    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.slider(
            "Chunk size (tokens)",
            min_value=100,
            max_value=2000,
            value=config.CHUNK_SIZE,
            step=50,
            help="Smaller chunks = finer retrieval; larger chunks = more context per hit.",
        )
    with col2:
        chunk_overlap = st.slider(
            "Chunk overlap (tokens)",
            min_value=0,
            max_value=400,
            value=config.CHUNK_OVERLAP,
            step=10,
        )

    if st.button("Ingest", type="primary", disabled=not uploaded_files):
        with tempfile.TemporaryDirectory() as tmp_dir:
            for f in uploaded_files:
                dest = os.path.join(tmp_dir, f.name)
                with open(dest, "wb") as out:
                    out.write(f.getbuffer())

            with st.spinner("Loading → chunking → embedding → storing …"):
                try:
                    n_chunks = ingest(
                        source_dir=tmp_dir,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        persist_dir=config.CHROMA_DIR,
                    )
                    st.success(f"Done! Created {n_chunks} chunks from {len(uploaded_files)} file(s).")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    st.divider()
    st.subheader("Or ingest sample HR docs")
    if st.button("Ingest sample docs"):
        if not os.path.isdir(config.DOCS_DIR):
            st.warning(f"Sample docs directory `{config.DOCS_DIR}` not found.")
        else:
            with st.spinner("Ingesting sample docs …"):
                try:
                    n = ingest(persist_dir=config.CHROMA_DIR)
                    st.success(f"Done! Created {n} chunks.")
                except Exception as e:
                    st.error(f"Failed: {e}")
