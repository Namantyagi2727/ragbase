"""
ui/local_app.py — Full local RAGBase: uploads, indexing, Standard and Agentic RAG with live inference.

Uploads and the index built from them live in a per-session workspace
(ui/uploads.py). An index that already exists in the shared local folder
(config.INDEX_DIR, built before session workspaces existed) is loaded as the
local library and is never modified here.
"""

from __future__ import annotations

import html
import logging
import os
import time
from datetime import datetime

import streamlit as st

import config
import ingest
import retriever as ret
import tabular
from agent import AgentSettings, run_agent
from llm import LLMError, get_chat_model
from ui import methodology, providers, uploads
from ui import render as rv

log = logging.getLogger(__name__)

MODES = {"Standard RAG": "standard", "Agentic RAG": "agentic"}
OLLAMA_MODELS = ["llama3.2", "llama3.1", "mistral", "gemma2:9b", "phi3"]


def _init_state() -> None:
    defaults = {
        "messages": [], "index_built": False, "index_source": None, "index_dir": None, "docs_dir": None,
        "workspace": None, "chunk_count": 0, "load_error": None, "mode": "Standard RAG", "backend": "ollama",
        "cfg_embed_model": config.EMBED_MODEL, "cfg_ollama_model": config.OLLAMA_MODEL,
        "cfg_chunk_size": config.CHUNK_SIZE, "cfg_chunk_overlap": config.CHUNK_OVERLAP,
        "cfg_top_k": config.TOP_K, "cfg_reranking": config.RERANKING_ENABLED,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


@st.cache_resource(show_spinner=False, max_entries=8)
def _load_retriever(index_dir, embed_model, top_k, reranking):
    # Shared across sessions but keyed by index directory, so each session's
    # workspace index gets its own entry; only indexes built by RAGBase are loaded.
    return ret.get_retriever(index_dir=index_dir, embed_model=embed_model, top_k=top_k, reranking=reranking)


def _activate(index_dir: str, docs_dir: str, source: str) -> None:
    """Load an index built by RAGBase; the index manifest decides the embedding model."""
    manifest = ingest.read_manifest(index_dir)
    if manifest.get("embed_model"):
        st.session_state.cfg_embed_model = manifest["embed_model"]
    retriever_obj = _load_retriever(index_dir, st.session_state.cfg_embed_model,
                                    st.session_state.cfg_top_k, st.session_state.cfg_reranking)
    base = getattr(retriever_obj, "base_retriever", retriever_obj)
    st.session_state.update(index_built=True, index_dir=index_dir, docs_dir=docs_dir, index_source=source,
                            chunk_count=manifest.get("chunk_count") or len(base.chunks_meta))


def _retriever():
    return _load_retriever(st.session_state.index_dir, st.session_state.cfg_embed_model,
                           st.session_state.cfg_top_k, st.session_state.cfg_reranking)


def _try_autoload_library() -> None:
    if st.session_state.index_built or st.session_state.load_error:
        return
    if os.path.exists(os.path.join(config.INDEX_DIR, "index.faiss")):
        try:
            _activate(config.INDEX_DIR, config.DOCS_DIR, "library")
        except Exception as e:
            log.exception("Failed to load existing index")
            st.session_state.load_error = f"Existing index could not be loaded ({type(e).__name__})."


def _build_session_index(uploaded_files) -> None:
    rejected, accepted = [], []
    for uf in uploaded_files:
        name = ingest.safe_filename(uf.name)
        if name is None:
            rejected.append((uf.name, "unsupported or unsafe file name"))
        elif uf.size > config.MAX_UPLOAD_MB * 1024 * 1024:
            rejected.append((uf.name, f"larger than {config.MAX_UPLOAD_MB} MB"))
        else:
            accepted.append((name, uf))
    if not accepted:
        st.warning("None of the uploaded files can be indexed.")
    else:
        previous = st.session_state.workspace
        ws = uploads.create()
        for name, uf in accepted:
            with open(os.path.join(ws.docs_dir, name), "wb") as fh:
                fh.write(uf.getbuffer())
        progress_bar = st.progress(0.0, text="Initialising…")
        with st.spinner("Indexing…"):
            report = ingest.ingest_documents(
                ingest_dir=ws.docs_dir, chunk_size=st.session_state.cfg_chunk_size,
                chunk_overlap=st.session_state.cfg_chunk_overlap, embed_model=st.session_state.cfg_embed_model,
                index_dir=ws.index_dir, progress_fn=lambda frac, msg: progress_bar.progress(frac, text=msg),
            )
        progress_bar.empty()
        if report.chunks:
            st.session_state.workspace = ws
            _activate(ws.index_dir, ws.docs_dir, "session")
            if previous is not None:
                _load_retriever.clear()
                uploads.remove(previous)
            st.success(f"✓  {report.chunks} chunks from {len(report.files_indexed)} file(s)")
        else:
            uploads.remove(ws)
            st.error("No text could be indexed from the uploaded files.")
        for fn, reason in report.skipped:
            rejected.append((fn, reason))
        for w in report.warnings:
            st.info(w)
    for fn, reason in rejected:
        st.warning(f"Skipped {fn}: {reason}")


def _sidebar() -> None:
    with st.sidebar:
        st.markdown("""
        <div class="rb-brand">
            <div class="rb-brand-logo-wrap">
                <span class="rb-brand-hex">⬡</span>
                <span class="rb-brand-logo">RAGBASE</span>
            </div>
            <div class="rb-brand-sub">// enterprise document intelligence</div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        st.markdown('<div class="rb-label">Answer mode</div>', unsafe_allow_html=True)
        st.session_state.mode = st.radio("Answer mode", list(MODES), key="mode_radio",
                                         index=list(MODES).index(st.session_state.mode),
                                         label_visibility="collapsed", horizontal=True)
        st.caption("Standard: one search, one model call. Agentic: plans searches, checks claims against "
                   "citations and may refine — more model calls and slower.")
        st.divider()

        st.markdown('<div class="rb-label">Documents</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Drag & drop or browse", type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "csv", "xlsx"],
            accept_multiple_files=True, label_visibility="collapsed",
        )
        if st.button("⬡  Build Index", type="primary", use_container_width=True):
            if not uploaded_files:
                st.warning("Upload at least one document first.")
            elif len(uploaded_files) > config.MAX_UPLOAD_FILES:
                st.warning(f"Upload at most {config.MAX_UPLOAD_FILES} files per build.")
            else:
                _build_session_index(uploaded_files)
        st.caption("Uploads are indexed in a temporary workspace for this session only.")
        if st.session_state.load_error:
            st.error(st.session_state.load_error)
        if st.session_state.index_built:
            source = ("this session's uploads" if st.session_state.index_source == "session"
                      else "existing local library")
            st.metric("Indexed Chunks", st.session_state.chunk_count, help=f"Index: {source}")
        st.divider()

        st.markdown('<div class="rb-label">Inference</div>', unsafe_allow_html=True)
        backends = providers.available_backends()
        if len(backends) > 1:
            st.session_state.backend = st.selectbox(
                "Backend", backends, index=backends.index(st.session_state.backend)
                if st.session_state.backend in backends else 0,
                format_func=lambda b: {"ollama": "Ollama (direct, local)", "prism": "Prism gateway"}[b])
        else:
            st.session_state.backend = "ollama"
        st.markdown(providers.notice_html(st.session_state.backend, st.session_state.cfg_ollama_model),
                    unsafe_allow_html=True)
        st.divider()

        st.markdown('<div class="rb-label">Configuration</div>', unsafe_allow_html=True)
        with st.expander("Model & Retrieval Settings"):
            embed_options = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "multi-qa-MiniLM-L6-cos-v1"]
            if st.session_state.cfg_embed_model not in embed_options:
                embed_options.append(st.session_state.cfg_embed_model)
            st.session_state.cfg_embed_model = st.selectbox(
                "Embedding Model", embed_options, index=embed_options.index(st.session_state.cfg_embed_model))
            st.session_state.cfg_ollama_model = st.selectbox(
                "LLM  (Ollama)", OLLAMA_MODELS,
                index=OLLAMA_MODELS.index(st.session_state.cfg_ollama_model)
                if st.session_state.cfg_ollama_model in OLLAMA_MODELS else 0)
            st.session_state.cfg_chunk_size = st.slider("Chunk Size (chars)", 128, 1024,
                                                        st.session_state.cfg_chunk_size, step=64)
            st.session_state.cfg_chunk_overlap = st.slider("Chunk Overlap", 0, 256,
                                                           st.session_state.cfg_chunk_overlap, step=32)
            st.session_state.cfg_top_k = st.slider("Top-K Chunks", 1, 10, st.session_state.cfg_top_k)
            st.session_state.cfg_reranking = st.toggle("Cross-Encoder Reranking",
                                                       value=st.session_state.cfg_reranking)
            st.caption("Embedding model and chunking apply on the next Build Index.")
            if st.button("Apply & Reload", use_container_width=True):
                if st.session_state.index_built:
                    built_with = ingest.read_manifest(st.session_state.index_dir).get("embed_model")
                    if built_with and built_with != st.session_state.cfg_embed_model:
                        st.info(f"Index was built with {built_with}; rebuild to switch embedding model.")
                    _activate(st.session_state.index_dir, st.session_state.docs_dir, st.session_state.index_source)
                    st.success("✓  Retriever reloaded")
                else:
                    st.info("Build an index first.")
        st.divider()

        st.markdown('<div class="rb-label">Session</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.session_state.messages:
                chat_export = "\n\n".join(f"**{m['role'].upper()}**: {m['content']}"
                                          for m in st.session_state.messages)
                st.download_button("Export", data=chat_export.encode(),
                                   file_name=f"ragbase_{datetime.now():%Y%m%d_%H%M}.md",
                                   mime="text/markdown", use_container_width=True)
        st.divider()

        if st.session_state.index_built:
            label = "Agentic" if MODES[st.session_state.mode] == "agentic" else "Standard"
            status_html = (f'<div class="rb-status"><div class="rb-status-dot on"></div>'
                           f'<span class="rb-status-label on">Index Ready</span>'
                           f'<span class="rb-model-chip">{html.escape(label)} · '
                           f'{html.escape(st.session_state.cfg_ollama_model)}</span></div>')
        else:
            status_html = ('<div class="rb-status"><div class="rb-status-dot off"></div>'
                           '<span class="rb-status-label">Awaiting Index</span></div>')
        st.markdown(status_html, unsafe_allow_html=True)


def _chat_model():
    return get_chat_model(backend=st.session_state.backend, model=st.session_state.cfg_ollama_model)


def answer(prompt: str) -> dict:
    """Answer with the selected mode; returns the chat message to store."""
    mode = MODES[st.session_state.mode]
    docs_dir = st.session_state.docs_dir or config.DOCS_DIR
    tables = tabular.load_tables(docs_dir) if config.TABULAR_QA_ENABLED else []
    message = {"role": "assistant", "content": "", "view": None}
    try:
        if tabular.route_to_tabular(prompt, tables):
            with st.spinner("Running data analysis…"):
                response = tabular.answer_tabular_question(prompt, tables, _chat_model())
            message["content"] = response.answer
        elif mode == "agentic":
            with st.spinner("Agent planning, retrieving and checking…"):
                res = run_agent(prompt, _retriever(), _chat_model(), settings=AgentSettings())
            message["view"] = rv.view_from_agent(res)
            message["content"] = res.answer
        else:
            t = time.perf_counter()
            with st.spinner("Searching documents…"):
                result = ret.build_rag_chain(_retriever(), chat_model=_chat_model()).invoke(prompt)
            message["view"] = rv.view_from_standard(result, time.perf_counter() - t)
            message["content"] = result["answer"]
    except LLMError as e:
        message["content"] = f"The language model request failed: {e}"
        message["error"] = True
    return message


def _render_message(msg: dict) -> None:
    if msg.get("error"):
        st.error(msg["content"])
        return
    st.markdown(msg["content"])
    if msg.get("view"):
        rv.render_view(msg["view"])


def main() -> None:
    _init_state()
    if st.session_state.workspace is not None:
        uploads.touch(st.session_state.workspace)   # retention counts idle time, not age
    removed = uploads.sweep()
    if removed:
        log.info("Removed %d expired session workspace(s)", removed)
    _try_autoload_library()
    _sidebar()

    st.markdown("""
    <div class="rb-title-wrap">
        <div class="rb-title-row">
            <span class="rb-title">RAGBASE</span>
            <span class="rb-badge">v2.0 · Local</span>
        </div>
    </div>
    <div class="rb-subtitle">
        <span>offline</span> · <span>private</span> · <span>open-source</span>
        &nbsp;·&nbsp; semantic document intelligence
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    tab_chat, tab_method = st.tabs(["Chat", "Methodology & evaluation"])
    prompt = st.chat_input("Query your documents…")
    with tab_method:
        methodology.render()
    with tab_chat:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                _render_message(msg)
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                if not st.session_state.index_built:
                    message = {"role": "assistant", "content": "Upload documents and build the index first.",
                               "view": None}
                    st.warning("Upload documents and build the index first (sidebar →)")
                else:
                    message = answer(prompt)
                    _render_message(message)
            st.session_state.messages.append(message)
