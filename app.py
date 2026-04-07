"""
app.py — Streamlit UI for RAGBase.

Run with:  streamlit run app.py
"""

import os
import io
import logging
from datetime import datetime

import streamlit as st

import config
import ingest
import retriever as ret

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAGBase",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Neural Terminal aesthetic
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=JetBrains+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Outfit:wght@300;400;500;600&display=swap');

/* ================================================================
   DESIGN TOKENS
   ================================================================ */
:root {
    --bg-deep:          #03050e;
    --bg-mid:           #070c18;
    --bg-panel:         #0a1020;
    --bg-raised:        #0f1830;
    --bg-card:          #0c1628;

    --cyan:             #00d4ff;
    --cyan-dim:         rgba(0,212,255,0.12);
    --cyan-border:      rgba(0,212,255,0.28);
    --cyan-glow-sm:     0 0 8px rgba(0,212,255,0.35);
    --cyan-glow-lg:     0 0 24px rgba(0,212,255,0.25);

    --amber:            #ffb800;
    --amber-dim:        rgba(255,184,0,0.10);
    --amber-border:     rgba(255,184,0,0.25);

    --green:            #00ff9d;
    --green-dim:        rgba(0,255,157,0.12);
    --red:              #ff3860;

    --text-bright:      #d8edf5;
    --text-mid:         #5a7d95;
    --text-dim:         #233044;

    --border-faint:     #0d1a28;
    --border-base:      #152235;
    --border-mid:       #1e3450;

    --font-display:     'Orbitron', monospace;
    --font-mono:        'JetBrains Mono', monospace;
    --font-body:        'Outfit', sans-serif;
}

/* ================================================================
   GLOBAL
   ================================================================ */
html, body {
    font-family: var(--font-body) !important;
    color: var(--text-bright) !important;
}

/* App shell */
[data-testid="stAppViewContainer"],
.stApp {
    background-color: var(--bg-deep) !important;
    background-image:
        radial-gradient(ellipse 60% 35% at 15% -5%, rgba(0,212,255,0.07) 0%, transparent 100%),
        radial-gradient(ellipse 50% 30% at 90% 105%, rgba(0,180,220,0.05) 0%, transparent 100%),
        radial-gradient(rgba(0,212,255,0.055) 1px, transparent 1px);
    background-size: 100% 100%, 100% 100%, 26px 26px;
}

.main .block-container {
    padding: 1.75rem 2.5rem 4rem !important;
    max-width: 1180px !important;
}

/* ================================================================
   TYPOGRAPHY
   ================================================================ */
h1, h2, h3, h4 {
    font-family: var(--font-display) !important;
    letter-spacing: 0.06em !important;
}

/* ================================================================
   SIDEBAR
   ================================================================ */
[data-testid="stSidebar"] {
    background-color: var(--bg-panel) !important;
    border-right: 1px solid var(--border-base) !important;
}

/* Glowing left-edge gradient rail */
[data-testid="stSidebar"] {
    position: relative !important;
    overflow: visible !important;
}
[data-testid="stSidebar"]::after {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 2px;
    background: linear-gradient(
        180deg,
        transparent 0%,
        var(--cyan) 30%,
        var(--amber) 70%,
        transparent 100%
    );
    opacity: 0.55;
    pointer-events: none;
    z-index: 10;
}

[data-testid="stSidebar"] > div {
    padding-top: 1.25rem !important;
}

/* Sidebar scrollbar */
[data-testid="stSidebar"]::-webkit-scrollbar { width: 3px; }
[data-testid="stSidebar"]::-webkit-scrollbar-track { background: var(--bg-deep); }
[data-testid="stSidebar"]::-webkit-scrollbar-thumb {
    background: var(--border-mid);
    border-radius: 2px;
}

/* ================================================================
   BUTTONS
   ================================================================ */
button[data-testid="baseButton-secondary"],
button[data-testid="baseButton-tertiary"] {
    font-family: var(--font-mono) !important;
    font-size: 0.74rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 3px !important;
    padding: 0.4rem 0.85rem !important;
    background: transparent !important;
    border: 1px solid var(--border-mid) !important;
    color: var(--text-mid) !important;
    transition: border-color 0.2s, color 0.2s, box-shadow 0.2s, background 0.2s !important;
}

button[data-testid="baseButton-secondary"]:hover,
button[data-testid="baseButton-tertiary"]:hover {
    border-color: var(--cyan) !important;
    color: var(--cyan) !important;
    background: var(--cyan-dim) !important;
    box-shadow: var(--cyan-glow-sm), inset 0 0 10px rgba(0,212,255,0.04) !important;
}

button[data-testid="baseButton-primary"] {
    font-family: var(--font-mono) !important;
    font-size: 0.74rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 3px !important;
    padding: 0.4rem 0.85rem !important;
    background: linear-gradient(135deg, rgba(0,212,255,0.18), rgba(0,212,255,0.06)) !important;
    border: 1px solid var(--cyan-border) !important;
    color: var(--cyan) !important;
    box-shadow: var(--cyan-glow-sm) !important;
    transition: all 0.2s !important;
}

button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, rgba(0,212,255,0.28), rgba(0,212,255,0.12)) !important;
    box-shadow: var(--cyan-glow-lg) !important;
}

/* Download button */
[data-testid="stDownloadButton"] button {
    font-family: var(--font-mono) !important;
    font-size: 0.74rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 3px !important;
    background: transparent !important;
    border: 1px solid var(--border-mid) !important;
    color: var(--text-mid) !important;
    transition: all 0.2s !important;
}

[data-testid="stDownloadButton"] button:hover {
    border-color: var(--amber) !important;
    color: var(--amber) !important;
    background: var(--amber-dim) !important;
    box-shadow: 0 0 10px rgba(255,184,0,0.2) !important;
}

/* ================================================================
   FILE UPLOADER
   ================================================================ */
[data-testid="stFileUploadDropzone"] {
    background: var(--bg-mid) !important;
    border: 1px dashed var(--border-mid) !important;
    border-radius: 4px !important;
    transition: all 0.2s !important;
}

[data-testid="stFileUploadDropzone"]:hover {
    border-color: var(--cyan-border) !important;
    background: var(--cyan-dim) !important;
    box-shadow: inset 0 0 20px rgba(0,212,255,0.04) !important;
}

[data-testid="stFileUploaderFile"] {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-base) !important;
    border-radius: 3px !important;
}

/* ================================================================
   SELECT / DROPDOWN
   ================================================================ */
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-mid) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: 3px !important;
    color: var(--text-bright) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.03em !important;
}

[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--cyan-border) !important;
    box-shadow: 0 0 0 1px var(--cyan-border) !important;
}

/* Dropdown list */
[data-baseweb="popover"] {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: 3px !important;
}

[role="option"] {
    background: transparent !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    color: var(--text-mid) !important;
}

[role="option"]:hover, [aria-selected="true"] {
    background: var(--cyan-dim) !important;
    color: var(--cyan) !important;
}

/* ================================================================
   SLIDERS
   ================================================================ */
[data-testid="stSlider"] [role="slider"] {
    background-color: var(--cyan) !important;
    border: 2px solid var(--bg-deep) !important;
    box-shadow: 0 0 8px rgba(0,212,255,0.6), 0 0 0 2px var(--cyan-border) !important;
    width: 14px !important;
    height: 14px !important;
}

[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    font-family: var(--font-mono) !important;
    font-size: 0.68rem !important;
    color: var(--text-dim) !important;
}

/* ================================================================
   TOGGLE
   ================================================================ */
[data-testid="stToggle"] label {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    color: var(--text-mid) !important;
}

/* ================================================================
   EXPANDER
   ================================================================ */
[data-testid="stExpander"] {
    background: var(--bg-mid) !important;
    border: 1px solid var(--border-base) !important;
    border-radius: 3px !important;
}

[data-testid="stExpander"] summary {
    color: var(--text-mid) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

[data-testid="stExpander"] summary:hover { color: var(--cyan) !important; }

/* ================================================================
   METRIC
   ================================================================ */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: 3px !important;
    padding: 0.8rem 1rem !important;
    position: relative;
    overflow: hidden;
}

[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent);
    opacity: 0.6;
}

[data-testid="stMetricValue"] {
    font-family: var(--font-display) !important;
    font-size: 1.65rem !important;
    color: var(--cyan) !important;
    text-shadow: 0 0 18px rgba(0,212,255,0.5) !important;
}

[data-testid="stMetricLabel"] {
    font-family: var(--font-mono) !important;
    font-size: 0.67rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: var(--text-dim) !important;
}

/* ================================================================
   PROGRESS BAR
   ================================================================ */
[data-testid="stProgress"] div {
    border-radius: 2px !important;
}
[data-testid="stProgress"] [role="progressbar"] {
    background: linear-gradient(90deg, var(--cyan), #0099cc) !important;
    box-shadow: 0 0 8px rgba(0,212,255,0.5) !important;
}

/* ================================================================
   ALERTS
   ================================================================ */
[data-testid="stAlert"] {
    background: var(--bg-mid) !important;
    border: 1px solid var(--border-base) !important;
    border-radius: 3px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
}

/* ================================================================
   SPINNER
   ================================================================ */
[data-testid="stSpinner"] > div {
    border-color: var(--border-mid) !important;
    border-top-color: var(--cyan) !important;
}

/* ================================================================
   CHAT MESSAGES
   ================================================================ */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.3rem 0 !important;
}

/* User avatar */
[data-testid="chatAvatarIcon-user"] {
    background: rgba(0,212,255,0.08) !important;
    border: 1px solid var(--cyan-border) !important;
    color: var(--cyan) !important;
    font-size: 0.85rem !important;
}

/* Assistant avatar */
[data-testid="chatAvatarIcon-assistant"] {
    background: rgba(255,184,0,0.08) !important;
    border: 1px solid var(--amber-border) !important;
    color: var(--amber) !important;
    font-size: 0.85rem !important;
}

/* Message text */
[data-testid="stChatMessage"] .stMarkdown p {
    font-family: var(--font-body) !important;
    font-size: 0.92rem !important;
    line-height: 1.65 !important;
    color: var(--text-bright) !important;
}

/* ================================================================
   CHAT INPUT
   ================================================================ */
[data-testid="stChatInput"] {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: 4px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--cyan-border) !important;
    box-shadow: 0 0 0 1px var(--cyan-border), 0 0 18px rgba(0,212,255,0.08) !important;
}

[data-testid="stChatInput"] textarea {
    font-family: var(--font-body) !important;
    font-size: 0.9rem !important;
    color: var(--text-bright) !important;
    background: transparent !important;
    caret-color: var(--cyan) !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    color: var(--text-dim) !important;
    letter-spacing: 0.04em !important;
}

[data-testid="stChatInput"] button {
    background: transparent !important;
    color: var(--cyan) !important;
    border-radius: 2px !important;
    transition: background 0.15s !important;
}

[data-testid="stChatInput"] button:hover {
    background: var(--cyan-dim) !important;
}

/* ================================================================
   DIVIDER
   ================================================================ */
hr {
    border: none !important;
    border-top: 1px solid var(--border-faint) !important;
    margin: 0.9rem 0 !important;
}

/* ================================================================
   SCROLLBARS
   ================================================================ */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border-mid); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }

/* ================================================================
   CUSTOM HTML COMPONENTS
   ================================================================ */

/* ---- Sidebar brand ---- */
.rb-brand {
    padding: 0.25rem 0.5rem 1rem;
}
.rb-brand-logo-wrap {
    display: flex;
    align-items: center;
    gap: 0.55rem;
}
.rb-brand-logo {
    font-family: var(--font-display);
    font-weight: 900;
    font-size: 1.2rem;
    letter-spacing: 0.18em;
    background: linear-gradient(130deg, #ffffff 0%, var(--cyan) 55%, #0090bb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.rb-brand-hex {
    width: 30px; height: 30px;
    border: 1.5px solid var(--cyan-border);
    border-radius: 5px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: var(--cyan-dim);
    flex-shrink: 0;
    -webkit-text-fill-color: var(--cyan);
    font-size: 1rem;
    box-shadow: 0 0 10px rgba(0,212,255,0.15);
}
.rb-brand-sub {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-top: 0.3rem;
    padding-left: 2px;
}

/* ---- Section label ---- */
.rb-label {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--text-mid);
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 0.1rem 0 0.55rem;
}
.rb-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border-base);
}

/* ---- Status bar ---- */
.rb-status {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.45rem 0.75rem;
    background: var(--bg-mid);
    border: 1px solid var(--border-base);
    border-radius: 3px;
}
.rb-status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
}
.rb-status-dot.on {
    background: var(--green);
    box-shadow: 0 0 5px var(--green), 0 0 10px rgba(0,255,157,0.4);
    animation: glow-pulse 2.2s ease-in-out infinite;
}
.rb-status-dot.off { background: var(--text-dim); }
.rb-status-label {
    font-family: var(--font-mono);
    font-size: 0.69rem;
    letter-spacing: 0.07em;
    color: var(--text-mid);
}
.rb-status-label.on { color: var(--green); }
.rb-model-chip {
    margin-left: auto;
    font-family: var(--font-mono);
    font-size: 0.62rem;
    letter-spacing: 0.06em;
    color: var(--text-dim);
    background: var(--bg-deep);
    border: 1px solid var(--border-faint);
    padding: 0.1rem 0.4rem;
    border-radius: 2px;
}

@keyframes glow-pulse {
    0%, 100% { box-shadow: 0 0 5px var(--green), 0 0 10px rgba(0,255,157,0.4); }
    50%       { box-shadow: 0 0 8px var(--green), 0 0 18px rgba(0,255,157,0.6); }
}

/* ---- Page title ---- */
.rb-title-wrap { margin-bottom: 0.15rem; }
.rb-title-row {
    display: inline-flex;
    align-items: baseline;
    gap: 0.9rem;
}
.rb-title {
    font-family: var(--font-display);
    font-weight: 900;
    font-size: 2rem;
    letter-spacing: 0.12em;
    background: linear-gradient(130deg, #fff 0%, var(--cyan) 55%, #007fa8 100%);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: title-shimmer 5s ease-in-out infinite;
}
.rb-badge {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    font-weight: 400;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--cyan);
    background: var(--cyan-dim);
    border: 1px solid var(--cyan-border);
    padding: 0.18rem 0.55rem;
    border-radius: 2px;
    position: relative;
    top: -4px;
}
.rb-subtitle {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    color: var(--text-dim);
    margin-bottom: 1.5rem;
}
.rb-subtitle span { color: var(--text-mid); }

@keyframes title-shimmer {
    0%, 100% { background-position: 0% 50%; }
    50%       { background-position: 100% 50%; }
}

/* ---- Source reference block ---- */
.rb-source {
    background: var(--bg-mid);
    border: 1px solid var(--border-base);
    border-left: 2px solid var(--cyan-border);
    border-radius: 2px;
    padding: 0.55rem 0.8rem;
    margin-bottom: 0.4rem;
    font-family: var(--font-mono);
    font-size: 0.73rem;
    line-height: 1.55;
}
.rb-source-path {
    color: var(--cyan);
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-bottom: 0.15rem;
}
.rb-source-icon { opacity: 0.55; font-style: normal; font-size: 0.68rem; }
.rb-source-loc {
    color: var(--text-dim);
    font-size: 0.67rem;
    margin-left: 1rem;
}
.rb-source-snippet {
    color: var(--text-mid);
    margin-top: 0.3rem;
    margin-left: 1rem;
    font-size: 0.7rem;
    line-height: 1.4;
    font-style: italic;
}

/* ================================================================
   HIDE STREAMLIT CHROME
   ================================================================ */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "index_built" not in st.session_state:
    st.session_state.index_built = False
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "cfg_embed_model" not in st.session_state:
    st.session_state.cfg_embed_model = config.EMBED_MODEL
if "cfg_ollama_model" not in st.session_state:
    st.session_state.cfg_ollama_model = config.OLLAMA_MODEL
if "cfg_chunk_size" not in st.session_state:
    st.session_state.cfg_chunk_size = config.CHUNK_SIZE
if "cfg_chunk_overlap" not in st.session_state:
    st.session_state.cfg_chunk_overlap = config.CHUNK_OVERLAP
if "cfg_top_k" not in st.session_state:
    st.session_state.cfg_top_k = config.TOP_K
if "cfg_reranking" not in st.session_state:
    st.session_state.cfg_reranking = config.RERANKING_ENABLED


# ---------------------------------------------------------------------------
# Cached resource loader
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_retriever(index_dir, embed_model, top_k, reranking):
    return ret.get_retriever(
        index_dir=index_dir,
        embed_model=embed_model,
        top_k=top_k,
        reranking=reranking,
    )


def _try_autoload():
    idx_path = os.path.join(config.INDEX_DIR, "index.faiss")
    if os.path.exists(idx_path) and not st.session_state.index_built:
        try:
            retriever_obj = _load_retriever(
                config.INDEX_DIR,
                st.session_state.cfg_embed_model,
                st.session_state.cfg_top_k,
                st.session_state.cfg_reranking,
            )
            st.session_state.rag_chain = ret.build_rag_chain(
                retriever_obj, st.session_state.cfg_ollama_model
            )
            st.session_state.index_built = True
        except Exception:
            pass


_try_autoload()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    # Brand
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

    # Documents
    st.markdown('<div class="rb-label">Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drag & drop or browse",
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "csv", "xlsx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("⬡  Build Index", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one document first.")
        else:
            os.makedirs(config.DOCS_DIR, exist_ok=True)
            for uf in uploaded_files:
                dest = os.path.join(config.DOCS_DIR, uf.name)
                with open(dest, "wb") as fh:
                    fh.write(uf.getbuffer())

            progress_bar = st.progress(0.0, text="Initialising…")

            def _progress(frac: float, msg: str):
                progress_bar.progress(frac, text=msg)

            with st.spinner("Indexing…"):
                n = ingest.main(
                    ingest_dir=config.DOCS_DIR,
                    chunk_size=st.session_state.cfg_chunk_size,
                    chunk_overlap=st.session_state.cfg_chunk_overlap,
                    embed_model=st.session_state.cfg_embed_model,
                    index_dir=config.INDEX_DIR,
                    progress_fn=_progress,
                )

            _load_retriever.clear()
            retriever_obj = _load_retriever(
                config.INDEX_DIR,
                st.session_state.cfg_embed_model,
                st.session_state.cfg_top_k,
                st.session_state.cfg_reranking,
            )
            st.session_state.rag_chain = ret.build_rag_chain(
                retriever_obj, st.session_state.cfg_ollama_model
            )
            st.session_state.index_built = True
            st.session_state.chunk_count = n
            progress_bar.empty()
            st.success(f"✓  {n} chunks from {len(uploaded_files)} file(s)")

    if st.session_state.index_built:
        st.metric("Indexed Chunks", st.session_state.chunk_count)

    st.divider()

    # Configuration
    st.markdown('<div class="rb-label">Configuration</div>', unsafe_allow_html=True)
    with st.expander("Model & Retrieval Settings"):
        st.session_state.cfg_embed_model = st.selectbox(
            "Embedding Model",
            ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "multi-qa-MiniLM-L6-cos-v1"],
            index=["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "multi-qa-MiniLM-L6-cos-v1"].index(
                st.session_state.cfg_embed_model
            ),
        )
        st.session_state.cfg_ollama_model = st.selectbox(
            "LLM  (Ollama)",
            ["llama3.2", "llama3.1", "mistral", "gemma2:9b", "phi3"],
            index=["llama3.2", "llama3.1", "mistral", "gemma2:9b", "phi3"].index(
                st.session_state.cfg_ollama_model
            ) if st.session_state.cfg_ollama_model in ["llama3.2", "llama3.1", "mistral", "gemma2:9b", "phi3"] else 0,
        )
        st.session_state.cfg_chunk_size = st.slider(
            "Chunk Size (chars)", 128, 1024,
            st.session_state.cfg_chunk_size, step=64,
        )
        st.session_state.cfg_chunk_overlap = st.slider(
            "Chunk Overlap", 0, 256,
            st.session_state.cfg_chunk_overlap, step=32,
        )
        st.session_state.cfg_top_k = st.slider(
            "Top-K Chunks", 1, 10,
            st.session_state.cfg_top_k,
        )
        st.session_state.cfg_reranking = st.toggle(
            "Cross-Encoder Reranking",
            value=st.session_state.cfg_reranking,
        )

        if st.button("Apply & Reload", use_container_width=True):
            if st.session_state.index_built:
                _load_retriever.clear()
                retriever_obj = _load_retriever(
                    config.INDEX_DIR,
                    st.session_state.cfg_embed_model,
                    st.session_state.cfg_top_k,
                    st.session_state.cfg_reranking,
                )
                st.session_state.rag_chain = ret.build_rag_chain(
                    retriever_obj, st.session_state.cfg_ollama_model
                )
                st.success("✓  Retriever reloaded")
            else:
                st.info("Build an index first.")

    st.divider()

    # Actions
    st.markdown('<div class="rb-label">Session</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.session_state.messages:
            chat_export = "\n\n".join(
                f"**{m['role'].upper()}**: {m['content']}"
                for m in st.session_state.messages
            )
            st.download_button(
                "Export",
                data=chat_export.encode(),
                file_name=f"ragbase_{datetime.now():%Y%m%d_%H%M}.md",
                mime="text/markdown",
                use_container_width=True,
            )

    st.divider()

    # Status
    if st.session_state.index_built:
        status_html = (
            f'<div class="rb-status">'
            f'<div class="rb-status-dot on"></div>'
            f'<span class="rb-status-label on">Index Ready</span>'
            f'<span class="rb-model-chip">{st.session_state.cfg_ollama_model}</span>'
            f'</div>'
        )
    else:
        status_html = (
            '<div class="rb-status">'
            '<div class="rb-status-dot off"></div>'
            '<span class="rb-status-label">Awaiting Index</span>'
            '</div>'
        )
    st.markdown(status_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

st.markdown("""
<div class="rb-title-wrap">
    <div class="rb-title-row">
        <span class="rb-title">RAGBASE</span>
        <span class="rb-badge">v1.0 · Local</span>
    </div>
</div>
<div class="rb-subtitle">
    <span>offline</span> · <span>private</span> · <span>open-source</span>
    &nbsp;·&nbsp; semantic document intelligence
</div>
""", unsafe_allow_html=True)

st.divider()

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        sources = msg.get("sources", [])
        if sources:
            with st.expander(f"◈  {len(sources)} source chunk{'s' if len(sources) != 1 else ''}"):
                for s in sources:
                    loc_html = f'<div class="rb-source-loc">@ {s["loc"]}</div>' if s["loc"] else ""
                    st.markdown(
                        f'<div class="rb-source">'
                        f'<div class="rb-source-path">'
                        f'<i class="rb-source-icon">◈</i>{s["filename"]}'
                        f'</div>'
                        f'{loc_html}'
                        f'<div class="rb-source-snippet">{s["snippet"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# Chat input
if prompt := st.chat_input("Query your documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not st.session_state.index_built:
        with st.chat_message("assistant"):
            st.warning("Upload documents and build the index first (sidebar →)")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Upload documents and build the index first.",
            "sources": [],
        })
        st.stop()

    with st.chat_message("assistant"):
        is_tab, dfs = ret.is_tabular_question(prompt, config.DOCS_DIR)

        if is_tab:
            with st.spinner("Running data analysis…"):
                answer = ret.answer_tabular(
                    prompt, dfs, st.session_state.cfg_ollama_model
                )
            st.markdown(answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": [],
            })
        else:
            with st.spinner("Searching documents…"):
                result = st.session_state.rag_chain.invoke(prompt)

            answer = result.get("answer", "")
            source_docs = result.get("context", [])

            seen_keys: set = set()
            sources: list = []
            for doc in source_docs:
                key = (
                    doc.metadata.get("source", ""),
                    doc.metadata.get("loc", ""),
                )
                if key not in seen_keys:
                    seen_keys.add(key)
                    sources.append({
                        "filename": doc.metadata.get("source", "unknown"),
                        "loc": doc.metadata.get("loc", ""),
                        "snippet": doc.page_content[:200].replace("\n", " "),
                    })

            st.markdown(answer)
            if sources:
                with st.expander(f"◈  {len(sources)} source chunk{'s' if len(sources) != 1 else ''}"):
                    for s in sources:
                        loc_html = f'<div class="rb-source-loc">@ {s["loc"]}</div>' if s["loc"] else ""
                        st.markdown(
                            f'<div class="rb-source">'
                            f'<div class="rb-source-path">'
                            f'<i class="rb-source-icon">◈</i>{s["filename"]}'
                            f'</div>'
                            f'{loc_html}'
                            f'<div class="rb-source-snippet">{s["snippet"]}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })
