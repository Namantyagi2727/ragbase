"""
app.py — Streamlit entry point for RAGBase.

Run with:  streamlit run app.py
Public showcase (read-only, recorded runs):  RAGBASE_PUBLIC_SHOWCASE=1 streamlit run app.py

The two experiences are separate modules. The showcase never imports the
ingestion, retrieval-model or inference code, so it cannot upload, index or
generate, whatever the browser sends.
"""

import streamlit as st

import config
from ui import theme

st.set_page_config(
    page_title="RAGBase",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)
theme.apply()   # Neural Terminal stylesheet (ui/theme.css)

if config.PUBLIC_SHOWCASE:
    from ui import showcase

    showcase.main()
else:
    from ui import local_app

    local_app.main()
