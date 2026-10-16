"""
Streamlit Community Cloud entry point for the public RAGBase showcase.

Always runs the read-only showcase (recorded runs, curated corpus); it does
not consult any configuration that could enable uploads or live inference.
Community Cloud starts apps from the repository root; this file adds the root
to the import path so the shared ui package is found.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from ui import showcase, theme  # noqa: E402

st.set_page_config(page_title="RAGBase — showcase", page_icon="⬡", layout="wide",
                   initial_sidebar_state="auto")   # collapsed on phones, open on desktop
theme.apply()
showcase.main()
