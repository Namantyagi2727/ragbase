"""ui/theme.py — The RAGBase "Neural Terminal" stylesheet (ui/theme.css), applied once per page run."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import streamlit as st

CSS_PATH = Path(__file__).resolve().parent / "theme.css"


@lru_cache(maxsize=1)
def stylesheet() -> str:
    return CSS_PATH.read_text()


def apply() -> None:
    st.markdown(f"<style>\n{stylesheet()}</style>", unsafe_allow_html=True)
