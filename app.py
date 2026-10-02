import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="For Krish ❤️",
    page_icon="💗",
    layout="wide"
)

html_file = Path("index.html")
html = html_file.read_text(encoding="utf-8")

st.components.v1.html(
    html,
    height=1000,
    scrolling=True
)