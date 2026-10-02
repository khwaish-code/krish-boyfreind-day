import streamlit as st
from pathlib import Path
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Happy Boyfriend Day ❤️",
    layout="wide"
)

html_file = Path(__file__).parent / "index.html"

if not html_file.exists():
    st.error("❌ index.html file not found!")
    st.write("Make sure index.html is uploaded in the same folder as app.py.")
else:
    html = html_file.read_text(encoding="utf-8")
    components.html(html, height=900, scrolling=True)