import streamlit as st
import pandas as pd
import sys
import os

# 👇 src folder path add pannrom
sys.path.append(os.path.abspath("src"))

from search import search
from summarizer import summarize

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Document Search & Summary",
    layout="wide"
)

st.title("📄 Document Search & Summarization using LLM")

# -------------------------------
# Load data
# -------------------------------
@st.cache_data
def load_documents():
    df = pd.read_csv("data/processed/documents.csv")
    return df["chunk"].tolist()

documents = load_documents()

# -------------------------------
# UI Inputs
# -------------------------------
query = st.text_input("🔍 Enter your query")
summary_length = st.slider(
    "📝 Summary length (words)",
    100, 500, 200, 50
)

# -------------------------------
# Search logic
# -------------------------------
if st.button("Search"):
    if query.strip() == "":
        st.warning("⚠️ Please enter a query")
    else:
        results = search(
            query=query,
            index_path="embeddings/vector_store.faiss",
            documents=documents,
            top_k=5
        )

        st.subheader("🔎 Top Results")
        combined_text = ""

        for i, r in enumerate(results, 1):
            st.markdown(f"**{i}.** {r[:200]}...")
            combined_text += " " + r

        st.subheader("📌 Summary")
        with st.spinner("Summarizing..."):
            summary = summarize(combined_text, summary_length)

        st.success(summary)
