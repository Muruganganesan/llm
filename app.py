import streamlit as st
import pandas as pd
from src.search import search
from src.summarizer import summarize

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
    min_value=100,
    max_value=500,
    value=200,
    step=50
)

# -------------------------------
# Search Button Logic
# -------------------------------
if st.button("Search"):
    if query.strip() == "":
        st.warning("⚠️ Please enter a query before searching.")
    else:
        # ---- Search ----
        results = search(
            query=query,
            index_path="embeddings/vector_store.faiss",
            documents=documents,
            top_k=5
        )

        # ---- Show Results ----
        st.subheader("🔎 Top Relevant Results")

        combined_text = ""

        for i, r in enumerate(results, start=1):
            st.markdown(f"**{i}.** {r[:200]}...")
            combined_text += " " + r

        # ---- Summarization ----
        st.subheader("📌 Summary")

        with st.spinner("Summarizing using LLM..."):
            summary = summarize(combined_text, summary_length)

        st.success(summary)
