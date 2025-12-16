import streamlit as st
import pandas as pd
from src.search import search
from src.summarizer import summarize


st.set_page_config(page_title="Document Search & Summary")
st.title("📄 Document Search & Summarization using LLM")


df = pd.read_csv("data/processed/documents.csv")
documents = df['chunk'].tolist()


query = st.text_input("🔍 Enter your query")
summary_length = st.slider("Summary length", 100, 500, 200)


if st.button("Search"):
results = search(query, "embeddings/vector_store.faiss", documents)


st.subheader("Top Results")
combined = " ".join(results)
for r in results:
st.write("•", r[:200], "...")


st.subheader("Summary")
summary = summarize(combined, summary_length)
st.success(summary)
