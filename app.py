import streamlit as st
from rag_qa import ask_question

st.set_page_config(page_title="Cricket RAG", layout="centered")

st.title("🏏 Cricket Players RAG QA")

question = st.text_input("Ask a question:")

if question:
    answer = ask_question(question)
    st.write("### Answer")
    st.write(answer)
