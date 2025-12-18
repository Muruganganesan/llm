import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_classic.chains import RetrievalQA

# ========================
# LOAD EMBEDDINGS
# ========================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ========================
# LOAD FAISS
# ========================
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever()

# ========================
# LLM
# ========================
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-base",
    temperature=0.3,
    max_new_tokens=256
)

# ========================
# RAG CHAIN
# ========================
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# ========================
# STREAMLIT UI
# ========================
st.title("🏏 Player RAG QA System")

question = st.text_input("Ask a question:")

if question:
    answer = qa.run(question)
    st.success(answer)
