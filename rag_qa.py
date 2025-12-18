from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA

# =========================
# EMBEDDINGS
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# LOAD FAISS INDEX
# =========================
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever()

# =========================
# LLM (HuggingFace Endpoint)
# =========================
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-base",
    temperature=0.3,
    max_new_tokens=256
)

# =========================
# RAG CHAIN
# =========================
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# =========================
# ASK FUNCTION
# =========================
def ask_question(question: str) -> str:
    return qa_chain.run(question)
