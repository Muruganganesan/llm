import streamlit as st
from transformers import pipeline

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ===============================
# 🧠 PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Wikipedia RAG Chatbot",
    page_icon="📘",
    layout="wide"
)

st.title("📘 Wikipedia RAG Chatbot")
st.write("Ask questions based on scraped Wikipedia documents")

# ===============================
# 🔹 LOAD MODELS (CACHED)
# ===============================
@st.cache_resource
def load_rag_components():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "faiss_index",
        embedding_model,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=256
    )

    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    prompt = PromptTemplate.from_template(
        """
        Answer the question using ONLY the context below.
        If you don't know the answer, say "I don't know".

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


rag_chain = load_rag_components()

# ===============================
# 💬 CHAT HISTORY
# ===============================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display old messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ===============================
# ❓ USER INPUT
# ===============================
user_query = st.chat_input("Ask a question from Wikipedia data...")

if user_query:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🤔"):
            answer = rag_chain.invoke(user_query)
            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
