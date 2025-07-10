import os
import streamlit as st
import pickle
import time

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    from dotenv import load_dotenv
    load_dotenv()


st.set_page_config(page_title="EduBot: Academic Paper Research", page_icon="📚", layout="wide")

# Custom CSS for Premium UI
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #141e30, #243b55);
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        font-size: 2.8em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
        color: #00c6ff;
    }
    .subtitle {
        font-size: 1.1em;
        text-align: center;
        color: #eeeeee;
        margin-bottom: 30px;
    }
    .stButton > button {
        background-color: #00c6ff;
        color: #000000;
        border-radius: 8px;
        padding: 0.6em 1.4em;
        font-weight: bold;
        border: none;
        transition: background 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #7df9ff;
    }
    .stTextInput > div > div > input {
        border-radius: 8px;
        padding: 0.5em;
    }
    .stSidebar {
        background-color: rgba(0,0,0,0);
    }
    .footer {
        position: relative;
        text-align: center;
        font-size: 0.9em;
        margin-top: 50px;
        color: #bbbbbb;
    }
    .stHeader, .stSubheader, .stMarkdown, .stExpander {
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)


# Title & Subtitle
st.markdown('<div class="title">EduBot: Academic Research Assistant 📚</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Summarize & Query Research Papers Instantly</div>', unsafe_allow_html=True)

# Sidebar Inputs (No logo)
with st.sidebar:
    st.subheader("📑 Enter Research Paper URLs")
    urls = []
    for i in range(3):
        url = st.text_input(f"🔗 Paper URL {i+1}")
        urls.append(url)
    process_url_clicked = st.button("🚀 Process Papers")

file_path = "edu_faiss_store.pkl"
progress_placeholder = st.empty()
llm = ChatOpenAI(temperature=0.2, max_tokens=1500, model_name="gpt-3.5-turbo")

if process_url_clicked:
    with st.spinner("Fetching and processing papers..."):
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()
        progress_placeholder.progress(33, "Loaded Papers ✅")

        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=3000
        )
        docs = text_splitter.split_documents(data)
        progress_placeholder.progress(66, "Split Text ✅")

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        progress_placeholder.progress(100, "Vectorstore Built ✅")

        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)

        st.success("✅ Papers Processed Successfully!")

query = st.text_input("💬 Ask Your Academic Question:")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        chain = RetrievalQA.from_chain_type(
            llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), return_source_documents=True
        )

        with st.spinner("Generating Answer..."):
            result = chain.invoke({"query": query})

        st.header("🔎 Answer")
        st.success(result["result"])

        source_docs = result.get("source_documents", [])
        unique_sources = set()
        for doc in source_docs:
            source = doc.metadata.get("source", "")
            if source:
                unique_sources.add(source)

        if unique_sources:
            with st.expander("📂 Sources Used"):
                for src in unique_sources:
                    st.write(src)
        else:
            st.info("No sources found for this answer.")

# Footer
st.markdown('<div class="footer">🚀 Created by PKK</div>', unsafe_allow_html=True)
