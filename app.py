#============STEP 1: lOAD MODULES===============#
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st
import numpy
import time
from PIL import Image

#============STEP 2: API KEY==================#
st.set_page_config(page_title = "Chat-with-pdf".
                   layout = "wide")

st.sidebar.title("SET API CONFIG")
st.title("RAG Based chat with PDF 📚")
GOOGLE_API_KEY = st.sidebar_text_input("GOOGLE_API_KEY",type = "password")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# =============STEP 3: LOAD PDF================#
uploaded_file = st.sidebar.file_uploaded("Upload PDF File", type = ["pdf"])

if uploaded_file:
    with st.spinner("Reading PDF File"):
        data = uploaded_file.read()
    st.sidebar.pdf(data)

# =================STEP 4: LOAD RESOURCES=================
@st.cache_data
def load_documents():
    loader = PyPDFLoader(uploaded_file)
    documents = loader.load()
    return documents

# st.cache_data: to load data only one time
# st.cache_resource : to load resource only one time

@st.cache_resource
def load_embedding():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

@st.cache_data
def get_splitted_chunks():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    return chunks

#================STEP 5: GET and LOAD DOCS=================
documents = load_documents()
embeddings = load_embedding()
chunks = get_splitted_chunks()

@st.cache_data
def create_vector_db(chunks,embeddings):
    # To Build Vector DB
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

@st.cache_data
def create_retriever(vectorstore, k_value):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_value})
    return retriever

vectorstore = create_vector_db(chunks,embeddings)
k_slider = st.sidebar.slider("Select Top K-Value",min_value = 1, max_value = 10)
retriever = create_retriever(vectorstore, k_slider)

#================STEP 6: LCEL RAG CHAIN=================
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know based on the document."

Context:
{context}

Question: {question}
""")

def format_docs(docs):
    # Join chunks of retrieved docs
    return "\n\n".join(doc.page_content for doc in docs)

with st.spinner("Building RAG Chain"):
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
