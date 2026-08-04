# ============STEP 1: LOAD MODULES=============
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st

#=================STEP 2: API KEYS=================
st.set_page_config(page_title = "Chat-With-PDF", layout = "wide")
st.sidebar.title("SET API CONFIG")
st.title("RAG Based Chat With PDF 📄")
GOOGLE_API_KEY = st.sidebar.text_input("GOOGLE_API_KEY", type = "password")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

#================STEP 3: LOAD PDF=================
uploaded_file = st.sidebar.file_uploader("Upload PDF File", type = ["pdf"])

#================STEP 4: LOAD RESOURCES=================
@st.cache_resource  # cache_resource for embeddings because it's heavy to load
def load_embedding():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_data  # data is fine for documents/chunks
def load_documents(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    return documents

@st.cache_data
def get_split_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    return chunks

@st.cache_resource  # vectorstore is also a resource
def create_vector_db(_chunks, _embeddings):  # note the _ to tell streamlit not to hash these
    vectorstore = FAISS.from_documents(_chunks, _embeddings)
    return vectorstore

@st.cache_resource
def create_retriever(_vectorstore, k_value):
    retriever = _vectorstore.as_retriever(search_kwargs={"k": k_value})
    return retriever

if uploaded_file:
    st.sidebar.success("PDF Uploaded Successfully")
    documents = load_documents(uploaded_file)
    chunks = get_split_chunks(documents)
    
    k_slider = st.sidebar.slider("Select Top K-Value", min_value = 1, max_value = 10, value=3)
    
    embeddings = load_embedding()
    vectorstore = create_vector_db(chunks, embeddings)  # passing with _ in func def
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
        return "\n\n".join(doc.page_content for doc in docs)

    with st.spinner("Building RAG Chain"):
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    # ===============GET USER INPUT================
    user_question = st.text_area("Ask Question: ")
    if user_question:
        if st.button("Get Answer"):
            with st.spinner("Generating Answer..."):
                answer = rag_chain.invoke(user_question)
                st.markdown(answer)
