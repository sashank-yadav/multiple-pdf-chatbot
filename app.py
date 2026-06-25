import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import Chroma

load_dotenv()

st.set_page_config(page_title="Multiple PDF Chatbot")

st.title("📚 Multiple PDF Chatbot")
st.write("Upload multiple PDFs and ask questions about them.")

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    all_docs = []

    for uploaded_file in uploaded_files:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        all_docs.extend(docs)

    st.success("PDFs Loaded Successfully!")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(all_docs)

    st.write(f"Number of Chunks: {len(chunks)}")

    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
    )

    vectorstore = Chroma.from_documents(
        chunks,
        embeddings
    )

    st.success("Vector Database Created!")

    question = st.text_input("Ask a question about your PDFs")

    if question:

        docs = vectorstore.similarity_search(
            question,
            k=3
        )

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
        Answer the question using ONLY the context below.

        Context:
        {context}

        Question:
        {question}
        """

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0
        )

        try:
            response = llm.invoke(prompt)

            st.subheader("Answer")
            st.write(response.content)

        except Exception as e:
            st.error(f"Error: {e}")