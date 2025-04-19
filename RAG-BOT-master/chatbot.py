import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

def load_pdfs_from_folder(folder_path):
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    all_docs = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        loader = PyPDFLoader(pdf_path)
        data = loader.load()
        all_docs.extend(data)
    return all_docs

st.title("CYBER SECURITY STUDY ASSISTANT - BY FIELD CISO ADVISORY")

folder_path = "files"
# Load and read the PDF file "linear.pdf" using PyPDFLoader.
# loader = PyPDFLoader(load_pdfs_from_folder(folder_path))
data = load_pdfs_from_folder(folder_path)

# Split the loaded PDF data into smaller chunks using RecursiveCharacterTextSplitter.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
docs = text_splitter.split_documents(data)

# FAISS vector store from the split documents using embeddings from Google Generative AI.
vectorstore = FAISS.from_documents(documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Define a language model (LLM) using Google Generative AI model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)

# Set up the chat history variable using session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Input: User's message
query = st.chat_input("Say something: ")

if query:
    # Append the user's message to the history
    st.session_state.history.append({"role": "user", "content": query})

    # Create the prompt for the assistant
    system_prompt = (
        "You are a question-answering assistant. Use the retrieved context to answer the user's question. "
        "If unsure, say you don't know. be more explainatory and provide more details. "
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Perform the retrieval and response generation
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query})

    # Append the assistant's response to the history
    st.session_state.history.append({"role": "assistant", "content": response["answer"]})

# Optionally display the conversation history
if st.session_state.history:
    for message in st.session_state.history:
        if message["role"] == "user":
            st.write(f"You: {message['content']}")
        else:
            st.write(f"Assistant: {message['content']}")
