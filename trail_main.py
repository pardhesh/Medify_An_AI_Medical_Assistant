from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

os.environ["GROQ_API"]=os.getenv("GROQ_API")

groq_api = os.getenv("GROQ_API")


loader = PyPDFDirectoryLoader("medical_docs/")
documents = loader.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""], 
)
docs = splitter.split_documents(documents)
print(f"✅ Split into {len(docs)} total chunks.")

os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    docs,
    embedding_model,
    collection_name="medical-bot"
)
retriever = vectorstore.as_retriever()

history_aware_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

history_aware_retriever = create_history_aware_retriever(
    llm=ChatGroq(model_name="Gemma2-9b-It", groq_api_key=groq_api),
    retriever=retriever,
    prompt=history_aware_prompt
)

document_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful medical assistant. Use the following documents to answer the question."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Here are the relevant documents:\n{context}")
])

doc_chain = create_stuff_documents_chain(
    llm=ChatGroq(model_name="Gemma2-9b-It", groq_api_key=groq_api),
    prompt=document_prompt
)

retrieval_chain = create_retrieval_chain(history_aware_retriever, doc_chain)

chat_with_history = RunnableWithMessageHistory(
    retrieval_chain,
    lambda session_id: ChatMessageHistory(), 
    input_messages_key="input",
    history_messages_key="chat_history",
)

st.title("🩺 Medical Assistant Chatbot")


session_id = "user-session"

user_input = st.text_input("Ask a question:")

if user_input:
    response = chat_with_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    st.write("🤖:", response["answer"] if isinstance(response, dict) else response)
