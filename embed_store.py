from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import pickle
import os
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

with open("split_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

os.environ["HF_TOKEN"]= os.getenv("HF_TOKEN")
embedding_model= HuggingFaceEmbeddings(model_name= "all-MiniLM-L6-v2")

os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
vectorstore =FAISS.from_documents(chunks,embedding_model)

FAISS.save_local(vectorstore,"faiss_db")

print("FAISS DB built and saved ")
