from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle

loader = PyPDFDirectoryLoader("medical_docs/")
documents = loader.load()


splitter =RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=30)
chunks =splitter.split_documents(documents)

# Save to file
with open("split_chunks.pkl","wb") as f:
    pickle.dump(chunks,f)

print(f"Saved{len(chunks)}chunks")
