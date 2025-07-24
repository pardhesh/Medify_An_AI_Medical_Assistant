import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import os
from dotenv import load_dotenv

load_dotenv()

from langchain.vectorstores import FAISS

load_dotenv()
groq_api = os.getenv("GROQ_API")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = FAISS.load_local("faiss_db",embedding_model,allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

print("Vectorstore loaded:",vectorstore.index.ntotal)


llm = ChatGroq(model_name="Gemma2-9b-It", groq_api_key=groq_api)

system_prompt= (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

history_prompt= ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user","{input}")
])

history_aware_retriever= create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=history_prompt
)



bot_prompt = (
    "You are a helpful medical assistant for question-answering tasks.\n"
    "Use the following pieces of retrieved context to answer the question.\n\n"
    "{context}\n\n"
    "Provide detailed answers. Include emojis for better understanding.\n"
    "Include a disclaimer: this is just for reference, consult a doctor before taking any medication."
)


doc_prompt = ChatPromptTemplate.from_messages([
    ("system", bot_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user","{input}"),
    
])

doc_chain = create_stuff_documents_chain(llm=llm, prompt=doc_prompt)
retrieval_chain = create_retrieval_chain(history_aware_retriever,doc_chain)

chat_with_history = RunnableWithMessageHistory(
    retrieval_chain,
    lambda session_id: ChatMessageHistory(),
    input_messages_key="input",
    history_messages_key="chat_history",
)

# UI
st.title("🩺 Medify - Your Medical Assistant")

session_id ="user-session"
user_input= st.text_input("Enter your medical question")

if user_input.strip(): 
    retrieved_docs =retriever.get_relevant_documents(user_input)
    
    if not retrieved_docs:
        st.warning("Couldn't find any relevant medical information. Please ask a health-related question.")
    else:
        with st.spinner("Analyzing your question..."):
            response= chat_with_history.invoke(
                {"input":user_input},
                config={"configurable":{"session_id": session_id}}
            )
        st.write("🤖: ", response["answer"] if isinstance(response, dict) else response)


st.markdown(
    "*This tool is for informational purposes only. Always consult a licensed doctor.*",
    unsafe_allow_html=True
)
