# 🩺 Medify – Your AI Medical Assistant

Medify is a privacy-respecting, PDF-aware medical chatbot that answers healthcare-related questions based on uploaded medical documents. It uses LangChain with FAISS vector store, HuggingFace embeddings, and a Groq-hosted LLM to provide informed responses — while maintaining chat history context.

---

## 🚀 Features

- 🔍 **Context-aware Q&A** from medical PDFs (e.g., hypertension, pregnancy, chronic illness).
- 💬 **Chat history support** – maintains conversation thread.
- 📄 **PDF document ingestion** with vector embedding via HuggingFace.
- 🧠 **Gemma2-9B-IT model** served via Groq API for fast responses.
- 🗃️ **FAISS vector store** for fast similarity-based retrieval.
- ⚠️ **Medical disclaimer** in every answer for safety.

---

## 🛠️ Tech Stack

| Component            | Technology Used               |
|---------------------|-------------------------------|
| LLM                 | [Gemma2-9B-IT via Groq](https://console.groq.com) |
| Vector Store        | FAISS                         |
| Embeddings          | HuggingFace (`all-MiniLM-L6-v2`) |
| RAG Framework       | LangChain                     |
| UI                  | Streamlit                     |
| Doc Parsing         | LangChain PDF Loader          |

---

## 📁 Project Structure

```bash
Medify/
│
├── medical_docs/                 # Folder containing medical PDFs
├── split_chunks.pkl          # Pre-split chunks stored via pickle
├── faiss_db/                 # FAISS vectorstore directory (auto-created)
│
├── build_docs.py              # Extract data from the pdfs and make it to chunks
├── embed_store.py            # Script to embed chunks and store in FAISS
├── chatbot.py                # Streamlit app for querying chatbot
├── .env                      # Store GROQ_API and HF_TOKEN securely
├── requirements.txt
└── README.md
```

## 🧪 Setup Instructions
# 1. Clone the Repository
    git clone https://github.com/pardhesh/Medify_An_Ai_Medical_Assistant.git

# 2. Install Dependencies
    pip install -r requirements.txt

# 3. Prepare .env File
    Create a .env file in the root directory
    GROQ_API=your_groq_api_key
    HF_TOKEN=your_huggingface_token

# 4. Embed Documents into FAISS
    python embed_store.py

# 5. Run the Chatbot
    streamlit run chatbot.py
    Open your browser at http://localhost:8501 to chat with Medify.

## Live Link
    https://medify-ai-assistant.streamlit.app/

 ## ⚠️ Disclaimers

      This chatbot is not a replacement for a licensed healthcare professional.
    
    It should only be used for informational purposes.
    
    Always consult a doctor before making health-related decisions.


