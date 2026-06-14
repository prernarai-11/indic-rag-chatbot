# 🤖 Indic RAG Chatbot

A multilingual document question-answering chatbot built using Retrieval Augmented Generation (RAG). Upload your PDF documents and ask questions in natural language — the chatbot finds the most relevant information and generates accurate, context-grounded answers.

🔗 **Live Demo:** https://indic-rag-chatbot-gty65zbaajgchep4e7ykcg.streamlit.app/

---

## 📌 What it does

- Upload any PDF document
- Ask questions in natural language
- Get accurate answers grounded in your documents
- See the exact source page the answer came from
- Supports multilingual documents (Hindi, Marathi, English)

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| PDF Extraction | PyMuPDF (fitz) |
| Text Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | HuggingFace paraphrase-multilingual-MiniLM-L12-v2 |
| Vector Database | ChromaDB |
| LLM (Cloud) | Groq API — LLaMA 3.3 70B |
| LLM (Local) | Ollama — Gemma 3 1B |
| Retrieval | Max Marginal Relevance (MMR) Search |

---

## 🏗️ Architecture
PDF Documents

↓

PyMuPDF (text extraction)

↓

RecursiveCharacterTextSplitter (chunking)

↓

HuggingFace Embeddings (text → vectors)

↓

ChromaDB (vector storage)

↓

User Question → MMR Search → Top 3 Relevant Chunks

↓

Groq API / Ollama (answer generation)

↓

Streamlit UI (display answer + sources)

---

## 📂 Project Structure
indic-rag-chatbot/

│

├── data/                  # Sample PDF documents

├── vector_store/          # Pre-built ChromaDB vector store

├── ingest.py              # PDF ingestion and vector store builder

├── app.py                 # Terminal version (uses Ollama locally)

├── streamlit_app.py       # Web UI version (uses Groq API)

└── requirements.txt       # Python dependencies

---

## 🚀 Run Locally

**1. Clone the repo**

```bash
git clone https://github.com/prernarai-11/indic-rag-chatbot.git
cd indic-rag-chatbot
```

**2. Create virtual environment**

```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Add your Groq API key**

Create a `.env` file:
GROQ_API_KEY=your_groq_api_key_here

**5. Ingest documents**

```bash
python ingest.py
```

**6. Run the app**

```bash
streamlit run streamlit_app.py
```

---

## 💻 Run Terminal Version (Ollama)

Install [Ollama](https://ollama.com) and pull the model:

```bash
ollama pull gemma3:1b
```

Then run:

```bash
python app.py
```

---

## 📄 Sample Documents Included

| Document | Domain |
|---|---|
| Environmental Management | Environment & Sustainability |
| Payment of Pension to Government Pensioners | Government & Finance |
| Data Centre Overview | Technology & Infrastructure |

---

## ✨ Features

- **MMR Retrieval** — returns diverse, relevant chunks instead of repetitive results
- **Query Expansion** — automatically expands abbreviations before searching
- **Source Tracking** — shows exact filename and page number for every answer
- **Multilingual Support** — embedding model supports Indian languages
- **Dual Mode** — cloud deployment via Groq, local deployment via Ollama

---

## 👩‍💻 Author

**Prerna Rai**
[LinkedIn](https://www.linkedin.com/in/prerna-rai-11j/) | [GitHub](https://github.com/prernarai-11)
