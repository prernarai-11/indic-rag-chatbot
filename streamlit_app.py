import streamlit as st
import os
import shutil
import fitz
from groq import Groq
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────
DATA_FOLDER = "data"
VECTOR_STORE = "vector_store"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"

# ─── FUNCTIONS ────────────────────────────────────────────

def ask_groq(prompt):
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful document assistant. Answer only from the context provided."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def expand_query(query):
    expansions = {
        "EM": "Environmental Management",
        "EMS": "Environmental Management System",
        "PPO": "Pension Payment Order",
        "AB": "Authorised Banks",
        "ABS": "Authorised Banks",
        "CG": "Central Government",
        "FAQ": "Frequently Asked Questions",
    }
    upper = query.strip().upper()
    last_word = query.strip().split()[-1].upper()

    if upper in expansions:
        return expansions[upper]
    elif last_word in expansions:
        return query.rsplit(" ", 1)[0] + " " + expansions[last_word]
    return query

def build_prompt(context, question):
    return f"""Answer the question using ONLY the context provided below.
Give a clear, complete answer in 2-3 sentences.
If the answer is not in the context, say exactly: "I could not find relevant information."

Context:
{context}

Question:
{question}

Answer:"""

def load_pdfs_from_folder(folder):
    documents = []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            try:
                pdf = fitz.open(path)
                for i, page in enumerate(pdf):
                    text = page.get_text().strip()
                    if text:
                        documents.append(Document(
                            page_content=text,
                            metadata={"source": file, "page": i + 1}
                        ))
            except Exception as e:
                st.error(f"Failed to load {file}: {e}")
    return documents

def ingest(status_box):
    status_box.info("Loading PDFs...")
    documents = load_pdfs_from_folder(DATA_FOLDER)
    if not documents:
        status_box.error("No PDFs found in data folder.")
        return False

    status_box.info(f"Loaded {len(documents)} pages. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    status_box.info(f"Created {len(chunks)} chunks. Building vector store...")
    if os.path.exists(VECTOR_STORE):
        shutil.rmtree(VECTOR_STORE)

    embeddings = get_embeddings()
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE
    )
    status_box.success(f"Done! {len(chunks)} chunks indexed from {len(documents)} pages.")
    return True

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

@st.cache_resource
def get_db():
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=VECTOR_STORE,
        embedding_function=embeddings
    )

# ─── PAGE SETUP ───────────────────────────────────────────

st.set_page_config(
    page_title="Indic RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Indic RAG Chatbot")
st.caption("Ask questions from your uploaded documents.")

# ─── SIDEBAR ──────────────────────────────────────────────

with st.sidebar:
    st.header("📁 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        os.makedirs(DATA_FOLDER, exist_ok=True)
        for uploaded_file in uploaded_files:
            save_path = os.path.join(DATA_FOLDER, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"{len(uploaded_files)} file(s) uploaded.")

    st.divider()

    st.header("⚙️ Ingest Documents")
    ingest_btn = st.button("🔄 Build Knowledge Base", use_container_width=True)
    status_box = st.empty()

    if ingest_btn:
        success = ingest(status_box)
        if success:
            get_db.clear()

    st.divider()

    st.header("📄 Indexed Files")
    if os.path.exists(DATA_FOLDER):
        pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".pdf")]
        if pdf_files:
            for f in pdf_files:
                st.write(f"• {f}")
        else:
            st.write("No files yet.")
    else:
        st.write("No files yet.")

# ─── CHAT AREA ────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📄 Sources"):
                for src in message["sources"]:
                    st.write(f"• **{src['file']}** — page {src['page']}")

if query := st.chat_input("Ask a question from your documents..."):

    if not os.path.exists(VECTOR_STORE):
        st.warning("Please upload PDFs and click 'Build Knowledge Base' first.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    expanded = expand_query(query)

    db = get_db()
    docs = db.max_marginal_relevance_search(expanded, k=3, fetch_k=10)

    if not docs:
        answer = "I could not find relevant information."
        sources = []
    else:
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        prompt = build_prompt(context, expanded)

        with st.spinner("Thinking..."):
            answer = ask_groq(prompt)

        sources = [
            {
                "file": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "?")
            }
            for doc in docs
        ]

    with st.chat_message("assistant"):
        st.markdown(answer)
        if sources:
            with st.expander("📄 Sources"):
                for src in sources:
                    st.write(f"• **{src['file']}** — page {src['page']}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })