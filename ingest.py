import os
import fitz  # pymupdf
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def load_pdfs(folder):
    documents = []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            try:
                pdf = fitz.open(path)
                for i, page in enumerate(pdf):
                    text = page.get_text().strip()
                    if text:  # skip empty pages
                        documents.append(Document(
                            page_content=text,
                            metadata={"source": file, "page": i + 1}
                        ))
                print(f"Loaded: {file} — {len(pdf)} pages")
            except Exception as e:
                print(f"Failed: {file} — {e}")
    return documents

print("\nLoading PDFs...")
documents = load_pdfs("data")
print(f"Total pages loaded: {len(documents)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
chunks = splitter.split_documents(documents)
print(f"Total chunks: {len(chunks)}")

print("\nLoading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

print("Building vector store...")
Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="vector_store"
)

print("\nDone! Vector store ready.\n")