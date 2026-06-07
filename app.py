import requests
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def ask_ollama(prompt, model="gemma3:1b"):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        return json.loads(response.text)["response"]
    except requests.exceptions.ConnectionError:
        return "Error: Ollama is not running. Start it first."
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
        expanded = expansions[upper]
        print(f"(Expanded: '{query}' → '{expanded}')")
        return expanded
    elif last_word in expansions:
        expanded = query.rsplit(" ", 1)[0] + " " + expansions[last_word]
        print(f"(Expanded: '{query}' → '{expanded}')")
        return expanded
    return query

def build_prompt(context, question):
    return f"""You are a helpful document assistant.
Answer the question using ONLY the context provided below.
Give a clear, complete answer in 2-3 sentences.
If the answer is not in the context, say exactly: "I could not find relevant information."

Context:
{context}

Question:
{question}

Answer:"""

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

print("Loading vector store...")
db = Chroma(
    persist_directory="vector_store",
    embedding_function=embeddings
)

print("\nChatbot Ready! Type 'exit' to quit.\n")

while True:

    query = input("\nAsk Question: ").strip()

    if query.lower() == "exit":
        print("Bye!")
        break

    if not query:
        print("Please enter a question.")
        continue

    query = expand_query(query)

    docs = db.max_marginal_relevance_search(query, k=3, fetch_k=10)

    if not docs:
        print("\nNo relevant documents found.")
        continue

    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    prompt = build_prompt(context, query)

    print("\nSources:")
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        print(f"  [{i+1}] {source} — page {page}")

    print("\nAnswer:")
    answer = ask_ollama(prompt)
    print(answer)
    print("\n" + "-" * 40)