import os
import re

from dotenv import load_dotenv
from qdrant_client import models
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# ==========================================
# ENVIRONMENT
# ==========================================

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY n'est pas définie dans le fichier .env")


# ==========================================
# CONFIGURATION
# ==========================================

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "invoices"
TOP_K = 6
HISTORY_TURNS = 3  # nombre d'échanges gardés en mémoire

# LangChain stocke les métadonnées sous la clé "metadata" dans Qdrant
FILE_KEY = "metadata.source_file"
INVOICE_KEY = "metadata.invoice_number"


# ==========================================
# MODELS & VECTOR STORE
# ==========================================

llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0)  # clé lue depuis l'env

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME,
)

client = vector_store.client


# ==========================================
# PROMPT
# ==========================================

prompt = ChatPromptTemplate.from_template(
    """
You are an assistant that answers questions about invoices.

Answer ONLY using the information provided in the context.
Do not invent information.
Answer in the same language as the question.

If the answer cannot be found in the context, say exactly:
"I don't know based on the provided invoices."

Previous conversation:
{history}

Question:
{question}

Context:
{context}

Answer:
"""
)


# ==========================================
# HELPERS
# ==========================================

def list_files():
    """Retourne la liste triée des PDF présents dans la collection."""
    files = set()
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            meta = (point.payload or {}).get("metadata", {})
            name = meta.get("source_file")
            if name:
                files.add(name)

        if offset is None:
            break

    return sorted(files)


def build_filter(selected_file, invoice_number):
    """Construit le filtre Qdrant (PDF choisi et/ou numéro de facture)."""
    conditions = []

    if selected_file:
        conditions.append(
            models.FieldCondition(
                key=FILE_KEY,
                match=models.MatchValue(value=selected_file),
            )
        )

    if invoice_number:
        conditions.append(
            models.FieldCondition(
                key=INVOICE_KEY,
                match=models.MatchValue(value=invoice_number),
            )
        )

    return models.Filter(must=conditions) if conditions else None


def build_context(documents):
    parts = []

    for doc in documents:
        m = doc.metadata
        parts.append(
            f"""Source: {m.get("source_file", "unknown")}
Invoice number: {m.get("invoice_number", "unknown")}
Date: {m.get("date", "unknown")}
Discount: {m.get("discount", "unknown")}
Total: {m.get("total", "unknown")}
Shipping: {m.get("shipping", "unknown")}
Ship to: {m.get("ship_to", "unknown")}
Bill to: {m.get("bill_to", "unknown")}
Invoice content:
{doc.page_content}"""
        )

    return "\n\n---\n\n".join(parts)


def format_history(history):
    if not history:
        return "(none)"
    return "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history)


def choose_file(files, arg):
    """Sélectionne un PDF par numéro ou par nom (partiel accepté)."""
    if arg.isdigit() and 1 <= int(arg) <= len(files):
        return files[int(arg) - 1]

    matches = [f for f in files if arg.lower() in f.lower()]
    return matches[0] if len(matches) == 1 else None


def print_files(files, selected):
    print()
    for i, f in enumerate(files, 1):
        mark = "  <== sélectionné" if f == selected else ""
        print(f"  {i}. {f}{mark}")
    print()


def print_help():
    print()
    print("Commandes :")
    print("  /files          liste des PDF disponibles")
    print("  /use <n|nom>    choisir le PDF sur lequel poser les questions")
    print("  /all            interroger tous les PDF")
    print("  /clear          effacer la mémoire de la conversation")
    print("  exit            quitter")
    print()


# ==========================================
# START CHAT
# ==========================================

files = list_files()

if not files:
    raise SystemExit("Aucun fichier trouvé dans la collection Qdrant.")

selected_file = None
history = []

print()
print("==========================================")
print("           INVOICE RAG CHAT")
print("==========================================")

print_help()
print_files(files, selected_file)
print("Aucun PDF sélectionné : toutes les factures sont interrogées.")
print("Utilise /use <numéro> pour cibler un PDF précis.")
print()

while True:

    scope = selected_file if selected_file else "tous les PDF"
    question = input(f"[{scope}] Question : ").strip()

    if not question:
        continue

    # ---------------- commandes ----------------

    if question.lower() == "exit":
        print("\nChat terminé.")
        break

    if question.lower() == "/help":
        print_help()
        continue

    if question.lower() == "/files":
        print_files(files, selected_file)
        continue

    if question.lower() == "/all":
        selected_file = None
        history.clear()
        print("\nTous les PDF sont maintenant interrogés.\n")
        continue

    if question.lower() == "/clear":
        history.clear()
        print("\nMémoire effacée.\n")
        continue

    if question.lower().startswith("/use"):
        arg = question[4:].strip()
        chosen = choose_file(files, arg) if arg else None

        if chosen:
            selected_file = chosen
            history.clear()
            print(f"\nPDF sélectionné : {selected_file}\n")
        else:
            print("\nFichier introuvable. Tape /files pour voir la liste.\n")
        continue

    # ---------------- retrieval ----------------

    match = re.search(r"\b\d{4,6}\b", question)
    invoice_number = match.group() if match else None

    query_filter = build_filter(selected_file, invoice_number)

    documents = vector_store.similarity_search(
        question,
        k=TOP_K,
        filter=query_filter,
    )

    # Si le numéro détecté n'existe pas (ex: c'était une année ou un montant),
    # on retente sans ce filtre pour ne pas bloquer la question.
    if not documents and invoice_number:
        documents = vector_store.similarity_search(
            question,
            k=TOP_K,
            filter=build_filter(selected_file, None),
        )

    if not documents:
        print("\nI don't know based on the provided invoices.\n")
        continue

    # ---------------- génération ----------------

    try:
        messages = prompt.format_messages(
            question=question,
            context=build_context(documents),
            history=format_history(history),
        )
        answer = llm.invoke(messages).content

    except Exception as e:
        print(f"\nErreur Groq : {e}\n")
        continue

    history.append((question, answer))
    history = history[-HISTORY_TURNS:]

    print()
    print("------------------------------------------")
    print("Réponse :")
    print("------------------------------------------")
    print(answer)

    print("\nSources utilisées :")
    for src in sorted({d.metadata.get("source_file", "unknown") for d in documents}):
        print("-", src)
    print()
