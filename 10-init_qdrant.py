from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_PATH = Path("data/1-s2.0-S1877050925029461-main.pdf")

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "my_first_rag"


# 1. Load PDF

loader = PyPDFLoader(str(PDF_PATH))
documents = loader.load()

print(f"Loaded {len(documents)} pages.")


# 2. Split PDF into chunks

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")


# 3. Create embedding model

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 4. Create Qdrant collection and insert chunks

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME,
)

print("PDF successfully stored in Qdrant.")