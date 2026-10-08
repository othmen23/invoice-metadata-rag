from pathlib import Path
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter



QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "my_first_rag"


# 1. Load PDF

all_documents = list()

print(os.getcwd())

folder_path = Path("data/")

for root, _, files in os.walk(folder_path):
    print('Processing folder:', root)
    for file in files:
        # Check if the file ends with .pdf (ignoring case)
        if file.lower().endswith(".pdf"):
            full_path = os.path.join(root, file)
            loader = PyPDFLoader(str(full_path))
            documents = loader.load()
            all_documents.extend(documents)

print(f"Loaded {len(all_documents)} documents.")


# 2. Split PDF into chunks

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)

chunks = splitter.split_documents(all_documents)

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