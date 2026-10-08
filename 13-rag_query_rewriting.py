import os

os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "my_first_rag"


llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
)

prompt = ChatPromptTemplate.from_template(
    '''
    You are a helpful assistant
    yo ualways answer the question given the context
    if you dn't have an answer just say i don't know

    question : 
    {question}

    context:
    {context}
    '''
)

rewrite_prompt = ChatPromptTemplate.from_template("""
    Rewrite the user's question into a concise search query.

    Do not answer the question.
    Keep the important keywords.

    Question:
    {question}

    Search query:
""")


# 1. Load the same embedding model

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Connect to the existing Qdrant collection

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME,
)

retriever = vector_store.as_retriever(
    search_kwargs = {
        "k" : 5
    }
)

def format_docs(documents):
    return "\n\n".join(doc.page_content for doc in documents)


rewrite_chain = rewrite_prompt | llm | StrOutputParser()

rag_chain = (
    {
        "context" : rewrite_chain| retriever | format_docs,
        "question" : RunnablePassthrough()
    }
    | prompt |llm |StrOutputParser()

)

question = input('Ask something : ')
answer = rag_chain.invoke(question)

print('AI : ', answer)