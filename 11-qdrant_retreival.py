from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "my_first_rag"

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-120b",
)

prompt = ChatPromptTemplate.from_template(
    '''
    You are a helpful assistant
    you always answer the questions given the context
    if you don't have an answer just say i don't know

    question : 
    {question}

    context:
    {context}
    '''
)
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

# 3. Ask a question

question = input("Question: ")

# 4. Retrieve the most relevant chunks
documents = vector_store.similarity_search(
    question,
    k=5,
)

context = "\n\n".join([document.page_content for document in documents])

context = "\n".join([ doc.page_content for doc in documents])

chain = prompt | llm | StrOutputParser()

answer = chain.invoke({
    'question' : question,
    'context' : context
})
print('AI : ', answer)


# # # 5. Display the results

# for i, document in enumerate(documents, start=1):

#     print(f"\n--- Result {i} ---")
#     print("Page:", document.metadata.get("page"))
#     print(document.page_content)













