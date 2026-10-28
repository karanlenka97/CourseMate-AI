from dotenv import load_dotenv
from langchain_chroma import Chroma #import vector store
from langchain_core.documents import Document #create your own document
from langchain_google_genai import GoogleGenerativeAIEmbeddings #embedding model
from langchain_community.embeddings import HuggingFaceEmbeddings 

load_dotenv()

docs = [
    Document(
        page_content="Python is widely used in Artificial Intelligence.",
        metadata={"source": "AI_book"},
    ),
    Document(
        page_content="Pandas is used for data analysis in Python.",
        metadata={"source": "DataScience_book"},
    ),
    Document(
        page_content="Neural networks are used in deep learning.",
        metadata={"source": "DL_book"},
    ),
]

embeddings_model = HuggingFaceEmbeddings()

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings_model,
    persist_directory="chroma_db",
)

results = vectorstore.similarity_search(
    "What is used for data analysis?",
    k=2,
)

for r in results:
    print(r.page_content, r.metadata)

retriever = vectorstore.as_retriever()

docs = retriever.invoke("Explain deep learning")

for d in docs:
    print(d.page_content)