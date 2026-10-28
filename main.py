from dotenv import load_dotenv
from langchain_chroma import Chroma #import vector store
from langchain_community.embeddings import HuggingFaceEmbeddings #embedding model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import FastEmbedEmbeddings

load_dotenv()

embeddings = FastEmbedEmbeddings()

vectorstore = Chroma(
    persist_directory="chroma_db", 
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 4,
        "fetch_k":10,
        "lambda_mult" :0.5
    }
)

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

#prompt templete
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

