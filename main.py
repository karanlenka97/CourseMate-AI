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

print("Rag system created ")

print("press 0 to exit ")

while True:
    query = input("\nYou: ")

    if query == "0":
        print("Exiting the program.")
        break

    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    response = llm.invoke(
        prompt.format_prompt(context=context, question=query).to_messages()
    )

    print("\nAnswer:\n")
    print(f"\n AI: {response.content}")