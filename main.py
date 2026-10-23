from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(Path(__file__).resolve().parent / ".env")

docs = TextLoader("document loaders/notes.txt", encoding="utf-8").load()

template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI that summarizes text."),
    ("human", "{data}"),
])

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

prompt = template.format_prompt(data=docs[0].page_content)
result = model.invoke(prompt)

print(result.content)