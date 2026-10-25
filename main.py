from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader #pdf loader
from langchain_community.document_loaders import WebBaseLoader #web page loader
from langchain_text_splitters import RecursiveCharacterTextSplitter #for txt files
from langchain_core.prompts import ChatPromptTemplate

load_dotenv() 

data = PyPDFLoader("document loaders/GRU.pdf")
docs = data.load()

template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI that summarizes text."),
    ("human", "{data}"),
])

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

prompt = template.format_prompt(data=docs[0].page_content)
result = model.invoke(prompt)

print(result.content)