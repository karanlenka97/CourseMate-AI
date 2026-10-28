from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader #pdf loader
from langchain_community.document_loaders import WebBaseLoader #web page loader
from langchain_text_splitters import RecursiveCharacterTextSplitter #for txt files
from langchain_core.prompts import ChatPromptTemplate #sys,human role division

load_dotenv() 

data = PyPDFLoader("document loaders/deeplearning.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
chunks = splitter.split_documents(docs)

template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI that summarizes text."),
    ("human", "{data}"),
])

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

prompt = template.format_prompt(data=chunks[0].page_content)
results = model.invoke(prompt)

print(results.content)