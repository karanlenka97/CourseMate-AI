#load pdf
#split into chunks
#create the embeddings
#store into chroma

from langchain_community.document_loaders import PyPDFLoader #pdf loader
from langchain_text_splitters import RecursiveCharacterTextSplitter #for txt files
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_community.embeddings import FastEmbedEmbeddings
load_dotenv()

data = PyPDFLoader("document loaders/deeplearning.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
chunks = splitter.split_documents(docs)


embeddings = FastEmbedEmbeddings()

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db",
)