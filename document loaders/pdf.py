from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
#imported pypdf loader for loading pdf
#imported for text split for embeddings

data = PyPDFLoader("document loaders/GRU.pdf")

docs = data.load()

splitter = TokenTextSplitter(
    chunk_size=100, 
    chunk_overlap=10)

chunks = splitter.split_documents(docs)

print(chunks[0].page_content)