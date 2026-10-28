from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#imported text loaders

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""], #check for multiple separators
    chunk_size=10, 
    chunk_overlap=1)

data = TextLoader("document loaders/notes.txt")

docs = data.load()
chunks = splitter.split_documents(docs)

# print(len(chunks))

for i in chunks:
    print(i.page_content)
    