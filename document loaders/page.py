from langchain_community.document_loaders import WebBaseLoader
#webpage loading module

url = "https://www.apple.com/in/macbook-pro/"

data = WebBaseLoader(url)

docs = data.load()

print(docs[0].page_content)