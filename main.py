from dotenv import load_dotenv
from langchain_chroma import Chroma #import vector store
from langchain_community.embeddings import HuggingFaceEmbeddings #embedding model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import FastEmbedEmbeddings

load_dotenv()

