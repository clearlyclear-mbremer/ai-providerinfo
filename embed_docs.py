import os
from langchain_community.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings  # NEW: updated import

# Initialize the Confluence loader
loader = ConfluenceLoader(
    url=os.environ["CONFLUENCE_URL"],
    username=os.environ["CONFLUENCE_USERNAME"],
    api_key=os.environ["CONFLUENCE_API_KEY"],
    space_key=os.environ["CONFLUENCE_SPACE_KEY"]
)

# Load documents (updated: init config instead of `load(limit=...)`)
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Embed and store using Chroma (auto-persisted)
vectordb = Chroma.from_documents(
    chunks,
    embedding=OpenAIEmbeddings(),  # Updated import
    persist_directory="./chroma_store"
)

print("✅ Confluence documents embedded successfully.")

