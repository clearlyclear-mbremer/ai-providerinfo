import os
from langchain_community.document_loaders import ConfluenceLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load Confluence docs
loader = ConfluenceLoader(
    url=os.environ["CONFLUENCE_URL"],
    username=os.environ["CONFLUENCE_USERNAME"],
    api_key=os.environ["CONFLUENCE_API_KEY"],
    space_key=os.environ["CONFLUENCE_SPACE_KEY"],
    limit=50
)
docs = loader.load()
print(f"✅ Loaded {len(docs)} Confluence pages")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Embed with updated OpenAI package
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# Recreate the vector store in a fresh Chroma instance
Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_store"
)

print(f"✅ Rebuilt Chroma vector store with {len(chunks)} fresh chunks")
