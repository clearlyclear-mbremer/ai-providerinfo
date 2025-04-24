import os
from langchain_community.document_loaders import ConfluenceLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# === Load config from environment ===
CONFLUENCE_URL = os.environ["CONFLUENCE_URL"]
CONFLUENCE_USERNAME = os.environ["CONFLUENCE_USERNAME"]
CONFLUENCE_API_KEY = os.environ["CONFLUENCE_API_KEY"]
SPACE_KEY = os.environ.get("CONFLUENCE_SPACE_KEY", "CPI")  # fallback default if not set
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))

# === Initialize loader ===
print(f"Connecting to Confluence space: {SPACE_KEY} ...")
loader = ConfluenceLoader(
    url=CONFLUENCE_URL,
    username=CONFLUENCE_USERNAME,
    api_key=CONFLUENCE_API_KEY,
    space_key=SPACE_KEY,
    limit=20  # future-proof: set here, not in .load()
)

# === Load pages ===
docs = loader.load()
print(f"✅ Loaded {len(docs)} pages from Confluence.")

# === Split documents ===
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
chunks = splitter.split_documents(docs)
print(f"✅ Split into {len(chunks)} chunks.")

# === Embed and persist ===
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_store"
)

print("✅ Confluence documents embedded and stored successfully.")
