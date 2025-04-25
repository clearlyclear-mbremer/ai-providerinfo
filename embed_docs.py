import os
import shutil
from langchain_community.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Set up persistence directory
persist_dir = "./chroma_store"

# Clean the existing Chroma DB to avoid stale data
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)

# Load Confluence docs
loader = ConfluenceLoader(
    url=os.environ["CONFLUENCE_URL"],
    username=os.environ["CONFLUENCE_USERNAME"],
    api_key=os.environ["CONFLUENCE_API_KEY"],
    space_key=os.environ["CONFLUENCE_SPACE_KEY"],
    limit=50  # move this into init to avoid the deprecation warning
)
docs = loader.load()

# Validate that documents are loaded
print(f"✅ Loaded {len(docs)} Confluence documents")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Embed and rebuild the vector store
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(
    chunks,
    embedding=embeddings,
    persist_directory=persist_dir
)

print("✅ Confluence documents embedded and Chroma vector store refreshed.")
