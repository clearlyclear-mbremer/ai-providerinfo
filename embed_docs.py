import os
import shutil
import requests

from langchain_community.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

store_dir = "./chroma_store"

# Step 1: Fully delete the old DB
if os.path.exists(store_dir):
    shutil.rmtree(store_dir)

# Step 2: Load Confluence docs
loader = ConfluenceLoader(
    url=os.environ["CONFLUENCE_URL"],
    username=os.environ["CONFLUENCE_USERNAME"],
    api_key=os.environ["CONFLUENCE_API_KEY"],
    space_key=os.environ["CONFLUENCE_SPACE_KEY"],
    limit=50
)
docs = loader.load()
print(f"✅ Loaded {len(docs)} Confluence pages")

# Step 3: Chunk docs
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Step 4: Rebuild the vector store from scratch
embeddings = OpenAIEmbeddings()
Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=store_dir
)

print(f"✅ Rebuilt Chroma vector store with {len(chunks)} chunks")

# After embedding, call the /refresh endpoint
refresh_url = os.getenv("REFRESH_URL", "https://ai-providerinfo.onrender.com/refresh")  # Default to localhost if not set
try:
    response = requests.post(refresh_url)
    if response.status_code == 200:
        print("✅ Successfully refreshed vector store in the running app!")
    else:
        print(f"⚠️ Refresh request failed with status code: {response.status_code}, message: {response.text}")
except Exception as e:
    print(f"❌ Failed to send refresh request: {e}")
