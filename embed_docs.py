import os
import shutil
import requests
import time

from langchain_community.document_loaders import ConfluenceLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

chroma_dir = "./chroma_store"
temp_dir = "./chroma_store_tmp"

# Step 1: Clean temp_dir first
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
    timeout = 10
    start_time = time.time()
    while os.path.exists(temp_dir):
        if time.time() - start_time > timeout:
            raise RuntimeError(f"Timeout: Temporary directory '{temp_dir}' could not be deleted in time.")
        time.sleep(0.1)

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

# Step 3: Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
print(f"✅ Split into {len(chunks)} chunks")

# Step 4: Build Chroma vectorstore into TEMP directory
embeddings = OpenAIEmbeddings()
vectordb = Chroma(
    persist_directory=temp_dir,
    embedding_function=embeddings
)
vectordb.add_documents(chunks)

print(f"✅ Temporary Chroma vectorstore created with {len(chunks)} chunks")

# Step 5: Atomically swap temp_dir into chroma_dir
if os.path.exists(chroma_dir):
    shutil.rmtree(chroma_dir)
os.rename(temp_dir, chroma_dir)

print("✅ Atomically replaced old Chroma store with new one")

# Step 6: Tell the app to refresh
refresh_url = os.getenv("REFRESH_URL", "https://ai-providerinfo.onrender.com/refresh")
try:
    response = requests.post(refresh_url)
    if response.status_code == 200:
        print("✅ Successfully refreshed vector store in the running app!")
    else:
        print(f"⚠️ Refresh request failed with status code: {response.status_code}, message: {response.text}")
except Exception as e:
    print(f"❌ Failed to send refresh request: {e}")
