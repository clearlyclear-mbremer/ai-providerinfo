import os
import shutil
import requests
import time

from langchain_community.document_loaders import ConfluenceLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Delete old Chroma store if it exists to ensure full refresh
chroma_dir = "./chroma_store"

if os.path.exists(chroma_dir):
    shutil.rmtree(chroma_dir)
    # Confirm deletion before proceeding
    timeout = 10  # seconds
    start_time = time.time()
    while os.path.exists(chroma_dir):
        if time.time() - start_time > timeout:
            raise RuntimeError(f"Timeout: Directory '{chroma_dir}' could not be deleted in time.")
        time.sleep(0.1)  # sleep 100ms and retry


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
for idx, doc in enumerate(docs):
    print(f"\n---- Document {idx+1} ----")
    print(doc.page_content)

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
print(f"✅ Split into {len(chunks)} chunks")
for idx, chunk in enumerate(chunks):
    print(f"\n---- Chunk {idx+1} ----")
    print(chunk.page_content)

# Embed with updated OpenAI package (make sure it's installed via requirements.txt)
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# Load or create Chroma DB
vectordb = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)

# Add new documents (duplicates are possible if not filtered manually)
vectordb.add_documents(chunks)

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
