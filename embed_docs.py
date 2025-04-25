import os
import shutil
import time

from langchain_community.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Set path
store_dir = "./chroma_store"

# Step 1: Clear the vector store with verification
if os.path.exists(store_dir):
    shutil.rmtree(store_dir)
    # Confirm deletion
    timeout = 10  # seconds
    start_time = time.time()
    while os.path.exists(store_dir):
        if time.time() - start_time > timeout:
            raise RuntimeError(f"Directory '{store_dir}' could not be deleted in time.")
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

# Step 4: Embed and persist to Chroma
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=store_dir)
print(f"✅ Added {len(chunks)} updated/new chunks")
