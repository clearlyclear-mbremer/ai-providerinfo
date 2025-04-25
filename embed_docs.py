import os
import shutil

from langchain_community.document_loaders import ConfluenceLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Config
PERSIST_DIR = "./chroma_store"

# 🧹 Step 1: Delete existing Chroma DB folder
if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)

# 📥 Step 2: Load Confluence docs
loader = ConfluenceLoader(
    url=os.environ["CONFLUENCE_URL"],
    username=os.environ["CONFLUENCE_USERNAME"],
    api_key=os.environ["CONFLUENCE_API_KEY"],
    space_key=os.environ["CONFLUENCE_SPACE_KEY"],
    limit=50
)
docs = loader.load()

# 🪓 Step 3: Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 🧠 Step 4: Embed and persist
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=PERSIST_DIR)

print("✅ Chroma store refreshed and Confluence documents embedded.")
