import os
import shutil
from langchain_community.document_loaders import ConfluenceLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Delete previous store to ensure a fresh embed
persist_dir = "./chroma_store"
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)

# Load Confluence docs
loader = ConfluenceLoader(
    url=os.environ["CONFLUENCE_URL"],
    username=os.environ["CONFLUENCE_USERNAME"],
    api_key=os.environ["CONFLUENCE_API_KEY"],
    space_key=os.environ["CONFLUENCE_SPACE_KEY"],
    limit=50
)
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Embed with updated OpenAI package
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# Rebuild the Chroma DB from scratch
vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
vectordb.add_documents(chunks)

print(f"✅ Loaded {len(docs)} documents and embedded {len(chunks)} chunks into Chroma.")
