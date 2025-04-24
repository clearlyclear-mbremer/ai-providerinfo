import os
import shutil
from langchain_community.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Step 1: Clear old vector store
persist_path = "./chroma_store"
if os.path.exists(persist_path):
    shutil.rmtree(persist_path)

# Step 2: Load docs from Confluence
loader = ConfluenceLoader(
    url=os.environ["CONFLUENCE_URL"],
    username=os.environ["CONFLUENCE_USERNAME"],
    api_key=os.environ["CONFLUENCE_API_KEY"],
    space_key=os.environ["CONFLUENCE_SPACE_KEY"],
    limit=50
)
docs = loader.load()

# Step 3: Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Step 4: Embed and persist
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_path)

print("✅ Confluence documents embedded fresh and saved to Chroma.")
