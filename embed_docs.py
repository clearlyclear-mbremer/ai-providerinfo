import os
from langchain_community.document_loaders import ConfluenceLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load Confluence docs
loader = ConfluenceLoader(
    url=os.environ["CONFLUENCE_URL"],
    username=os.environ["CONFLUENCE_USERNAME"],
    api_key=os.environ["CONFLUENCE_API_KEY"],
    space_key=os.environ["CONFLUENCE_SPACE_KEY"]
)
docs = loader.load(limit=50)  # bump the limit if needed

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Embed with updated OpenAI package (make sure it's installed via requirements.txt)
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# Load or create Chroma DB
vectordb = Chroma(persist_directory="./chroma_store", embedding_function=embeddings)

# Add new documents (duplicates are possible if not filtered manually)
vectordb.add_documents(chunks)

print("Confluence documents updated and embedded.")
