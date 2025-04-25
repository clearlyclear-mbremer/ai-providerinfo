import os
import shutil
from langchain_community.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

persist_dir = "./chroma_store"

# Load fresh Confluence docs
loader = ConfluenceLoader(
    url=os.environ["CONFLUENCE_URL"],
    username=os.environ["CONFLUENCE_USERNAME"],
    api_key=os.environ["CONFLUENCE_API_KEY"],
    space_key=os.environ["CONFLUENCE_SPACE_KEY"],
    limit=50
)
docs = loader.load()

if not docs:
    print("⚠️ No documents were loaded from Confluence.")
else:
    print(f"✅ Loaded {len(docs)} Confluence documents")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Delete only the index so data is re-built without residuals
    if os.path.exists(persist_dir):
        for f in os.listdir(persist_dir):
            file_path = os.path.join(persist_dir, f)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    # Embed documents and reinitialize the store
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    print("✅ Confluence documents refreshed and embedded.")
