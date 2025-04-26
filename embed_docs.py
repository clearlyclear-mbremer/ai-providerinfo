import os
import shutil
import time
import requests
import json
from uuid import uuid4

from langchain_community.document_loaders import ConfluenceLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

chroma_dir = "./chroma_store"
temp_dir = "./chroma_store_tmp"
collection_file = "./collection_name.json"  # ⭐️ store the active collection name

def safe_delete(path, timeout=10):
    """Safely delete a directory and confirm removal."""
    if os.path.exists(path):
        shutil.rmtree(path)
        start = time.time()
        while os.path.exists(path):
            if time.time() - start > timeout:
                raise RuntimeError(f"Timeout: Directory '{path}' could not be deleted in time.")
            time.sleep(0.1)

def main():
    print("🚀 Starting document embedding and vectorstore rebuild...")

    # Step 1: Clean temp_dir
    safe_delete(temp_dir)

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

    if not docs:
        raise RuntimeError("❌ No documents fetched from Confluence!")

    # Step 3: Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"✅ Split into {len(chunks)} chunks")

    if not chunks:
        raise RuntimeError("❌ No document chunks created!")

    # Step 4: Create unique collection name
    collection_name = f"collection_{uuid4().hex}"
    print(f"🆕 Using collection: {collection_name}")

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(
        persist_directory=temp_dir,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    vectordb.add_documents(chunks)
    print(f"✅ Temporary Chroma vectorstore created with {len(chunks)} chunks")

    # Step 5: Swap
    safe_delete(chroma_dir)
    os.rename(temp_dir, chroma_dir)
    print("✅ Atomically replaced old Chroma store")

    # Step 6: Save active collection
    with open(collection_file, "w") as f:
        json.dump({"collection_name": collection_name}, f)
    print(f"✅ Saved active collection: {collection_name}")

    # Step 7: Tell app to refresh
    refresh_url = os.getenv("REFRESH_URL", "https://ai-providerinfo.onrender.com/refresh")
    try:
        response = requests.post(refresh_url)
        if response.status_code == 200:
            print("✅ Successfully refreshed vector store in the running app!")
        else:
            print(f"⚠️ Refresh request failed: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"❌ Failed to send refresh request: {e}")

if __name__ == "__main__":
    main()
