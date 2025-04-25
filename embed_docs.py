import os
import hashlib
import time
from langchain_community.document_loaders import ConfluenceLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

CHROMA_DIR = "./chroma_store"
MAX_DELETE_WAIT_SECONDS = 5

def compute_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

# Load Confluence documents
loader = ConfluenceLoader(
    url=os.environ["CONFLUENCE_URL"],
    username=os.environ["CONFLUENCE_USERNAME"],
    api_key=os.environ["CONFLUENCE_API_KEY"],
    space_key=os.environ["CONFLUENCE_SPACE_KEY"],
    limit=50
)
docs = loader.load()
print(f"✅ Loaded {len(docs)} Confluence pages")

# Initialize vector store
embeddings = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

# Initialize splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Get existing IDs
try:
    existing_ids = set(vectordb.get()["ids"])
except Exception:
    existing_ids = set()

chunks_to_add = []

for doc in docs:
    doc_id = doc.metadata.get("source", "") or compute_hash(doc.page_content)
    content_hash = compute_hash(doc.page_content)

    if doc_id in existing_ids:
        vectordb.delete(doc_id=doc_id)
        
        # Confirm deletion before proceeding
        start = time.time()
        while True:
            remaining = set(vectordb.get()["ids"])
            if doc_id not in remaining:
                break
            if time.time() - start > MAX_DELETE_WAIT_SECONDS:
                print(f"⚠️ Timeout waiting for deletion of {doc_id}, skipping update.")
                doc_id = None  # Prevent re-adding
                break
            time.sleep(0.5)

    if doc_id:
        split_chunks = splitter.split_documents([doc])
        for chunk in split_chunks:
            chunk.metadata["doc_id"] = doc_id
            chunk.metadata["content_hash"] = content_hash
        chunks_to_add.extend(split_chunks)

# Add to vector store
if chunks_to_add:
    vectordb.add_documents(chunks_to_add)
    print(f"✅ Added {len(chunks_to_add)} updated/new chunks")
else:
    print("⚠️ No new documents added.")
