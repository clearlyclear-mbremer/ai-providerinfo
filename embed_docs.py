from langchain_community.document_loaders import ConfluenceLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize the Confluence loader
loader = ConfluenceLoader(
    url=os.environ["CONFLUENCE_URL"],
    username=os.environ["CONFLUENCE_USERNAME"],
    api_key=os.environ["CONFLUENCE_API_KEY"],                     # replace with your Atlassian API token
    space_key=os.environ["CONFLUENCE_SPACE_KEY"]
)

# Load documents from Confluence
docs = loader.load(limit=20)

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(docs)

# Embed and store documents using Chroma
vectordb = Chroma.from_documents(
    chunks,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_store"
)

# Since Chroma >=0.4 auto-persists, this is optional now
vectordb.persist()

print("Confluence documents embedded successfully.")
