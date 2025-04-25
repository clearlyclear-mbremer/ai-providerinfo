import os
from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from chromadb.config import Settings as ChromaSettings

# Initialize Flask app
app = Flask(__name__)

# Initialize empty vectorstore and chain
vectordb = None
qa_chain = None

def load_vectorstore():
    global vectordb, qa_chain
    print("🔄 Loading vectorstore...")
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(
        persist_directory="./chroma_store",
        embedding_function=embeddings,
        client_settings=ChromaSettings(
            anonymized_telemetry=False,
            allow_reset=True,
        )
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4"),
        retriever=vectordb.as_retriever(search_kwargs={"k": 3})
    )
    print("✅ Vectorstore and QA chain loaded.")

@app.route("/ask", methods=["POST"])
def ask():
    global vectordb, qa_chain
    if vectordb is None or qa_chain is None:
        load_vectorstore()

    data = request.get_json()
    query = data.get("question")
    if not query:
        return jsonify({"error": "Missing question"}), 400

    try:
        answer = qa_chain.run(query)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/refresh", methods=["POST"])
def refresh():
    global vectordb, qa_chain
    load_vectorstore()
    return jsonify({"status": "refreshed"})

# Optional: Health check
@app.route("/", methods=["GET"])
def healthcheck():
    return "API is running!", 200
