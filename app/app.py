import os
import subprocess
import threading
import time
import json

from flask import Flask, request, jsonify, render_template
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from chromadb.config import Settings as ChromaSettings

# Initialize Flask app
app = Flask(__name__, template_folder="../templates")

vectordb = None
qa_chain = None
store_lock = threading.Lock()

def get_current_collection():
    """Get the latest collection name from file."""
    try:
        with open("./collection_name.json", "r") as f:
            data = json.load(f)
            return data["collection_name"]
    except Exception as e:
        print(f"❌ Failed to load collection_name.json: {e}")
        return None

def load_vectorstore():
    global vectordb, qa_chain
    with store_lock:
        print("🔄 Loading vectorstore...")

        # Close old
        if vectordb:
            print("🧹 Closing old vectorstore...")
            vectordb._client.reset()
            vectordb = None
            qa_chain = None

        if not os.path.exists("./chroma_store") or not os.listdir("./chroma_store"):
            print("⚠️ Chroma store missing or empty. Skipping load.")
            return

        collection_name = get_current_collection()
        if not collection_name:
            print("⚠️ No valid collection found.")
            return

        embeddings = OpenAIEmbeddings()
        vectordb = Chroma(
            persist_directory="./chroma_store",
            collection_name=collection_name,
            embedding_function=embeddings,
            client_settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-4"),
            retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        )
        print(f"✅ Vectorstore and QA chain loaded using collection: {collection_name}")

def async_embed_docs():
    try:
        print("🚀 Running embed_docs.py asynchronously...")
        subprocess.run(
            [
                "/opt/render/project/src/.venv/bin/python",
                "/opt/render/project/go/src/github.com/clearlyclear-mbremer/ai-providerinfo/embed_docs.py"
            ],
            check=True
        )
        print("✅ embed_docs.py completed asynchronously.")

        load_vectorstore()
        print("✅ Vectorstore refreshed after re-embedding.")

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run embed_docs.py asynchronously: {e}")

threading.Thread(target=async_embed_docs, daemon=True).start()
load_vectorstore()

@app.route("/ask", methods=["POST"])
def ask():
    global vectordb, qa_chain

    data = request.get_json()
    query = data.get("question")
    if not query:
        return jsonify({"error": "Missing question"}), 400

    with store_lock:
        if vectordb is None or qa_chain is None:
            print("⚠️ Vectorstore missing, reloading...")
            load_vectorstore()

        try:
            answer = qa_chain.run(query)
            return jsonify({"answer": answer})
        except Exception as e:
            print(f"❌ Error during ask: {e}")
            return jsonify({"error": str(e)}), 500

@app.route("/ask_ui", methods=["GET"])
def ask_ui():
    return render_template("index.html")

@app.route("/refresh", methods=["POST"])
def refresh():
    print("🔄 Manual refresh requested!")
    load_vectorstore()
    return jsonify({"status": "refreshed"})

@app.route("/confluence-webhook", methods=["POST"])
def confluence_webhook():
    try:
        print("🔔 Received webhook event!")
        threading.Thread(target=async_embed_docs, daemon=True).start()
        return jsonify({"status": "refresh triggered"}), 200
    except Exception as e:
        print(f"❌ Error processing webhook: {e}")
        return jsonify({"status": "error", "details": str(e)}), 500

@app.route("/", methods=["GET"])
def healthcheck():
    return "API is running!", 200
