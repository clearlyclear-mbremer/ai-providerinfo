import os
import subprocess
import threading
import time

from flask import Flask, request, jsonify, render_template
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from chromadb.config import Settings as ChromaSettings

# Initialize Flask app
app = Flask(__name__, template_folder="../templates")

# Global variables
vectordb = None
qa_chain = None

def load_vectorstore():
    """Load the Chroma vectorstore and set up the QA chain."""
    global vectordb, qa_chain
    print("🔄 Loading vectorstore...")

    if not os.path.exists("./chroma_store") or not os.listdir("./chroma_store"):
    print("⚠️ Chroma store missing or empty. Skipping load.")
    vectordb = None
    qa_chain = None
    return

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

def async_embed_docs():
    """Run embed_docs.py asynchronously after startup."""
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
        
        # After re-embedding, reload the vectorstore
        time.sleep(1)  # slight buffer just in case
        load_vectorstore()
        print("✅ Vectorstore refreshed after re-embedding.")

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run embed_docs.py asynchronously: {e}")

# Run the background embed_docs task immediately
threading.Thread(target=async_embed_docs, daemon=True).start()

# Load the vectorstore initially (may be empty first run)
load_vectorstore()

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
        # Special handling if Chroma store is missing
        if "does not exist" in str(e):
            print("⚠️ Vectorstore was empty. Retrying load...")
            time.sleep(2)  # Wait briefly
            load_vectorstore()  # Try reloading

            try:
                answer = qa_chain.run(query)
                return jsonify({"answer": answer})
            except Exception as inner_e:
                return jsonify({"error": str(inner_e)}), 500

        return jsonify({"error": str(e)}), 500


@app.route("/ask_ui", methods=["GET"])
def ask_ui():
    """Serve the basic web UI."""
    return render_template("index.html")

@app.route("/refresh", methods=["POST"])
def refresh():
    """Manual refresh endpoint (can be triggered externally)."""
    global vectordb, qa_chain
    load_vectorstore()
    return jsonify({"status": "refreshed"})

@app.route("/confluence-webhook", methods=["POST"])
def confluence_webhook():
    """Endpoint Confluence can POST to when content is updated."""
    try:
        print("🔔 Received Confluence webhook event!")

        # Trigger embed_docs refresh in background
        threading.Thread(target=async_embed_docs, daemon=True).start()

        return jsonify({"status": "refresh triggered"}), 200
    except Exception as e:
        print(f"❌ Error processing webhook: {e}")
        return jsonify({"status": "error", "details": str(e)}), 500

@app.route("/", methods=["GET"])
def healthcheck():
    """Simple health check."""
    return "API is running!", 200

