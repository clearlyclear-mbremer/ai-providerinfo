import os
import subprocess
import threading

from flask import Flask, request, jsonify, render_template
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from chromadb.config import Settings as ChromaSettings

# Initialize Flask app
app = Flask(__name__, template_folder="../templates")

# Global state
vectordb = None
qa_chain = None
store_lock = threading.Lock()  # 🛡️ Protect reloading with a lock

def load_vectorstore():
    """Load the Chroma vectorstore and set up the QA chain."""
    global vectordb, qa_chain
    with store_lock:
        print("🔄 Loading vectorstore...")

        # 🧹 Close old vectorstore if exists
        if vectordb:
            print("🧹 Closing old vectorstore connection...")
            vectordb._client.reset()
            vectordb = None
            qa_chain = None

        if not os.path.exists("./chroma_store") or not os.listdir("./chroma_store"):
            print("⚠️ Chroma store missing or empty. Skipping load.")
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
    """Run embed_docs.py asynchronously after startup or webhook."""
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
        
        load_vectorstore()  # Immediately reload
        print("✅ Vectorstore refreshed after re-embedding.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run embed_docs.py asynchronously: {e}")

# 🚀 Run embed_docs.py immediately at app startup
threading.Thread(target=async_embed_docs, daemon=True).start()

# 🚀 Load the vectorstore once initially
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
            print("⚠️ Vectorstore not loaded. Attempting reload...")
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
    """External manual refresh trigger."""
    print("🔄 Manual refresh requested!")
    load_vectorstore()
    return jsonify({"status": "refreshed"})

@app.route("/confluence-webhook", methods=["POST"])
def confluence_webhook():
    """Triggered when Confluence sends webhook."""
    try:
        print("🔔 Received Confluence webhook event!")
        threading.Thread(target=async_embed_docs, daemon=True).start()
        return jsonify({"status": "refresh triggered"}), 200
    except Exception as e:
        print(f"❌ Webhook error: {e}")
        return jsonify({"status": "error", "details": str(e)}), 500

@app.route("/", methods=["GET"])
def healthcheck():
    return "API is running!", 200
