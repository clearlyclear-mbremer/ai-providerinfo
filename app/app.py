import os
import subprocess
from flask import Flask, request, jsonify
from flask import render_template
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from chromadb.config import Settings as ChromaSettings

# Initialize Flask app
app = Flask(__name__, template_folder="../templates")

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

def run_embed_docs_at_startup():
    """Run embed_docs.py automatically at app startup if needed."""
    try:
        print("🚀 Running embed_docs.py at startup...")
        subprocess.run(
            [
                "/opt/render/project/src/.venv/bin/python",
                "/opt/render/project/go/src/github.com/clearlyclear-mbremer/ai-providerinfo/embed_docs.py"
            ],
            check=True
        )
        print("✅ embed_docs.py completed at startup.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run embed_docs.py at startup: {e}")

# 🧠 Actually run it at startup
run_embed_docs_at_startup()

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

@app.route("/ask_ui", methods=["GET"])
def ask_ui():
    return render_template("index.html")

@app.route("/refresh", methods=["POST"])
def refresh():
    global vectordb, qa_chain

    # Hard reset everything
    vectordb = None
    qa_chain = None

    # Reload cleanly
    load_vectorstore()

    return jsonify({"status": "refreshed"})

@app.route("/confluence-webhook", methods=["POST"])
def confluence_webhook():
    """Endpoint Confluence can POST to when content is updated."""
    try:
        print("🔔 Received Confluence webhook event!")

        # Re-run embed_docs.py when webhook received
        subprocess.run(
            [
                "/opt/render/project/src/.venv/bin/python",
                "/opt/render/project/go/src/github.com/clearlyclear-mbremer/ai-providerinfo/embed_docs.py"
            ],
            check=True
        )
        print("✅ embed_docs.py successfully refreshed after webhook.")

        return jsonify({"status": "success"}), 200
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running embed_docs.py after webhook: {e}")
        return jsonify({"status": "error", "details": str(e)}), 500


# Optional: Health check
@app.route("/", methods=["GET"])
def healthcheck():
    return "API is running!", 200
