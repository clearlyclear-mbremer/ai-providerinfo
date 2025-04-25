import os
from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

app = Flask(__name__)

# Define global objects
vectordb = None
qa_chain = None

def load_vectorstore():
    global vectordb, qa_chain
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory="./chroma_store", embedding_function=embeddings)
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4"),
        retriever=vectordb.as_retriever(search_kwargs={"k": 3})
    )
    print("✅ Vector store and QA chain reloaded.")

# Initial load
load_vectorstore()

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("question")
    if not query:
        return jsonify({"error": "Missing question"}), 400
    answer = qa_chain.run(query)
    return jsonify({"answer": answer})

@app.route("/refresh", methods=["POST"])
def refresh():
    try:
        load_vectorstore()
        return jsonify({"message": "Vector store refreshed."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
