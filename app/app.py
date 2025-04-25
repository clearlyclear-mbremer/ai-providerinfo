import os
from flask import Flask, request, jsonify
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Initialize Flask app FIRST
app = Flask(__name__)

# Load persisted vector store
vectordb = Chroma(persist_directory="./chroma_store", embedding_function=OpenAIEmbeddings())

# Setup chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4"),
    retriever=vectordb.as_retriever(search_kwargs={"k": 3})
)

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
    global vectordb, qa_chain
    vectordb = Chroma(persist_directory="./chroma_store", embedding_function=OpenAIEmbeddings())
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4"),
        retriever=vectordb.as_retriever(search_kwargs={"k": 3})
    )
    return jsonify({"message": "Vector store refreshed from disk!"})
