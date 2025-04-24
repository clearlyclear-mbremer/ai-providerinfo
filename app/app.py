from flask import Flask, request, jsonify
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load persisted vector store
vectordb = Chroma(persist_directory="./chroma_store", embedding_function=OpenAIEmbeddings())

# Setup chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4"),
    retriever=vectordb.as_retriever(search_kwargs={"k": 3})
)

app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("question")
    if not query:
        return jsonify({"error": "Missing question"}), 400
    answer = qa_chain.run(query)
    return jsonify({"answer": answer})
