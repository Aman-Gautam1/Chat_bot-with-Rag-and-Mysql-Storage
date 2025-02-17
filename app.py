import pickle
import mysql.connector
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import datetime
from flask import Flask, request, jsonify

app = Flask(__name__)

# ------------------ Load Wikipedia Data ------------------
loader = WikipediaLoader(query='History of India')
docs = loader.load()

# ------------------ Split Documents ------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=70
)
clean_text = docs[0].page_content.strip()
clean_text = " ".join(clean_text.split())
chunks = splitter.split_documents(docs)
print(f"Total chunks are {len(chunks)}")

# ------------------ Initialize ChromaDB ------------------
model1 = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(chunks, model1)

# ------------------ Load Groq Model ------------------
api_key = 'your_api_key'
model = ChatGroq(model='deepseek-r1-distill-llama-70b', groq_api_key=api_key)

# ------------------ Set up Prompt ------------------
prompt = ChatPromptTemplate.from_messages(
    [
        '''
        You are an AI agent who answers the questions based on the provided context only.
        Please provide the most accurate response from the context.
        <context>
        {context}
        </context>
        question: {input}
        Provide a direct answer without any additional explanation or thinking process.
        '''
    ]
)

# ------------------ Create Retrieval Chain ------------------
document_chain = create_stuff_documents_chain(model, prompt)
retriever = vector_db.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# ------------------ Database Configuration ------------------
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "12345",
    "database": "testing",
}

def get_db_connection():
    return mysql.connector.connect(**db_config)

# ------------------ Store Chat History ------------------
def store_chat_history(query, answer):
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute(
        "INSERT INTO chat_history (timestamp, role, content) VALUES (%s, %s, %s)",
        (datetime.datetime.now(), "user", query),
    )
    cursor.execute(
        "INSERT INTO chat_history (timestamp, role, content) VALUES (%s, %s, %s)",
        (datetime.datetime.now(), "system", answer),
    )
    connection.commit()
    cursor.close()
    connection.close()

# ------------------ Query Output Function ------------------
def query_output(retrieval_chain, query):
    response = retrieval_chain.invoke({'input': query})
    return response['answer']

def output(response):
    if '</think>' in response:
        start_idx = response.find('</think>') + len('</think>')
        answer = response[start_idx:].strip()  # Get the text after </think>
        return answer
    return response

# ------------------ Flask Endpoints ------------------

@app.route('/', methods=['GET'])
def get_answer():
    return jsonify("Hello! I am ready to assist you.")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    # Retrieve the top K relevant chunks based on the user's query
    top_k_chunks = retrival_top_k(query)

    # Generate an answer based on the retrieved chunks
    answer = query_output(retrieval_chain, query)
    answer = output(answer)  # Extract the content after </think> tag

    # Store the chat history (query and answer) in the database
    store_chat_history(query, answer)

    # Return the answer along with the retrieved chunks
    return jsonify({
        "query": query,
        "answer": answer,
        "retrieved_chunks": top_k_chunks
    })


@app.route('/history', methods=['GET'])
def get_history():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM chat_history ORDER BY timestamp DESC")
    history = cursor.fetchall()
    cursor.close()
    connection.close()
    return jsonify(history)

# ------------------ Run Flask App ------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
