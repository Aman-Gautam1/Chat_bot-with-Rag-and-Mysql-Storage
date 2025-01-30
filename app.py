import pickle
import mysql.connector
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import T5Tokenizer, T5ForConditionalGeneration
import datetime
from flask import Flask, request, jsonify

app = Flask(__name__)

# ------------------ Load Pickle Files ------------------
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

with open("embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

# ------------------ Initialize ChromaDB ------------------
client = chromadb.Client()
collection = client.create_collection("Indianhistory")

# Add embeddings to ChromaDB if not already added
if len(collection.get()["ids"]) == 0:
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=[chunk.page_content for chunk in chunks],
        embeddings=embeddings
    )

# ------------------ Load Models ------------------
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # For query embedding
tokenizer = T5Tokenizer.from_pretrained("t5-small")
qa_model = T5ForConditionalGeneration.from_pretrained("t5-small")

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
        "INSERT INTO chat_history (timestamp, role, content) VALUES ( %s, %s,%s)",
        (datetime.datetime.now(), "user", query),
    )
    cursor.execute(
        "INSERT INTO chat_history (timestamp, role, content) VALUES (%s, %s, %s)",
        (datetime.datetime.now(), "system", answer),
    )
    connection.commit()
    cursor.close()
    connection.close()


# ------------------ Retrieval Function ------------------
# def retrival_top_k(query, k=4):
#     query_embedding = embedding_model.encode([query])
#     results = collection.query(query_embeddings=query_embedding, n_results=k)
#     top_k_chunks = results['documents']
#     return top_k_chunks

def retrival_top_k(query, k=4):

    query_embedding = embedding_model.encode([query])  # Pass the query as a list

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k  # Number of results to retrieve
    )

    # Extract the top-k documents (context chunks)
    top_k_chunks = results['documents']
    return top_k_chunks

# ------------------ Answer Generation ------------------
# def generate_answer(query, retrieved_chunks):
#     context = " ".join(retrieved_chunks)
#     input_text = f"question: {query} context: {context}"
#     inputs = tokenizer(input_text, return_tensors="pt", max_length=500, truncation=True)
#     outputs = qa_model.generate(**inputs, max_length=550)
#     answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return answer

def generate_answer(query, retrieved_chunks):
    retrieved_chunks = [item for sublist in retrieved_chunks for item in sublist]
    context = " ".join(retrieved_chunks)
    input_text = f"question: {query} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=500, truncation=True)
    outputs = qa_model.generate(**inputs, max_length=550)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# ------------------ Flask Endpoints ------------------

@app.route('/',methods=['Get'])
def get_answer():
    return jsonify("hiiii")
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    # Retrieve the top K relevant chunks based on the user's query
    top_k_chunks = retrival_top_k(query)

    # Generate an answer based on the retrieved chunks
    answer = generate_answer(query, top_k_chunks)

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
