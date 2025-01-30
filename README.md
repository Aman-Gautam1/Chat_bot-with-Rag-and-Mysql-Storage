# Chatbot with RAG and MySQL Storage

## Overview

This project implements a chatbot using the Retrieval-Augmented Generation (RAG) approach. It retrieves relevant chunks from a precomputed vector database and uses a T5 model for generating responses. The chat history is stored in a MySQL database.

## Requirements

- Python 3.9
- Flask
- Transformers
- Sentence Transformers
- Torch
- MySQL

## Installation & Setup

### 1. Clone the Repository

```
git clone https://github.com/your-repo/chatbot_project.git
cd chatbot_project
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Set Up MySQL Database

- Start MySQL server.
- Run the SQL script to create the required tables:
  ```
  mysql -u root -p < db_setup.sql
  ```
- Update `app.py` with your MySQL credentials.

### 4. Run the Flask API

```
python app.py
```

## Usage

### 1. Send a Chat Query

- **POST /chat**
- Example:
  ```json
  { "query": "Who was Gandhi?" }
  ```
- Response:
  ```json
  { "answer": "Mahatma Gandhi was a leader in the Indian independence movement..." }
  ```

### 2. Retrieve Chat History

- **GET /history**
- Example:
  ```
  http://localhost:5000/history
  ```
- Response:
  ```json
  { "history": [ { "role": "user", "content": "Who was Gandhi?" }, { "role": "system", "content": "Mahatma Gandhi was..." } ] }
  ```

## Notes

- Ensure embeddings and chunks are precomputed and saved as `embeddings.pkl` and `chunks.pkl`.
- Uses **T5-small** for text generation.
- Uses **MySQL** for storing chat history.

## (Optional) Docker Setup

To containerize the application:

1. Build the Docker image:
   ```
   docker build -t chatbot .
   ```
2. Run the container:
   ```
   docker run -p 5000:5000 chatbot
   ```

This will expose the API on port 5000

