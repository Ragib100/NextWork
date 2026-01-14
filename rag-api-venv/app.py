from fastapi import FastAPI
import chromadb
import ollama
import os

app = FastAPI()

# Initialize Chroma DB
chroma = chromadb.PersistentClient(path="./db")
collection = chroma.get_or_create_collection("docs")

# Ollama host environment variable (for documentation; library uses this internally)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

@app.post("/query")
def query(q: str):
    # Query Chroma for context
    results = collection.query(query_texts=[q], n_results=1)
    context = ""
    if results["documents"] and results["documents"][0]:
        context = results["documents"][0][0]

    # Generate answer using Ollama
    try:
        answer = ollama.generate(
            model="tinyllama",
            prompt=f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer clearly and concisely:"
        )
        response = answer.get("response", "No response from Ollama")
    except Exception as e:
        response = f"Error connecting to Ollama: {str(e)}"

    return {"answer": response}