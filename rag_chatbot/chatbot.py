import os
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

VECTOR_STORE_DIR = "vector_store/"

def load_vector_store():
    index = faiss.read_index(f"{VECTOR_STORE_DIR}/index.faiss")
    with open(f"{VECTOR_STORE_DIR}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def embed_query(query):
    try:
        response = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query"
        )
        return response["embedding"]
    except Exception as e:
        print(f"❌ Error generating query embedding: {e}")
        return None

def get_relevant_context(query_embedding, index, chunks, k=4):
    query_vector = np.array([query_embedding]).astype("float32")
    D, I = index.search(query_vector, k)
    return [chunks[i] for i in I[0]]

def generate_answer(context_chunks, user_query):
    prompt = (
        "You are AskUni, an intelligent assistant for university students.\n"
        "Use the following context to answer the user's question as clearly and helpfully as possible.\n\n"
        "Context:\n" + "\n---\n".join(context_chunks) +
        f"\n\nUser: {user_query}\nBot:"
    )
    try:
        model = genai.GenerativeModel(model_name="models/gemini-2.5-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error generating answer: {e}"


def chat_loop():
    index, chunks = load_vector_store()
    print("🧠 AskUni Gemini Chatbot (type 'exit' to quit)")

    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break
        query_embedding = embed_query(query)
        if query_embedding is None:
            print("Bot: Sorry, I couldn't understand your query.")
            continue
        context = get_relevant_context(query_embedding, index, chunks)
        answer = generate_answer(context, query)
        print("Bot:", answer)

if __name__ == "__main__":
    chat_loop()
