import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import faiss
import pickle
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

DATA_DIR = "data/"
VECTOR_STORE_DIR = "vector_store/"
EMBEDDING_DIM = 768  # Gemini's embedding size

def read_text_files():
    text = ""
    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)
        if file.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text += f.read() + "\n"
        elif file.endswith(".pdf"):
            pdf = PdfReader(path)
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def embed_text(chunks):
    embeddings = []
    valid_chunks = []

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            print(f"⚠️ Skipping empty chunk at index {i}")
            continue
        try:
            response = genai.embed_content(
                model="models/embedding-001",
                content=chunk,
                task_type="retrieval_document"
            )
            embedding = response.get("embedding")
            if embedding:
                embeddings.append(embedding)
                valid_chunks.append(chunk)
            else:
                print(f"⚠️ No embedding returned for chunk {i}")
        except Exception as e:
            print(f"❌ Error at chunk {i}: {e}")

    return valid_chunks, embeddings



def save_vector_store(chunks, vectors):
    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype("float32"))
    
    faiss.write_index(index, f"{VECTOR_STORE_DIR}/index.faiss")
    with open(f"{VECTOR_STORE_DIR}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

def build():
    text = read_text_files()
    chunks = chunk_text(text)
    chunks, vectors = embed_text(chunks)
    if not vectors:
        print("❌ No embeddings generated. Aborting.")
        return
    save_vector_store(chunks, vectors)
    print("✅ Vector store built with Gemini embeddings.")


if __name__ == "__main__":
    import numpy as np
    build()
