import os
import pickle

import faiss
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_DIR = "vector_store/saved_index"


def create_vector_store(chunks):
    """
    Create embeddings for document chunks and store them in FAISS.
    """

    if not chunks:
        raise ValueError("No document chunks were provided.")

    # Load the embedding model
    model = SentenceTransformer(MODEL_NAME)

    # Extract text from chunks
    texts = [chunk["text"] for chunk in chunks]

    # Create embeddings
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Convert embeddings to float32 for FAISS
    embeddings = embeddings.astype("float32")

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)

    # Add embeddings
    index.add(embeddings)

    # Create storage directory
    os.makedirs(INDEX_DIR, exist_ok=True)

    # Save FAISS index
    index_path = os.path.join(INDEX_DIR, "index.faiss")
    faiss.write_index(index, index_path)

    # Save chunks and metadata
    chunks_path = os.path.join(INDEX_DIR, "chunks.pkl")

    with open(chunks_path, "wb") as file:
        pickle.dump(chunks, file)

    print("Vector store created successfully.")
    print(f"Number of vectors: {index.ntotal}")
    print(f"Vector dimension: {dimension}")

    return index


def load_vector_store():
    """
    Load the saved FAISS index and chunks.
    """

    index_path = os.path.join(INDEX_DIR, "index.faiss")
    chunks_path = os.path.join(INDEX_DIR, "chunks.pkl")

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            "FAISS index not found. Please create the vector store first."
        )

    if not os.path.exists(chunks_path):
        raise FileNotFoundError(
            "Chunk metadata not found. Please create the vector store first."
        )

    index = faiss.read_index(index_path)

    with open(chunks_path, "rb") as file:
        chunks = pickle.load(file)

    return index, chunks


def search_documents(query, top_k=5):
    """
    Search the FAISS vector store for relevant document chunks.
    """

    if not query or not query.strip():
        return []

    # Load saved vector store
    index, chunks = load_vector_store()

    # Load embedding model
    model = SentenceTransformer(MODEL_NAME)

    # Create query embedding
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Convert to float32
    query_embedding = query_embedding.astype("float32")

    # Do not request more results than available
    top_k = min(top_k, index.ntotal)

    # Search FAISS
    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index_number in zip(
        scores[0],
        indices[0]
    ):

        if index_number == -1:
            continue

        chunk = chunks[index_number].copy()

        chunk["score"] = float(score)

        results.append(chunk)

    return results