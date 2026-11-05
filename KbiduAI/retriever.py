# retriever.py
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
import os

class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = SentenceTransformer(model_name, device=self.device)
        self.index = None
        self.chunks = []

    def build_index(self, texts, chunk_size=200, overlap=50):
        self.chunks = []
        for text in texts:
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i+chunk_size]
                if chunk:
                    self.chunks.append(chunk)
        if not self.chunks:
            print("No chunks to index.")
            return
        embeddings = self.encoder.encode(self.chunks, convert_to_numpy=True, show_progress_bar=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        print(f"Index built with {len(self.chunks)} chunks.")

    def add_texts(self, new_texts, chunk_size=200, overlap=50):
        new_chunks = []
        for text in new_texts:
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i+chunk_size]
                if chunk:
                    new_chunks.append(chunk)
        if not new_chunks:
            return
        embeddings = self.encoder.encode(new_chunks, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunks.extend(new_chunks)
        print(f"Added {len(new_chunks)} chunks.")

    def save_index(self, path="faiss_index"):
        if self.index is not None:
            faiss.write_index(self.index, f"{path}.faiss")
            with open(f"{path}_chunks.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(self.chunks))
            print(f"Index saved to {path}")

    def load_index(self, path="faiss_index"):
        if not os.path.exists(f"{path}.faiss"):
            raise FileNotFoundError(f"Index file {path}.faiss not found.")
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}_chunks.txt", "r", encoding="utf-8") as f:
            self.chunks = f.read().split("\n")
        print(f"Index loaded with {len(self.chunks)} chunks.")

    def retrieve(self, query, top_k=3):
        if self.index is None:
            raise ValueError("Index not built or loaded.")
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append({"chunk": self.chunks[idx], "score": float(score)})
        return results
