"""
Embeddings - Generate and search vector embeddings for RAG
Uses sentence-transformers for local embeddings (no API key needed)
"""
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer


class EmbeddingStore:
    """Simple in-memory vector store for document embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding store with a sentence transformer model.
        
        Args:
            model_name: HuggingFace model name for embeddings
                       all-MiniLM-L6-v2 is fast and lightweight (~80MB)
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.chunks: List[str] = []
        self.embeddings: np.ndarray = None
        print("Embedding model loaded successfully!")
    
    def add_documents(self, chunks: List[str]) -> None:
        """
        Add document chunks to the store and compute embeddings.
        
        Args:
            chunks: List of text chunks to embed
        """
        self.chunks = chunks
        print(f"Computing embeddings for {len(chunks)} chunks...")
        self.embeddings = self.model.encode(chunks, show_progress_bar=True)
        print(f"Embeddings computed! Shape: {self.embeddings.shape}")
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Search for the most relevant chunks for a query.
        
        Args:
            query: The search query
            top_k: Number of top results to return
        
        Returns:
            List of (chunk, similarity_score) tuples
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        
        # Encode the query
        query_embedding = self.model.encode([query])[0]
        
        # Compute cosine similarity
        similarities = self._cosine_similarity(query_embedding, self.embeddings)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return chunks with scores
        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(similarities[idx])))
        
        return results
    
    def _cosine_similarity(self, query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and document vectors."""
        # Normalize vectors
        query_norm = query_vec / np.linalg.norm(query_vec)
        doc_norms = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
        
        # Compute dot product (cosine similarity for normalized vectors)
        return np.dot(doc_norms, query_norm)


# Global embedding store instance
_store: EmbeddingStore = None


def get_embedding_store() -> EmbeddingStore:
    """Get or create the global embedding store."""
    global _store
    if _store is None:
        _store = EmbeddingStore()
    return _store


def initialize_store(chunks: List[str]) -> EmbeddingStore:
    """Initialize the global store with document chunks."""
    store = get_embedding_store()
    store.add_documents(chunks)
    return store


if __name__ == "__main__":
    # Test embeddings
    test_chunks = [
        "Mohammad Hamza is a software engineer at Siemens.",
        "He studied at IIT Hyderabad with a CGPA of 8.90.",
        "His skills include Python, C++, and machine learning.",
        "He worked at iCIMS India as a software engineering intern."
    ]
    
    store = EmbeddingStore()
    store.add_documents(test_chunks)
    
    query = "What is Hamza's education?"
    results = store.search(query, top_k=2)
    
    print(f"\nQuery: {query}")
    for chunk, score in results:
        print(f"  Score: {score:.4f} | {chunk[:80]}...")
