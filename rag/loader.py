"""
Document Loader - Loads and chunks the knowledge base for RAG
"""
import os
from typing import List


def load_document(file_path: str) -> str:
    """Load a text document from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks for better retrieval.
    
    Args:
        text: The full text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of overlapping characters between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    
    # Split by sections (## headers) first for better semantic chunking
    sections = text.split('\n## ')
    
    for i, section in enumerate(sections):
        # Add back the ## for non-first sections
        if i > 0:
            section = '## ' + section
        
        # If section is small enough, keep it as one chunk
        if len(section) <= chunk_size:
            chunks.append(section.strip())
        else:
            # Split larger sections into smaller chunks with overlap
            words = section.split()
            current_chunk = []
            current_length = 0
            
            for word in words:
                current_chunk.append(word)
                current_length += len(word) + 1  # +1 for space
                
                if current_length >= chunk_size:
                    chunks.append(' '.join(current_chunk))
                    # Keep last few words for overlap
                    overlap_words = int(overlap / 5)  # Approximate words for overlap
                    current_chunk = current_chunk[-overlap_words:] if overlap_words > 0 else []
                    current_length = sum(len(w) + 1 for w in current_chunk)
            
            # Add remaining chunk
            if current_chunk:
                remaining = ' '.join(current_chunk)
                if remaining.strip():
                    chunks.append(remaining.strip())
    
    return [chunk for chunk in chunks if chunk.strip()]


def load_and_chunk_knowledge_base(base_path: str = None) -> List[str]:
    """
    Load the knowledge base file and return chunks.
    
    Args:
        base_path: Base directory path (defaults to current directory)
    
    Returns:
        List of text chunks
    """
    if base_path is None:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    kb_path = os.path.join(base_path, 'knowledge_base.txt')
    
    if not os.path.exists(kb_path):
        raise FileNotFoundError(f"Knowledge base not found at {kb_path}")
    
    text = load_document(kb_path)
    chunks = chunk_text(text)
    
    return chunks


if __name__ == "__main__":
    # Test the loader
    chunks = load_and_chunk_knowledge_base()
    print(f"Loaded {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
