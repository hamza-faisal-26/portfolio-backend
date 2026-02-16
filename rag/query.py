"""
Query Module - Handle RAG queries with Groq Cloud LLM
"""
import os
from groq import Groq
from typing import List, Tuple

import re

from .embeddings import get_embedding_store

# Initialize Groq client
_groq_client = None


def _get_groq_client() -> Groq:
    """Get or create the Groq client."""
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def rewrite_query(query: str) -> str:
    """
    Rewrite user query to replace pronouns with 'Hamza' so that
    embedding search matches the knowledge base content.
    E.g. "tell me about your AI projects" -> "tell me about Hamza's AI projects"
    """
    # Replace possessive pronouns first, then subject/object pronouns
    replacements = [
        (r'\byour\b', "Hamza's"),
        (r'\byours\b', "Hamza's"),
        (r'\byou\b', "Hamza"),
        (r'\bhe\b', "Hamza"),
        (r'\bhis\b', "Hamza's"),
        (r'\bhim\b', "Hamza"),
    ]
    rewritten = query
    for pattern, replacement in replacements:
        rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)
    return rewritten


def retrieve_context(query: str, top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Retrieve relevant context from the knowledge base.

    Args:
        query: User's question
        top_k: Number of relevant chunks to retrieve

    Returns:
        List of (chunk, score) tuples
    """
    store = get_embedding_store()
    return store.search(query, top_k=top_k)


def format_context(results: List[Tuple[str, float]]) -> str:
    """Format retrieved chunks into a context string."""
    if not results:
        return "No relevant information found."

    context_parts = []
    for chunk, score in results:
        context_parts.append(chunk)

    return "\n\n".join(context_parts)


def generate_response(query: str, model: str = "llama-3.1-8b-instant") -> str:
    """
    Generate a response using RAG with Groq Cloud API.

    Args:
        query: User's question
        model: Groq model to use (llama-3.1-8b-instant is free and fast)

    Returns:
        Generated response string
    """
    # Rewrite query to replace pronouns with "Hamza"
    search_query = rewrite_query(query)

    # Retrieve relevant context using the rewritten query
    results = retrieve_context(search_query, top_k=3)
    context = format_context(results)

    # Build the prompt
    system_prompt = """You are Mohammad Hamza's AI assistant on his portfolio website.
Answer questions about Hamza using ONLY the provided context. 

Be friendly and helpful. Keep responses short — maximum 1-2 sentences.
Do NOT use any markdown formatting whatsoever. No bold (**), no italics (*), no backticks, no headings (#). Respond in plain text only.
If the context contains relevant info, always use it to answer."""

    user_prompt = f"""Context about Hamza:
{context}

User Question: {query}

Please answer the question based on the context provided. Be helpful and conversational."""

    try:
        # Call Groq API
        client = _get_groq_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=150,
            temperature=0.7,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}. Please try again later."


def check_groq_connection(model: str = "llama-3.1-8b-instant") -> bool:
    """Check if Groq API is reachable and the API key is valid."""
    try:
        client = _get_groq_client()
        # Make a minimal test request
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )
        return True
    except Exception as e:
        print(f"Groq connection error: {e}")
        return False


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # Test query
    print("Testing Groq connection...")
    if check_groq_connection():
        print("✓ Groq API is connected!")

        # Test query (would need embeddings initialized)
        query = "What are Hamza's skills?"
        print(f"\nQuery: {query}")
        print("(Note: Run with full initialization for actual results)")
    else:
        print("✗ Groq API is not available")
