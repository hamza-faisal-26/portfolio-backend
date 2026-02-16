"""
FastAPI Backend for Portfolio RAG Chatbot
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from rag.loader import load_and_chunk_knowledge_base
from rag.embeddings import initialize_store, get_embedding_store
from rag.query import generate_response, check_groq_connection


# Request/Response models
class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    response: str
    success: bool = True


class HealthResponse(BaseModel):
    status: str
    groq_connected: bool
    knowledge_base_loaded: bool
    chunks_count: int


# Startup/shutdown lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG components on startup."""
    print("=" * 50)
    print("🚀 Starting Portfolio RAG Backend...")
    print("=" * 50)
    
    # Check Groq connection
    print("\n📡 Checking Groq API connection...")
    groq_ok = check_groq_connection()
    if groq_ok:
        print("✓ Groq API is connected and ready!")
    else:
        print("⚠ Groq API is not available - check your API key")
    
    # Load knowledge base
    print("\n📚 Loading knowledge base...")
    try:
        chunks = load_and_chunk_knowledge_base()
        print(f"✓ Loaded {len(chunks)} chunks from knowledge base")
        
        # Initialize embeddings
        print("\n🧠 Initializing embeddings (this may take a moment)...")
        initialize_store(chunks)
        print("✓ Embeddings ready!")
        
    except FileNotFoundError as e:
        print(f"⚠ Knowledge base not found: {e}")
    except Exception as e:
        print(f"⚠ Error loading knowledge base: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Backend is ready! API available at http://localhost:8000")
    print("📖 API docs at http://localhost:8000/docs")
    print("=" * 50 + "\n")
    
    yield
    
    # Cleanup on shutdown
    print("\n👋 Shutting down backend...")


# Create FastAPI app
app = FastAPI(
    title="Portfolio RAG Chatbot API",
    description="RAG-powered chatbot for Mohammad Hamza's portfolio",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with welcome message."""
    return {
        "message": "Welcome to Hamza's Portfolio RAG Chatbot API",
        "docs": "/docs",
        "health": "/health",
        "chat": "/chat (POST)"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    store = get_embedding_store()
    chunks_count = len(store.chunks) if store.chunks else 0
    
    return HealthResponse(
        status="healthy",
        groq_connected=check_groq_connection(),
        knowledge_base_loaded=chunks_count > 0,
        chunks_count=chunks_count
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Chat endpoint - send a query and get a RAG-powered response.
    
    The query is processed through:
    1. Semantic search in the knowledge base
    2. Context retrieval
    3. LLM generation with Ollama
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Generate response using RAG
        response = generate_response(request.query)
        
        return ChatResponse(
            response=response,
            success=True
        )
    
    except Exception as e:
        print(f"Chat error: {e}")
        return ChatResponse(
            response=f"I encountered an error processing your request. Please try again.",
            success=False
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
