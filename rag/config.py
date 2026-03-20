"""
RAG Configuration
=================
All configurable settings for the RAG system.

API keys should be set via environment variables:
- GEMINI_API_KEY: Your Google Gemini API key
"""

import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class RAGConfig:
    """Configuration for the RAG system"""
    
    # Document paths
    DOCS_DIR: str = os.getenv("DOCS_DIR", "docs")
    RAG_KNOWLEDGE_FILE: str = os.getenv("RAG_KNOWLEDGE_FILE", "game_rules_rag.txt")
    SUPPORTED_EXTENSIONS: tuple = (".md", ".txt")
    
    # Chunking settings (improved)
    CHUNK_SIZE: int = 400  # Smaller chunks for better precision
    CHUNK_OVERLAP: int = 50
    SEPARATORS: List[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " "])
    
    # Embedding settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"
    
    # Vector store settings
    VECTOR_STORE_TYPE: str = "faiss"  # "faiss" or "chroma"
    VECTOR_STORE_PATH: str = "./vector_db"
    COLLECTION_NAME: str = "game_docs"
    
    # Retrieval settings
    TOP_K: int = 3
    SCORE_THRESHOLD: float = 0.3  # Lower = more results
    
    # LLM settings
    LLM_TYPE: str = "gemini"  # "gemini", "ollama", "local"
    GEMINI_API_KEY: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    GEMINI_MODEL: str = "gemini-1.5-flash"
    
    
    # Response settings
    MAX_RESPONSE_LENGTH: int = 200
    TEMPERATURE: float = 0.7


# Global config instance
config = RAGConfig()
