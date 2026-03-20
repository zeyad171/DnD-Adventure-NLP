# IMPORTANT: Suppress TensorFlow before any imports
import os
import sys
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TORCH"] = "1"
sys.modules['tensorflow'] = None

"""
RAG System Package
==================
Retrieval-Augmented Generation for the D&D Adventure Game.

Components:
- config: Configuration settings
- llm_handler: LLM integration (Gemini/Ollama)
- rag_chain: Main RAG pipeline with FAISS
- rag_system: Simple wrapper interface
"""

from rag.config import config, RAGConfig
from rag.rag_system import get_rag_system, RAGSystem

__all__ = ['config', 'RAGConfig', 'get_rag_system', 'RAGSystem']
