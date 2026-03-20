"""
LLM Handler
===========
Handles LLM initialization and response generation.
Supports Gemini API (primary) and Ollama (local fallback).
"""

import re
import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class LLMHandler:
    """
    Handles LLM initialization and inference.
    
    Primary: Google Gemini API
    Fallback: Ollama local models
    """
    
    def __init__(self, config):
        self.config = config
        self.llm = None
        self.llm_type = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the LLM based on configuration."""
        if self.config.LLM_TYPE == "gemini":
            self._init_gemini()
        elif self.config.LLM_TYPE == "ollama":
            self._init_ollama()
        else:
            logger.warning(f"Unknown LLM type: {self.config.LLM_TYPE}")
    
    def _init_gemini(self):
        """Initialize Google Gemini API."""
        import sys
        try:
            print(f"[LLM] Initializing Gemini API...", flush=True)
            sys.stdout.flush()
            import google.generativeai as genai
            
            if not self.config.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not set in environment")
            
            print(f"[LLM] Configuring Gemini with API key...", flush=True)
            sys.stdout.flush()
            genai.configure(api_key=self.config.GEMINI_API_KEY)
            
            print(f"[LLM] Creating GenerativeModel: {self.config.GEMINI_MODEL}...", flush=True)
            sys.stdout.flush()
            self.llm = genai.GenerativeModel(self.config.GEMINI_MODEL)
            self.llm_type = "gemini"
            
            print(f"[LLM] Gemini LLM initialized successfully!", flush=True)
            sys.stdout.flush()
            logger.info(f"Gemini LLM initialized: {self.config.GEMINI_MODEL}")
            
        except Exception as e:
            print(f"[LLM ERROR] Failed to initialize Gemini: {e}", flush=True)
            sys.stdout.flush()
            logger.error(f"Failed to initialize Gemini: {e}")
            import traceback
            traceback.print_exc()
            self._init_ollama()  # Fallback
    
    def _init_ollama(self):
        """Initialize Ollama local LLM."""
        try:
            from langchain.llms import Ollama
            
            self.llm = Ollama(
                model=self.config.OLLAMA_MODEL,
                base_url=self.config.OLLAMA_BASE_URL,
                temperature=self.config.TEMPERATURE
            )
            self.llm_type = "ollama"
            
            logger.info(f"Ollama LLM initialized: {self.config.OLLAMA_MODEL}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            self.llm_type = "fallback"
    
    def generate(self, prompt: str, max_tokens: int = None) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum tokens in response
        
        Returns:
            Generated text response
        """
        max_tokens = max_tokens or self.config.MAX_RESPONSE_LENGTH
        
        try:
            if self.llm_type == "gemini":
                response = self.llm.generate_content(prompt)
                return response.text.strip()
            
            elif self.llm_type == "ollama":
                response = self.llm(prompt)
                return response.strip()
            
            else:
                return self._fallback_response(prompt)
                
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._fallback_response(prompt)
    
    def generate_json(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Generate a JSON response from the LLM.
        
        Args:
            prompt: Prompt that should return JSON
        
        Returns:
            Parsed JSON dict or None
        """
        try:
            response = self.generate(prompt)
            return self._parse_json(response)
        except Exception as e:
            logger.error(f"JSON generation failed: {e}")
            return None
    
    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling markdown."""
        text = text.strip()
        
        # Remove markdown code blocks
        if text.startswith('```'):
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
        
        return json.loads(text)
    
    def _fallback_response(self, prompt: str) -> str:
        """Generate a fallback response when LLM fails."""
        return "The spirits are unclear... Try looking around for clues!"
    
    def get_info(self) -> Dict[str, str]:
        """Get LLM information."""
        return {
            "type": self.llm_type or "none",
            "model": self.config.GEMINI_MODEL if self.llm_type == "gemini" else self.config.OLLAMA_MODEL,
            "status": "ready" if self.llm else "failed"
        }
