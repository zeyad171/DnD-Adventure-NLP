"""
RAG Chain
=========
Main RAG pipeline combining all components:
- Document loading and chunking
- Embedding and vector storage
- Retrieval and response generation

Optimized for the D&D Adventure Game.
"""

import os
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class GameRAGChain:
    """
    Complete RAG pipeline for the D&D Adventure Game.
    
    Features:
    - Automatic document indexing
    - Persistent vector storage (FAISS)
    - Gemini-powered response generation
    - Game-aware fallback logic
    """
    
    # Room name mapping
    ROOM_NAMES = {
        'hall': "Dragon's Rest Tavern",
        'library': "Wizard's Study", 
        'study': "Treasure Chamber",
        'cellar': "Dwarven Forge",
        'gallery': "Hall of Heroes",
        'chapel': "Temple of Pelor",
        'attic': "Dragon's Hoard",
        'crypt': "Tomb of the Lich",
        'dungeon': "Goblin Prison",
        'vault': "Portal Chamber"
    }
    
    # Intent categories
    INTENT_MAP = [
        "inspect", "navigate", "get_item", "use_item", 
        "unlock", "read", "solve_puzzle", "interact", "escape"
    ]
    
    def __init__(self, docs_dir: str = None, config=None):
        """
        Initialize the RAG chain.
        
        Args:
            docs_dir: Path to documentation directory
            config: RAGConfig instance (uses default if None)
        """
        from rag.config import config as default_config
        self.config = config or default_config
        
        self.docs_dir = Path(docs_dir or self.config.DOCS_DIR)
        
        # Components (lazy loaded)
        self._embedder = None
        self._index = None
        self._chunks = None
        self._llm = None
        
        # Initialize
        self._setup()
    
    def _setup(self):
        """Initialize all RAG components."""
        import sys
        print("[RAG CHAIN] Starting setup...", flush=True)
        sys.stdout.flush()
        logger.info("Initializing RAG Chain...")
        
        # Load embedder
        print("[RAG CHAIN] Loading embedder (this may take 10-30 seconds first time)...", flush=True)
        sys.stdout.flush()
        self._load_embedder()
        print("[RAG CHAIN] Embedder loaded!", flush=True)
        sys.stdout.flush()
        
        # Load or build index
        print("[RAG CHAIN] Loading/building index...", flush=True)
        sys.stdout.flush()
        self._load_or_build_index()
        print("[RAG CHAIN] Index ready!", flush=True)
        sys.stdout.flush()
        
        # Load LLM
        print("[RAG CHAIN] Loading LLM handler...", flush=True)
        sys.stdout.flush()
        self._load_llm()
        print("[RAG CHAIN] LLM ready!", flush=True)
        sys.stdout.flush()
        
        logger.info("RAG Chain ready!")
        print("[RAG CHAIN] Setup complete!", flush=True)
        sys.stdout.flush()
    
    def _load_embedder(self):
        """Load the sentence transformer embedding model."""
        import sys
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"Loading embedder: {self.config.EMBEDDING_MODEL}")
        print(f"[RAG] Loading SentenceTransformer model: {self.config.EMBEDDING_MODEL}...", flush=True)
        sys.stdout.flush()
        try:
            self._embedder = SentenceTransformer(self.config.EMBEDDING_MODEL)
            print(f"[RAG] Embedder loaded successfully!", flush=True)
            sys.stdout.flush()
        except Exception as e:
            print(f"[RAG ERROR] Failed to load embedder: {e}", flush=True)
            sys.stdout.flush()
            raise
    
    def _load_or_build_index(self):
        """Load existing index or build new one."""
        import sys
        import faiss
        import numpy as np
        
        index_path = Path(self.config.VECTOR_STORE_PATH)
        index_file = index_path / "index.faiss"
        chunks_file = index_path / "chunks.json"
        
        # Try to load existing index
        if index_file.exists() and chunks_file.exists():
            try:
                print("[RAG] Loading existing FAISS index...", flush=True)
                sys.stdout.flush()
                import json
                self._index = faiss.read_index(str(index_file))
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    self._chunks = json.load(f)
                print(f"[RAG] Loaded existing index with {len(self._chunks)} chunks", flush=True)
                sys.stdout.flush()
                logger.info(f"Loaded existing index with {len(self._chunks)} chunks")
                return
            except Exception as e:
                print(f"[RAG WARNING] Failed to load index: {e}, will rebuild...", flush=True)
                sys.stdout.flush()
                logger.warning(f"Failed to load index: {e}")
        
        # Build new index
        print("[RAG] Building new index (this may take 30-60 seconds)...", flush=True)
        sys.stdout.flush()
        self._build_index()
    
    def _build_index(self):
        """Build FAISS index from documentation."""
        import sys
        import faiss
        import numpy as np
        import json
        
        logger.info("Building new index...")
        print("[RAG] Loading and chunking documents...", flush=True)
        sys.stdout.flush()
        
        # Load and chunk documents
        self._chunks = self._load_and_chunk_docs()
        
        if not self._chunks:
            raise ValueError("No chunks created from documents!")
        
        print(f"[RAG] Creating embeddings for {len(self._chunks)} chunks...", flush=True)
        sys.stdout.flush()
        # Create embeddings
        texts = [chunk['content'] for chunk in self._chunks]
        embeddings = self._embedder.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        print("[RAG] Building FAISS index...", flush=True)
        sys.stdout.flush()
        # Build FAISS index
        dimension = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self._index.add(embeddings)
        
        print("[RAG] Saving index to disk...", flush=True)
        sys.stdout.flush()
        # Save index
        index_path = Path(self.config.VECTOR_STORE_PATH)
        index_path.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self._index, str(index_path / "index.faiss"))
        with open(index_path / "chunks.json", 'w', encoding='utf-8') as f:
            json.dump(self._chunks, f, indent=2)
        
        print(f"[RAG] Index built and saved with {len(self._chunks)} chunks!", flush=True)
        sys.stdout.flush()
        logger.info(f"Built and saved index with {len(self._chunks)} chunks")
    
    def _load_and_chunk_docs(self) -> List[Dict[str, Any]]:
        """Load the game rules document and split into chunks."""
        chunks = []
        
        # Load only the dedicated RAG knowledge file
        doc_path = self.docs_dir / self.config.RAG_KNOWLEDGE_FILE
        
        if doc_path.exists():
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by sections (better than fixed size for game docs)
            doc_chunks = self._smart_chunk(content, doc_path.name)
            chunks.extend(doc_chunks)
            logger.info(f"Loaded RAG knowledge from: {doc_path.name}")
        else:
            # Fallback: load all markdown files if specific file not found
            logger.warning(f"RAG knowledge file not found: {doc_path}, falling back to all .md files")
            for doc_path in self.docs_dir.glob("**/*.md"):
                with open(doc_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                doc_chunks = self._smart_chunk(content, doc_path.name)
                chunks.extend(doc_chunks)
        
        logger.info(f"Created {len(chunks)} chunks from documents")
        return chunks
    
    def _smart_chunk(self, content: str, source: str) -> List[Dict[str, str]]:
        """
        Smart chunking that preserves semantic boundaries.
        Splits on markdown headers and paragraphs.
        """
        chunks = []
        
        # Split by markdown headers first
        import re
        sections = re.split(r'\n(?=#{1,3}\s)', content)
        
        for section in sections:
            section = section.strip()
            if len(section) < 20:
                continue
            
            # If section is too long, split by paragraphs
            if len(section) > self.config.CHUNK_SIZE:
                paragraphs = section.split('\n\n')
                current_chunk = ""
                
                for para in paragraphs:
                    if len(current_chunk) + len(para) < self.config.CHUNK_SIZE:
                        current_chunk += para + "\n\n"
                    else:
                        if current_chunk.strip():
                            chunks.append({
                                'content': current_chunk.strip(),
                                'source': source
                            })
                        current_chunk = para + "\n\n"
                
                if current_chunk.strip():
                    chunks.append({
                        'content': current_chunk.strip(),
                        'source': source
                    })
            else:
                chunks.append({
                    'content': section,
                    'source': source
                })
        
        return chunks
    
    def _load_llm(self):
        """Load the LLM handler."""
        import sys
        from rag.llm_handler import LLMHandler
        print("[RAG] Initializing LLM handler...", flush=True)
        sys.stdout.flush()
        self._llm = LLMHandler(self.config)
        print("[RAG] LLM handler ready!", flush=True)
        sys.stdout.flush()
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: Search query
            top_k: Number of results (default from config)
        
        Returns:
            List of relevant chunks with scores
        """
        import numpy as np
        import faiss
        
        top_k = top_k or self.config.TOP_K
        
        # Embed query
        query_vec = self._embedder.encode([query])
        query_vec = np.array(query_vec).astype('float32')
        faiss.normalize_L2(query_vec)
        
        # Search
        scores, indices = self._index.search(query_vec, top_k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= self.config.SCORE_THRESHOLD:
                chunk = self._chunks[idx].copy()
                chunk['score'] = float(score)
                results.append(chunk)
        
        return results
    
    def generate_response(self, query: str, game_state: Dict[str, Any]) -> Tuple[str, str]:
        """
        Generate a response using RAG.
        
        Args:
            query: User's question
            game_state: Current game state dict
        
        Returns:
            Tuple of (message, intent)
        """
        current_room = game_state.get('currentRoom', 'hall')
        inventory = game_state.get('inventory', [])
        puzzles_solved = game_state.get('puzzlesSolved', [])
        door_locked = game_state.get('doorLocked', True)
        step = game_state.get('step', 1)
        
        room_name = self.ROOM_NAMES.get(current_room, current_room)
        
        try:
            # Retrieve relevant context
            context_chunks = self.retrieve(query, top_k=3)
            context = "\n\n".join([c['content'] for c in context_chunks])
            
            # Build prompt
            if context.strip():
                knowledge_section = f"""GAME KNOWLEDGE (use this to answer):
{context}"""
            else:
                knowledge_section = "GAME KNOWLEDGE: No specific knowledge found for this query. Use your general knowledge of D&D adventure games to provide helpful guidance."
            
            prompt = f"""You are the Dungeon Master for a D&D adventure game.
IMPORTANT: Base your response on the GAME KNOWLEDGE provided below. If knowledge is provided, use it. If not, provide helpful general guidance in D&D style.
Speak in a fantasy medieval style. Keep responses concise (1-2 sentences).

GAME STATE:
- Current Location: {room_name}
- Rooms Visited: {step}
- Inventory: {inventory if inventory else "Empty"}
- Puzzles Solved: {puzzles_solved if puzzles_solved else "None yet"}
- Portal Status: {"Sealed" if door_locked else "Activated"}

{knowledge_section}

ADVENTURER ASKS: {query}

INSTRUCTIONS:
1. Answer based on the GAME KNOWLEDGE above (or provide helpful guidance if no knowledge found)
2. Choose ONE intent from: {self.INTENT_MAP}
3. Reply as JSON only: {{"message": "your answer", "intent": "chosen_intent"}}"""

            # Generate response
            result = self._llm.generate_json(prompt)
            
            if result:
                message = result.get('message', 'Look around for clues.')
                intent = result.get('intent', 'inspect')
                
                if intent not in self.INTENT_MAP:
                    intent = 'inspect'
                
                return message, intent
            
        except Exception as e:
            logger.error(f"RAG generation failed: {e}")
        
        # Fallback
        return self._fallback_response(query, game_state)
    
    # Item knowledge database
    ITEM_INFO = {
        'torch': ("The Everburning Torch provides magical light! Required to access the Temple of Pelor and Hall of Heroes. Take it and use it to unlock dark passages.", "get_item"),
        'candle': ("The Ritual Candle is used in the Wizard's Study puzzle. Light the candles from shortest to tallest to reveal a hidden compartment with a key!", "use_item"),
        'dagger': ("The Vorpal Dagger +1 is essential for the Mirror Puzzle in the Hall of Heroes. Hold it at the midnight position before the magical mirror.", "use_item"),
        'cross': ("The Holy Symbol grants passage between the Tavern and Temple. It also protects against evil.", "use_item"),
        'holy': ("The Blessed Potion (Holy Water) is required for the final Portal Ritual. Pour it on the arcane symbols to begin the escape sequence.", "use_item"),
        'water': ("The Blessed Potion (Holy Water) is required for the final Portal Ritual. Pour it on the arcane symbols to begin the escape sequence.", "use_item"),
        'potion': ("The Blessed Potion (Holy Water) is required for the final Portal Ritual. Pour it on the arcane symbols to begin the escape sequence.", "use_item"),
        'scroll': ("The Scroll of Teleportation contains the incantation for the final ritual. Read it after pouring holy water on the symbols.", "read"),
        'tome': ("The Tome of Portal Magic is in the Portal Chamber. Take it, read it to learn the ritual, then perform the ritual to escape!", "get_item"),
        'map': ("The Dungeon Map reveals all room locations! Find it in the Dragon's Hoard (attic).", "get_item"),
        'thieves': ("Thieves' Tools unlock the passage to the Treasure Chamber. Found after solving the candle puzzle in Wizard's Study.", "unlock"),
        'tools': ("Thieves' Tools unlock the passage to the Treasure Chamber. Found after solving the candle puzzle in Wizard's Study.", "unlock"),
        'dragon': ("The Dragon Scale Key is one of 3 keys needed for the Portal Chamber. Found in the Goblin Prison.", "get_item"),
        'lich': ("The Lich's Phylactery Key is one of 3 keys needed for the Portal Chamber. Found in the Tomb of the Lich.", "get_item"),
        'skeleton': ("The Lich's Phylactery Key (Skeleton Key) is one of 3 keys for the Portal Chamber. Take it from the coffin in the Tomb.", "get_item"),
        'golden': ("The Golden Key is found inside the safe in the Treasure Chamber. Solve the portrait puzzle (3-7-4-1) to open it.", "get_item"),
    }
    
    def _fallback_response(self, query: str, game_state: Dict) -> Tuple[str, str]:
        """Context-aware fallback when RAG fails."""
        current_room = game_state.get('currentRoom', 'hall')
        inventory = game_state.get('inventory', [])
        query_lower = query.lower()
        
        # ITEM QUESTIONS - Check FIRST before generic patterns
        for item_key, (response, intent) in self.ITEM_INFO.items():
            if item_key in query_lower:
                return (response, intent)
        
        # Game rules
        if any(w in query_lower for w in ['rule', 'how to play', 'instructions', 'controls', 'help']):
            return (
                "Game Rules: Explore 10 chambers, collect items, solve 5 puzzles! "
                "Use WASD/arrows to move, L to look around, click buttons for actions. "
                "Get 3 keys to reach the Portal Chamber, then perform the ritual to escape!",
                "inspect"
            )
        
        # Game introduction (only if NOT asking about specific items)
        if 'what is this game' in query_lower or 'about the game' in query_lower:
            return (
                "Welcome to the D&D Adventure! You're trapped in a magical dungeon with 10 chambers. "
                "Explore, collect items, solve puzzles, and activate the escape portal to win!",
                "inspect"
            )
        
        # Keys (general)
        if 'key' in query_lower and 'what' not in query_lower:
            return ("You need 3 keys: Thieves' Tools (Wizard's Study), Dragon Scale Key (Goblin Prison), Lich's Key (Tomb).", "get_item")
        
        # What to do next - general advice
        if any(w in query_lower for w in ['what to do', 'next', 'now what', 'stuck', 'where']):
            room_name = self.ROOM_NAMES.get(current_room, current_room)
            if not inventory:
                return (f"Look around the {room_name}! Search for items and clues.", "inspect")
            elif current_room == 'vault':
                return ("Take the tome, read it, then perform the ritual to escape!", "get_item")
            elif current_room == 'dungeon' and len([k for k in inventory if 'key' in str(k).lower()]) >= 3:
                return ("You have the keys! Pull the lever to open the Portal Chamber!", "solve_puzzle")
            else:
                return (f"Explore the {room_name}. Look for puzzles and items to collect.", "inspect")
        
        # Room-specific hints (only if nothing else matched)
        if current_room == 'hall' and 'torch' not in str(inventory).lower():
            return ("Take the Everburning Torch to light your way!", "get_item")
        
        if current_room == 'vault':
            return ("The Portal Chamber! Take the tome and perform the ritual to escape!", "get_item")
        
        return ("Explore your surroundings! Look around for items and clues.", "inspect")
    
    def rebuild_index(self):
        """Force rebuild the vector index."""
        import shutil
        
        index_path = Path(self.config.VECTOR_STORE_PATH)
        if index_path.exists():
            shutil.rmtree(index_path)
        
        self._build_index()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        return {
            "chunks_indexed": len(self._chunks) if self._chunks else 0,
            "embedding_model": self.config.EMBEDDING_MODEL,
            "llm": self._llm.get_info() if self._llm else None,
            "docs_dir": str(self.docs_dir),
            "index_path": self.config.VECTOR_STORE_PATH
        }


# Singleton instance
_rag_chain = None

def get_rag_chain() -> GameRAGChain:
    """Get or create the RAG chain singleton."""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = GameRAGChain()
    return _rag_chain
