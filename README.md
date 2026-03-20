# D&D Adventure 🐉⚔️

An interactive Dungeons & Dragons themed adventure game featuring a **RAG-based AI Dungeon Master** and a **trainable LSTM neural network agent**.

## 🎮 Live Demo

Open `index.html` in your browser after starting the backend server.

---

## Project Structure

```
final project/
├── app.py              # Flask API server (main entry point)
├── lstm_agent.py       # LSTM neural network model
├── game_utils.py       # Game constants & state vectorization
├── rag/                # RAG system package
│   ├── __init__.py
│   ├── config.py       # Configuration (API keys, models)
│   ├── llm_handler.py  # Gemini/Ollama integration
│   ├── rag_chain.py    # Main RAG pipeline with FAISS
│   └── rag_system.py   # Simple wrapper interface
├── docs/
│   └── game_rules_rag.txt         # RAG knowledge base
├── index.html          # Game interface
├── app.js              # Frontend game logic
├── game_config.js      # Rooms, items, puzzles
├── maze_renderer.js    # SVG map visualization
├── style.css           # Dark gothic styling
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install flask flask-cors torch sentence-transformers faiss-cpu google-generativeai numpy
```

### 2. Set API Key

```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your_api_key_here"

# Linux/Mac
export GEMINI_API_KEY="your_api_key_here"
```

Get your key from: https://makersuite.google.com/app/apikey

### 3. Start the Backend

```bash
python app.py
```

You should see:
```
==================================================
  Initializing D&D Adventure Backend
==================================================
LSTM Agent (initialized fresh): 14,057 parameters
  D&D Adventure - Backend Server
  Running on http://127.0.0.1:5000
```

### 4. Open the Game

Open `index.html` in your browser (double-click or use Live Server).

---

## Features

### 🎮 Game
- **10 D&D themed chambers** (Tavern, Wizard's Study, Dragon's Hoard, etc.)
- **11 magical items** to collect (Vorpal Dagger, Holy Symbol, etc.)
- **5 puzzles** to solve (Arcane Rituals, Dragon Riddles, Mirror Challenges)
- **Fantasy medieval atmosphere** with animated SVG map
- **Smart navigation** - agent avoids loops and explores intelligently

### 🤖 AI Dungeon Master (RAG)
- **Google Gemini 2.5 Flash Lite** for natural language responses
- **FAISS vector store** with persistent indexing
- **SentenceTransformer embeddings** (all-MiniLM-L6-v2) for semantic search
- **Context-aware responses** based on current room, inventory, and progress
- **Comprehensive item knowledge** - knows what every item does
- **Smart fallback logic** for offline operation

### 🧠 LSTM Action Agent
- **25-feature state vector** (room encoding, intent, inventory, progress)
- **2-layer LSTM** with 64 hidden units and 30% dropout
- **9 action categories** (look, navigate, take_item, solve_puzzle, etc.)
- **Model persistence** - saves to `model_weights.pth`
- **Batch training** on complete game history
- **Smart navigation** - prefers unvisited rooms, avoids backtracking
- **Can navigate between rooms** and complete the game autonomously

---

## Controls

| Key | Action |
|-----|--------|
| W / ↑ | Move North |
| S / ↓ | Move South |
| A / ← | Move West |
| D / → | Move East |
| L | Look around |
| E | Export game log to clipboard |
| Enter | Send chat message |

---

## Training the LSTM Agent

### Quick Training
1. **Play 3-5 complete games** (vary your strategy)
2. **Click "Train on History"** button after each game
3. **Watch loss decrease** (target: < 1.0)
4. **Enable Auto-Train** to train automatically after each game

### Training Metrics
| Metric | Good Value | Meaning |
|--------|------------|---------|
| Loss | < 1.0 | Model is learning patterns |
| Samples | > 500 | Enough training data |
| Games | > 5 | Diverse experience |
| Accuracy | > 60% | Well-trained model |

### Model Persistence
- Model saves to `model_weights.pth` after batch training
- Automatically loads on server restart (checks architecture compatibility)
- Delete file to reset: `Remove-Item model_weights.pth` (Windows) or `rm model_weights.pth` (Linux/Mac)



## API Endpoints

### POST `/chatbot`
Ask the AI Dungeon Master for help.

```json
// Request
{
  "query": "What is the torch used for?",
  "currentRoom": "hall",
  "inventory": [],
  "puzzlesSolved": []
}

// Response
{
  "message": "The Everburning Torch provides magical light! Required for Temple access.",
  "intent": "get_item"
}
```

### POST `/agent/act`
Get agent's action prediction.

```json
// Request
{
  "state": {"step": 5, "inventory": ["torch"], "currentRoom": "library"},
  "intent": "get_item",
  "mask": [1,1,1,1,1,1,1,1,1]
}

// Response
{
  "action_id": "take_item",
  "action_index": 6,
  "sequence_length": 10
}
```

### POST `/agent/train`
Train on a single action (online learning).

### POST `/agent/batch_train`
Train on entire game history (more effective).

```json
// Request
{
  "history": [...],  // Array of {state, intent, actionIndex}
  "epochs": 3
}

// Response
{
  "status": "success",
  "final_loss": 0.45,
  "samples_trained": 600
}
```

### POST `/agent/reset`
Reset agent's state buffer (call on new game).

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Vanilla HTML5, CSS3, JavaScript (ES6+) |
| Backend | Python 3.x, Flask |
| LLM | Google Gemini 2.5 Flash Lite |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2, 384-dim) |
| Vector Store | FAISS (IndexFlatIP) |
| Neural Network | PyTorch LSTM (2 layers, 64 hidden, ~14K parameters) |

---



## Troubleshooting

### "Connection failed" or "Agent offline"
- Check if backend is running on port 5000
- Run `python app.py` and check for errors
- Verify no firewall is blocking localhost:5000

### "The spirits are unclear..."
- Check if `GEMINI_API_KEY` is set correctly
- Verify internet connection for API calls
- Check API key is valid at https://makersuite.google.com/app/apikey

### Model not learning (loss increasing)
- Delete `model_weights.pth` to reset
- Play more diverse games (don't repeat same path)
- Don't spam the Train button (wait for batch training)
- Ensure you have at least 3-5 complete games before training

### Agent stuck in navigation loop
- This was fixed with smart navigation selection
- Agent now prefers unvisited rooms and avoids backtracking
- If still occurs, delete `model_weights.pth` and retrain

### CORS errors
- Backend must be running before opening game
- Try clearing browser cache
- Ensure Flask-CORS is installed

### Vector DB errors
- Delete `vector_db/` folder to rebuild index
- Index rebuilds automatically on next startup

---

## Files Generated at Runtime

| File | Purpose | Can Delete? |
|------|---------|-------------|
| `model_weights.pth` | Trained LSTM weights | Yes (resets training) |
| `vector_db/index.faiss` | FAISS vector index | Yes (rebuilds on startup) |
| `vector_db/chunks.json` | Document chunks metadata | Yes (rebuilds on startup) |

---

## Development

### Project Architecture
- **Modular design**: Backend separated into `app.py`, `lstm_agent.py`, `game_utils.py`
- **RAG package**: Organized in `rag/` folder with clear separation of concerns
- **Frontend**: Vanilla JS with clear separation of game logic, config, and rendering

### Key Features Implemented
- ✅ RAG-based chatbot with FAISS vector search
- ✅ LSTM agent with state history buffer
- ✅ Smart navigation (anti-loop, exploration preference)
- ✅ Model persistence and auto-loading
- ✅ Batch training on game history
- ✅ Comprehensive logging and metrics

---

*May your dice rolls be ever in your favor!* 🎲
