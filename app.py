"""
==============================================
D&D ADVENTURE - FLASK API SERVER
==============================================
Main application entry point providing:
1. RAG-based chatbot (Dungeon Master AI)
2. LSTM agent for action prediction
3. Training endpoints for the neural network

Author: NLP Final Project
==============================================
"""

# IMPORTANT: Set these BEFORE any imports to prevent TensorFlow loading
import os
import sys

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TORCH"] = "1"
sys.modules['tensorflow'] = None

import torch
import torch.nn as nn
import random
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import our modules
from lstm_agent import create_agent, save_model
from rag.rag_system import get_rag_system
from game_utils import ACTION_MAP, vectorize_state


# ==============================================
# FLASK APP SETUP
# ==============================================

app = Flask(__name__)
CORS(app)


# ==============================================
# INITIALIZE COMPONENTS
# ==============================================

print("\n" + "=" * 50)
print("  Initializing D&D Adventure Backend")
print("=" * 50)

# Initialize LSTM Agent
model, optimizer, criterion, state_buffer = create_agent(num_actions=len(ACTION_MAP))

# Initialize RAG System (lazy loading on first request)
rag_system = None

def get_rag():
    """Lazy load the RAG system."""
    global rag_system
    if rag_system is None:
        rag_system = get_rag_system()
    return rag_system


# ==============================================
# HEALTH CHECK ENDPOINT
# ==============================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint to verify server is running."""
    return jsonify({
        "status": "ok",
        "message": "Server is running",
        "lstm_ready": model is not None,
        "rag_ready": rag_system is not None
    })


# ==============================================
# CHATBOT ENDPOINT
# ==============================================

@app.route('/chatbot', methods=['POST'])
def chatbot():
    """
    RAG-based chatbot endpoint (Dungeon Master AI).
    
    Expects JSON: {query, step, inventory, doorLocked, currentRoom, puzzlesSolved}
    Returns JSON: {message, intent}
    """
    data = request.json
    query = data.get('query', 'What should I do next?')
    
    # Build game state dict
    game_state = {
        'currentRoom': data.get('currentRoom', 'hall'),
        'step': data.get('step', 1),
        'inventory': data.get('inventory', []),
        'puzzlesSolved': data.get('puzzlesSolved', []),
        'doorLocked': data.get('doorLocked', True)
    }
    
    try:
        print(f"[CHATBOT] Received query: {query}")
        print(f"[CHATBOT] Getting RAG system...")
        rag = get_rag()
        print(f"[CHATBOT] RAG system obtained, generating response...")
        message, intent = rag.generate_response(query, game_state)
        print(f"[CHATBOT] Response generated: intent={intent}")
        return jsonify({"message": message, "intent": intent})
    except Exception as e:
        print(f"[CHATBOT ERROR] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "message": "The spirits are unclear... Try looking around for clues!",
            "intent": "inspect"
        })


# ==============================================
# AGENT ENDPOINTS
# ==============================================

@app.route('/agent/act', methods=['POST'])
def agent_act():
    """
    Agent action prediction endpoint.
    
    Extracts game state features and predicts the next optimal action using the trained LSTM model.
    Reads from the state buffer to generate sequences for sequential learning.
    
    Expects JSON: {state, intent, mask}
    Returns JSON: {action_id, action_index, sequence_length}
    """
    try:
        import torch
        data = request.json
        state_data = data['state']
        
        intent = data.get('intent', 'inspect')
        mask_list = data.get('mask', [1] * len(ACTION_MAP))
        mask = torch.FloatTensor(mask_list).unsqueeze(0)
        
        # Extract features and update state buffer
        state_vec = vectorize_state(state_data, intent)
        state_buffer.add(state_vec.squeeze(0))
        state_sequence = state_buffer.get_sequence()
        
        if state_sequence is None:
            return jsonify({
                "action_id": ACTION_MAP[0],
                "action_index": 0,
                "sequence_length": 0
            })
            
        model.eval()
        with torch.no_grad():
            logits, _ = model(state_sequence, mask=mask)
            # Find the action with highest probability
            action_idx = torch.argmax(logits, dim=-1).item()
            
        action_id = ACTION_MAP[action_idx]
        
        print(f"[LSTM AGENT] Room: {state_data.get('currentRoom', 'hall')}, Predicted Action: {action_id} (Index: {action_idx})")
        
        return jsonify({
            "action_id": action_id,
            "action_index": action_idx,
            "sequence_length": len(state_buffer)
        })
        
    except Exception as e:
        print(f"Agent act error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "action_id": ACTION_MAP[0],
            "action_index": 0,
            "error": str(e)
        })


@app.route('/agent/train', methods=['POST'])
def agent_train():
    """
    LSTM Agent single-step training endpoint.
    
    Trains on the current state sequence from the history buffer.
    
    Expects JSON: {state, intent, correct_action_id}
    Returns JSON: {status, loss, sequence_length}
    """
    try:
        data = request.json
        
        # Prepare current state vector and add to buffer
        state_vec = vectorize_state(data['state'], data['intent'])
        state_buffer.add(state_vec.squeeze(0))
        
        # Get target action (ensure it's within valid range)
        action_id = data.get('correct_action_id', 0)
        if action_id >= len(ACTION_MAP):
            action_id = 0
        target = torch.LongTensor([action_id])
        
        # Get the sequence of states from buffer
        state_sequence = state_buffer.get_sequence()
        
        # Handle empty buffer case
        if state_sequence is None:
            return jsonify({
                "status": "skipped",
                "loss": 0.0,
                "message": "Buffer empty, state added",
                "sequence_length": len(state_buffer)
            })
        
        # Training step with LSTM
        model.train()
        optimizer.zero_grad()
        
        # Forward pass with sequence
        logits, _ = model(state_sequence)
        
        # Compute loss and backpropagate
        loss = criterion(logits, target)
        
        # Only backprop if loss is finite (prevent NaN issues)
        if torch.isfinite(loss):
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        else:
            print(f"Warning: Skipping training step with non-finite loss: {loss.item()}")
            return jsonify({
                "status": "error",
                "error": "Non-finite loss detected",
                "loss": 0.0
            })
        
        return jsonify({
            "status": "trained",
            "loss": loss.item(),
            "sequence_length": len(state_buffer)
        })
        
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "error": str(e),
            "loss": 0.0
        })


@app.route('/agent/batch_train', methods=['POST'])
def agent_batch_train():
    """
    Batch training endpoint - trains on entire game history.
    
    More effective for LSTM as it processes full sequences.
    
    Expects JSON: {history: [{state, intent, actionIndex}], epochs: int}
    Returns JSON: {status, final_loss, samples_trained}
    """
    try:
        data = request.json
        history = data.get('history', [])
        epochs = data.get('epochs', 5)  # Increased default epochs for better learning
        
        if len(history) == 0:
            return jsonify({
                "status": "error",
                "error": "No history provided",
                "final_loss": 0.0,
                "samples_trained": 0
            })
        
        print(f"Batch training on {len(history)} moves for {epochs} epochs...")
        
        # Convert history to training sequences
        training_data = []
        for move in history:
            state_data = move.get('state', {})
            intent = move.get('intent', 'inspect')
            action_idx = move.get('actionIndex', 0)
            
            state_vec = vectorize_state(state_data, intent)
            training_data.append({
                'state': state_vec.squeeze(0),
                'action': action_idx
            })
        
        total_loss = 0.0
        total_samples_trained = 0  # Track total samples across all epochs
        
        model.train()
        
        # Improved learning rate scheduling with warmup
        # Start with lower LR, then use ReduceLROnPlateau
        initial_lr = optimizer.param_groups[0]['lr']
        warmup_epochs = 1
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=1, verbose=True, min_lr=1e-5
        )
        
        # Calculate class weights to handle imbalanced actions
        action_counts = {}
        for item in training_data:
            action = item['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        total_samples = len(training_data)
        num_actions = len(ACTION_MAP)
        class_weights = torch.ones(num_actions)
        for action_idx, count in action_counts.items():
            if count > 0:
                # Weight inversely proportional to frequency
                class_weights[action_idx] = total_samples / (num_actions * count)
        
        # Normalize weights
        class_weights = class_weights / class_weights.sum() * num_actions
        weighted_criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Class weights: {dict(enumerate(class_weights.tolist()))}")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_samples = 0  # Samples trained in this epoch
            
            # Learning rate warmup for first epoch
            if epoch < warmup_epochs:
                warmup_lr = initial_lr * (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            # Clear buffer for each epoch
            state_buffer.clear()
            model.reset_hidden()
            
            # DO NOT shuffle training_data within an episode! 
            # The LSTM must see the states in chronological order to learn the sequence properly.
            for item in training_data:
                state_buffer.add(item['state'])
                
                # Need at least 2 states for meaningful sequence
                if len(state_buffer) < 2:
                    continue
                
                state_sequence = state_buffer.get_sequence()
                if state_sequence is None:
                    continue
                
                target = torch.LongTensor([item['action']])
                
                optimizer.zero_grad()
                logits, _ = model(state_sequence)
                
                # Use weighted loss to handle class imbalance
                loss = weighted_criterion(logits, target)
                
                # Only backprop if loss is finite (prevent NaN issues)
                if torch.isfinite(loss):
                    loss.backward()
                    
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    epoch_samples += 1
                    total_samples_trained += 1
                else:
                    print(f"Warning: Skipping training step with non-finite loss: {loss.item()}")
            
            total_loss += epoch_loss
            avg_epoch_loss = epoch_loss / max(1, epoch_samples)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch + 1}/{epochs}: loss = {avg_epoch_loss:.4f}, lr = {current_lr:.6f}, samples = {epoch_samples}")
            
            # Update learning rate based on loss (after warmup)
            if epoch >= warmup_epochs:
                scheduler.step(avg_epoch_loss)
        
        final_loss = total_loss / max(1, total_samples_trained)
        print(f"Batch training complete! Final avg loss: {final_loss:.4f}, total samples trained: {total_samples_trained}")
        
        # Save model after batch training
        save_model(model, optimizer)
        
        return jsonify({
            "status": "success",
            "final_loss": final_loss,
            "samples_trained": total_samples_trained,
            "epochs": epochs
        })
        
    except Exception as e:
        print(f"Batch training error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "error": str(e),
            "final_loss": 0.0,
            "samples_trained": 0
        })


@app.route('/agent/reset', methods=['POST'])
def agent_reset():
    """
    Reset the LSTM agent's state buffer.
    
    Call this when starting a new game.
    """
    state_buffer.clear()
    model.reset_hidden()
    
    return jsonify({
        "status": "reset",
        "message": "LSTM state buffer cleared"
    })


# ==============================================
# MAIN ENTRY POINT
# ==============================================

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  D&D Adventure - Backend Server")
    print("  Running on http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    app.run(port=5000, debug=True)