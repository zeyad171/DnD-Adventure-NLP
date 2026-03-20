"""
==============================================
GAME UTILITIES
==============================================
Game configuration constants and helper functions
for state vectorization.

Author: NLP Final Project
==============================================
"""

import torch


# ==============================================
# ACTION CONFIGURATION
# ==============================================

# Actions the LSTM agent can predict
ACTION_MAP = [
    "look",           # 0 - Look around
    "read_books",     # 1 - Read books/scrolls
    "take_key",       # 2 - Take key items
    "use_keys",       # 3 - Use keys/unlock
    "open_exit",      # 4 - Open exit/escape
    "navigate",       # 5 - Move between rooms
    "take_item",      # 6 - Take general items
    "solve_puzzle",   # 7 - Solve puzzles
    "interact"        # 8 - General interaction
]

# Intents recognized by the chatbot
INTENT_MAP = [
    "inspect",        # 0 - Look around, examine
    "navigate",       # 1 - Move between rooms
    "get_item",       # 2 - Pick up items
    "use_item",       # 3 - Use an item
    "unlock",         # 4 - Unlock doors
    "read",           # 5 - Read books, scrolls
    "solve_puzzle",   # 6 - Solve a puzzle
    "interact",       # 7 - Generic interaction
    "escape"          # 8 - Final escape
]

# Room IDs for one-hot encoding
ROOM_IDS = ['hall', 'library', 'study', 'cellar', 'gallery', 'chapel', 'attic', 'crypt', 'dungeon', 'vault']

# Room name mapping for display
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

# State vector size: 10 (room) + 9 (intent) + 6 (game state) = 25 features
STATE_SIZE = 25


# ==============================================
# STATE VECTORIZATION
# ==============================================

def vectorize_state(state_data, intent_str):
    """
    Convert game state to neural network input tensor.
    
    The state vector has 25 features:
        - room_vector (10 values): One-hot encoded current room
        - intent_vector (9 values): One-hot encoded intent
        - step_progress (0-1): Normalized progress
        - inventory_count (0-1): Normalized inventory size
        - has_torch (0 or 1): Has light source
        - has_key (0 or 1): Has any key
        - has_dagger (0 or 1): Has weapon
        - door_locked (0 or 1): Portal status
    
    Args:
        state_data: Dict with 'step', 'inventory', 'doorLocked', 'currentRoom'
        intent_str: String intent from INTENT_MAP
    
    Returns:
        torch.FloatTensor of shape (1, 25)
    """
    # Extract state features
    current_room = state_data.get('currentRoom', 'hall')
    step = state_data.get('step', 1)
    inventory = state_data.get('inventory', [])
    door_locked = state_data.get('doorLocked', True)
    
    # Convert inventory to lowercase strings for checking
    inv_str = ' '.join(str(item).lower() for item in inventory)
    
    # One-hot encode room (10 features)
    room_vec = [0.0] * len(ROOM_IDS)
    if current_room in ROOM_IDS:
        room_vec[ROOM_IDS.index(current_room)] = 1.0
    else:
        room_vec[0] = 1.0  # Default to hall
    
    # One-hot encode intent (9 features)
    intent_vec = [0.0] * len(INTENT_MAP)
    if intent_str in INTENT_MAP:
        intent_vec[INTENT_MAP.index(intent_str)] = 1.0
    else:
        intent_vec[0] = 1.0  # Default to 'inspect'
    
    # Game state features (6 features)
    step_norm = min(step / 50.0, 1.0)  # Normalize over 50 moves
    inventory_count = min(len(inventory) / 10.0, 1.0)  # Normalize over 10 items
    has_torch = 1.0 if 'torch' in inv_str else 0.0
    has_key = 1.0 if 'key' in inv_str or 'tools' in inv_str else 0.0
    has_dagger = 1.0 if 'dagger' in inv_str else 0.0
    door_locked_val = 1.0 if door_locked else 0.0
    
    # Combine all features: room(10) + intent(9) + game_state(6) = 25
    features = room_vec + intent_vec + [
        step_norm, 
        inventory_count, 
        has_torch, 
        has_key, 
        has_dagger, 
        door_locked_val
    ]
    
    return torch.FloatTensor(features).unsqueeze(0)


def get_action_name(action_idx):
    """Get the action name for a given index."""
    if 0 <= action_idx < len(ACTION_MAP):
        return ACTION_MAP[action_idx]
    return ACTION_MAP[0]


def get_intent_index(intent_str):
    """Get the index for an intent string."""
    if intent_str in INTENT_MAP:
        return INTENT_MAP.index(intent_str)
    return 0


def get_room_display_name(room_id):
    """Get the display name for a room ID."""
    return ROOM_NAMES.get(room_id, room_id)
