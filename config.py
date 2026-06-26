# config.py

# Player (Chnge the id for someone else)
USERNAME = "anshhgoel"

# Paths here username will automatically update it.
PGN_FILE        = f"PGN_File/{USERNAME}_all.pgn"
ANALYSIS_FILE   = f"Analysis/{USERNAME}_analysis.json"
MODEL_DIR       = f"models/{USERNAME}"
TRAINING_DATA   = f"training_data/{USERNAME}"

#  Data filters here for all catagories. my mistake earlier i only looked at rapid 
INCLUDE_TIME_CONTROLS = ["rapid", "blitz", "bullet"] 
MIN_PLIES = 20
# Note : A ply is half move (one user moved ) so it means min there is 10 moves in the game.

# Maia base model
MAIA_BASE_RATING = 1100
MAIA_BASE_MODEL  = f"base_models/maia-{MAIA_BASE_RATING}.pb.gz"

# Training hyperparameters 
LEARNING_RATE       = 1e-4
MAX_STEPS           = 20000
EARLY_STOP_PATIENCE = 10
GENERIC_DATA_MIX    = 0.20   # for better :  20% generic Maia data mixed in to prevent overfit
FREEZE_RESIDUALS    = False  # Set to True to only train the heads, False for full fine-tune
