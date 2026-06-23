from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"

MODEL_FILE = MODELS_DIR / "trained_model.json"
MOVE_SCORE_DATASET_FILE = MODELS_DIR / "move_score_dataset.json"
NEURAL_MODEL_FILE = MODELS_DIR / "neural_model.json"

TEST_FILE = PROJECT_ROOT / "tests" / "run_tests.py"

# Ancien modèle tabulaire.
TRAINING_GAMES_COUNT = 300
SIMULATIONS_PER_MOVE = 30
EVALUATION_GAMES_COUNT = 200

# Dataset Monte-Carlo destiné au moteur neuronal.
MOVE_SCORE_DATASET_MAX_EXAMPLES = 200

# Modèle neuronal sauvegardé.
NEURAL_TRAINING_GAMES_COUNT = 300
NEURAL_SIMULATIONS_PER_MOVE = 30
NEURAL_MAX_EXAMPLES = 200
NEURAL_HIDDEN_SIZE = 18
NEURAL_EPOCHS = 160
NEURAL_LEARNING_RATE = 0.12
NEURAL_EVALUATION_GAMES_COUNT = 200

# Affichage progression.
PROGRESS_BAR_WIDTH = 30
SHOW_PROGRESS_DURING_TRAINING = True