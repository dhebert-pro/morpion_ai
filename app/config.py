MODEL_FILE = "data/models/trained_model.json"
TEST_FILE = "tests/run_tests.py"

# Nombre de parties aléatoires utilisées pour découvrir des états à apprendre.
TRAINING_GAMES_COUNT = 300

# Nombre de simulations pour évaluer chaque coup possible dans un état donné.
SIMULATIONS_PER_MOVE = 30

# Nombre de parties utilisées pour évaluer le modèle après entraînement.
EVALUATION_GAMES_COUNT = 200

# Largeur visuelle de la barre de progression.
PROGRESS_BAR_WIDTH = 30

# Afficher ou non la progression pendant l'entraînement.
SHOW_PROGRESS_DURING_TRAINING = True