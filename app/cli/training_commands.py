import app.ai.strategies as strategies

from app.config import (
    MODEL_FILE,
    MOVE_SCORE_DATASET_FILE,
    MOVE_SCORE_DATASET_MAX_EXAMPLES,
    TRAINING_GAMES_COUNT,
    SIMULATIONS_PER_MOVE,
    EVALUATION_GAMES_COUNT,
    SHOW_PROGRESS_DURING_TRAINING,
)
from app.storage.model_storage import save_model
from app.storage.json_storage import save_json
from app.ai.training import train_model
from app.ai.evaluation import evaluate_model, print_evaluation_results
from app.ai.training_dataset import (
    build_move_score_dataset,
    summarize_move_score_dataset,
)


def run_training_command():
    print("Paramètres d'entraînement :")
    print("Parties simulées pour collecter les états :", TRAINING_GAMES_COUNT)
    print("Simulations par coup :", SIMULATIONS_PER_MOVE)
    print("Parties d'évaluation :", EVALUATION_GAMES_COUNT)
    print()

    print("Entraînement de l'IA...")
    strategies.TRAINED_MODEL = train_model(
        training_games_count=TRAINING_GAMES_COUNT,
        simulations_per_move=SIMULATIONS_PER_MOVE,
        show_progress=SHOW_PROGRESS_DURING_TRAINING,
    )
    save_model(strategies.TRAINED_MODEL)

    print("Entraînement terminé.")
    print("Nombre d'états appris :", len(strategies.TRAINED_MODEL))
    print("Modèle sauvegardé dans :", MODEL_FILE)
    print()

    print("Évaluation rapide du modèle entraîné...")
    results = evaluate_model(
        strategies.TRAINED_MODEL,
        games_count=EVALUATION_GAMES_COUNT,
    )
    print_evaluation_results(results)


def run_build_dataset_command():
    print("Création du dataset d'apprentissage :")
    print("Parties simulées pour collecter les états :", TRAINING_GAMES_COUNT)
    print("Simulations par coup :", SIMULATIONS_PER_MOVE)
    print("Nombre maximal d'exemples :", MOVE_SCORE_DATASET_MAX_EXAMPLES)
    print()

    dataset = build_move_score_dataset(
        training_games_count=TRAINING_GAMES_COUNT,
        simulations_per_move=SIMULATIONS_PER_MOVE,
        max_examples=MOVE_SCORE_DATASET_MAX_EXAMPLES,
        show_progress=SHOW_PROGRESS_DURING_TRAINING,
    )
    save_json(dataset, MOVE_SCORE_DATASET_FILE)
    summary = summarize_move_score_dataset(dataset)

    print("Dataset créé.")
    print("Fichier sauvegardé dans :", MOVE_SCORE_DATASET_FILE)
    print()
    print("Résumé :")
    print("Jeu :", summary["game"])
    print("Exemples :", summary["examples_count"])
    print("Coups scorés :", summary["scored_moves_count"])
    print("Nombre moyen de coups légaux :", summary["average_legal_moves"])
    print("Score moyen du meilleur coup :", summary["average_best_score"])
