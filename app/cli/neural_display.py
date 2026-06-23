from app.config import (
    NEURAL_MODEL_FILE,
    NEURAL_TRAINING_GAMES_COUNT,
    NEURAL_SIMULATIONS_PER_MOVE,
    NEURAL_MAX_EXAMPLES,
    NEURAL_TACTICAL_REPEAT_COUNT,
    NEURAL_HIDDEN_SIZE,
    NEURAL_EPOCHS,
    NEURAL_LEARNING_RATE,
    NEURAL_EVALUATION_GAMES_COUNT,
)
from app.ai.neural_model_service import evaluate_saved_neural_model_package
from app.ai.neural_evaluation import format_neural_evaluation_summary


def print_neural_training_parameters():
    print("Parties simulées pour collecter les états :", NEURAL_TRAINING_GAMES_COUNT)
    print("Simulations par coup :", NEURAL_SIMULATIONS_PER_MOVE)
    print("Nombre maximal d'exemples Monte-Carlo :", NEURAL_MAX_EXAMPLES)
    print("Répétitions des exemples tactiques :", NEURAL_TACTICAL_REPEAT_COUNT)
    print("Taille couche cachée :", NEURAL_HIDDEN_SIZE)
    print("Époques :", NEURAL_EPOCHS)
    print("Taux d'apprentissage :", NEURAL_LEARNING_RATE)


def print_neural_training_result(model_package):
    training_summary = model_package["training_summary"]

    print()
    print("Modèle neuronal sauvegardé dans :", NEURAL_MODEL_FILE)
    print("Reprise d'un modèle existant :", training_summary["started_from_existing_model"])
    print("Exemples Monte-Carlo :", training_summary["base_examples_count"])
    print("Exemples tactiques ajoutés :", training_summary["extra_examples_count"])
    print("Exemples totaux :", training_summary["examples_count"])
    print("Erreur initiale :", round(training_summary["initial_error"], 6))
    print("Erreur finale :", round(training_summary["final_error"], 6))
    print("Amélioration erreur :", round(training_summary["error_improvement"], 6))
    _print_validation_summary_if_present(training_summary)
    print()

    print("Évaluation rapide du modèle neuronal sauvegardé...")
    evaluation = evaluate_saved_neural_model_package(
        model_package,
        games_count=NEURAL_EVALUATION_GAMES_COUNT,
    )
    print(format_neural_evaluation_summary(evaluation["summary"]))


def _print_validation_summary_if_present(training_summary):
    if "final_validation_error" not in training_summary:
        return

    print("Exemples validation :", training_summary.get("validation_examples_count"))
    print("Erreur validation finale :", round(training_summary["final_validation_error"], 6))
