from app.config import EVALUATION_GAMES_COUNT, NEURAL_EVALUATION_GAMES_COUNT, NEURAL_MODEL_FILE
from app.storage.model_storage import load_model
from app.ai.evaluation import evaluate_model, print_evaluation_results
from app.ai.neural_evaluation import format_neural_evaluation_summary
from app.ai.neural_model_service import (
    load_neural_model_package,
    evaluate_saved_neural_model_package,
    get_model_data_from_package,
)
from app.ai.tactical_evaluation import (
    run_default_morpion_tactical_evaluation,
    format_tactical_evaluation_report,
)


def run_evaluate_command():
    model = load_model()

    if len(model) == 0:
        print("Aucun modèle entraîné trouvé.")
        print("Lance d'abord : python main.py train")
        return

    print("Évaluation du modèle...")
    print("Adversaire X : coups aléatoires")
    print("IA O : modèle entraîné")
    print()

    results = evaluate_model(model, games_count=EVALUATION_GAMES_COUNT)
    print_evaluation_results(results)


def run_neural_evaluate_command():
    model_package = load_neural_model_package(NEURAL_MODEL_FILE)

    if not model_package:
        print("Aucun modèle neuronal sauvegardé trouvé.")
        print("Lance d'abord : python main.py train-neural")
        return

    print("Évaluation du modèle neuronal sauvegardé")
    print("Fichier :", NEURAL_MODEL_FILE)
    print("Parties d'évaluation :", NEURAL_EVALUATION_GAMES_COUNT)
    print()

    evaluation = evaluate_saved_neural_model_package(
        model_package,
        games_count=NEURAL_EVALUATION_GAMES_COUNT,
    )
    print(format_neural_evaluation_summary(evaluation["summary"]))


def run_neural_tactical_evaluate_command():
    model_package = load_neural_model_package(NEURAL_MODEL_FILE)

    if not model_package:
        print("Aucun modèle neuronal sauvegardé trouvé.")
        print("Lance d'abord : python main.py train-neural --watch")
        return

    model_data = get_model_data_from_package(model_package)

    if not model_data:
        print("Le fichier neuronal existe, mais ne contient pas de modèle exploitable.")
        print("Relance : python main.py reset-neural --watch")
        return

    results = run_default_morpion_tactical_evaluation(model_data)

    print("Évaluation tactique du modèle neuronal sauvegardé")
    print("Fichier :", NEURAL_MODEL_FILE)
    print()
    print(format_tactical_evaluation_report(results))
