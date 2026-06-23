from app.config import (
    NEURAL_MODEL_FILE,
    SHOW_PROGRESS_DURING_TRAINING,
    NEURAL_BENCHMARK_TRAINING_GAMES_COUNT,
    NEURAL_BENCHMARK_SIMULATIONS_PER_MOVE,
    NEURAL_BENCHMARK_MAX_EXAMPLES,
    NEURAL_BENCHMARK_TACTICAL_REPEAT_COUNT,
    NEURAL_BENCHMARK_HIDDEN_SIZE,
    NEURAL_BENCHMARK_CHECKPOINTS_COUNT,
    NEURAL_BENCHMARK_EPOCHS_PER_CHECKPOINT,
    NEURAL_BENCHMARK_LEARNING_RATE,
    NEURAL_BENCHMARK_EVALUATION_GAMES_COUNT,
    NEURAL_BENCHMARK_VALIDATION_RATIO,
    NEURAL_BENCHMARK_EARLY_STOP_PATIENCE,
)
from app.storage.json_storage import save_json
from app.ai.neural_model_service import (
    load_neural_model_package,
    get_model_data_from_package,
)
from app.ai.neural_benchmark import (
    run_neural_training_benchmark,
    format_neural_benchmark_report,
    create_model_package_from_benchmark_result,
)


def run_neural_training_with_watch_command(start_from_saved_model):
    print("Entraînement surveillé du modèle neuronal")
    print("Mode : affichage de l'amélioration par paliers")
    print("Le meilleur modèle rencontré sera sauvegardé.")
    print()

    initial_model_data = _load_initial_model_data(start_from_saved_model)
    print()
    print_neural_watch_configuration()
    print()

    benchmark_result = _run_watch_benchmark(initial_model_data)
    model_package = create_model_package_from_benchmark_result(benchmark_result)
    save_json(model_package, NEURAL_MODEL_FILE)

    print()
    print(format_neural_benchmark_report(benchmark_result))
    print()
    print("Modèle neuronal sauvegardé dans :", NEURAL_MODEL_FILE)
    print("Palier sauvegardé :", benchmark_result["best_checkpoint_index"])
    print("Époques du modèle sauvegardé :", benchmark_result["best_total_epochs"])
    print("Efficacité du modèle sauvegardé :", round(benchmark_result["best_evaluation_efficiency"], 2), "%")


def run_neural_benchmark_command(start_from_saved_model):
    print("Benchmark du modèle neuronal")
    print("Aucune sauvegarde ne sera effectuée.")
    print_neural_watch_configuration()
    print()

    initial_model_data = _load_initial_model_data(start_from_saved_model)
    print()

    benchmark_result = _run_watch_benchmark(initial_model_data)

    print()
    print(format_neural_benchmark_report(benchmark_result))


def print_neural_watch_configuration():
    print("Parties simulées pour collecter les états :", NEURAL_BENCHMARK_TRAINING_GAMES_COUNT)
    print("Simulations par coup :", NEURAL_BENCHMARK_SIMULATIONS_PER_MOVE)
    print("Nombre maximal d'exemples Monte-Carlo :", NEURAL_BENCHMARK_MAX_EXAMPLES)
    print("Répétitions tactiques :", NEURAL_BENCHMARK_TACTICAL_REPEAT_COUNT)
    print("Taille couche cachée :", NEURAL_BENCHMARK_HIDDEN_SIZE)
    print("Paliers :", NEURAL_BENCHMARK_CHECKPOINTS_COUNT)
    print("Époques par palier :", NEURAL_BENCHMARK_EPOCHS_PER_CHECKPOINT)
    print("Taux d'apprentissage :", NEURAL_BENCHMARK_LEARNING_RATE)
    print("Parties d'évaluation par palier :", NEURAL_BENCHMARK_EVALUATION_GAMES_COUNT)
    print("Part validation :", NEURAL_BENCHMARK_VALIDATION_RATIO)
    print("Arrêt après paliers sans meilleur modèle :", NEURAL_BENCHMARK_EARLY_STOP_PATIENCE)


def _load_initial_model_data(start_from_saved_model):
    if not start_from_saved_model:
        print("Réinitialisation demandée.")
        print("L'entraînement surveillé repart de zéro.")
        return None

    existing_model_package = load_neural_model_package(NEURAL_MODEL_FILE)
    initial_model_data = get_model_data_from_package(existing_model_package)

    if initial_model_data:
        print("Modèle existant trouvé :", NEURAL_MODEL_FILE)
        print("L'entraînement surveillé va continuer depuis ce modèle.")
    else:
        print("Aucun modèle existant trouvé.")
        print("L'entraînement surveillé démarre de zéro.")

    return initial_model_data


def _run_watch_benchmark(initial_model_data):
    return run_neural_training_benchmark(
        training_games_count=NEURAL_BENCHMARK_TRAINING_GAMES_COUNT,
        simulations_per_move=NEURAL_BENCHMARK_SIMULATIONS_PER_MOVE,
        max_examples=NEURAL_BENCHMARK_MAX_EXAMPLES,
        tactical_repeat_count=NEURAL_BENCHMARK_TACTICAL_REPEAT_COUNT,
        hidden_size=NEURAL_BENCHMARK_HIDDEN_SIZE,
        checkpoints_count=NEURAL_BENCHMARK_CHECKPOINTS_COUNT,
        epochs_per_checkpoint=NEURAL_BENCHMARK_EPOCHS_PER_CHECKPOINT,
        learning_rate=NEURAL_BENCHMARK_LEARNING_RATE,
        evaluation_games_count=NEURAL_BENCHMARK_EVALUATION_GAMES_COUNT,
        show_progress=SHOW_PROGRESS_DURING_TRAINING,
        print_checkpoints=True,
        seed=0,
        initial_model_data=initial_model_data,
        validation_ratio=NEURAL_BENCHMARK_VALIDATION_RATIO,
        early_stop_patience=NEURAL_BENCHMARK_EARLY_STOP_PATIENCE,
    )
