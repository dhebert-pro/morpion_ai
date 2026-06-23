import sys

from app.config import (
    SANTORINI_DATASET_FILE,
    SANTORINI_DATASET_SEED,
    SANTORINI_NEURAL_EPOCHS,
    SANTORINI_NEURAL_EVALUATION_GAMES_COUNT,
    SANTORINI_NEURAL_HIDDEN_SIZE,
    SANTORINI_NEURAL_LEARNING_RATE,
    SANTORINI_NEURAL_MAX_EXAMPLES,
    SANTORINI_NEURAL_MODEL_FILE,
    SANTORINI_NEURAL_SIMULATIONS_PER_MOVE,
    SANTORINI_NEURAL_TRAINING_GAMES_COUNT,
    SANTORINI_NEURAL_WATCH_CHECKPOINTS_COUNT,
    SANTORINI_NEURAL_WATCH_EPOCHS_PER_CHECKPOINT,
    SANTORINI_NEURAL_WATCH_EVALUATION_GAMES_COUNT,
)
from app.games.santorini.evaluation_summary import (
    format_o_evaluation_comparison,
    summarize_o_results,
)
from app.ai.neural_network import SimpleNeuralNetwork
from app.games.santorini.paired_evaluation import evaluate_santorini_neural_vs_random_paired
from app.games.santorini.neural_training import (
    build_train_and_save_santorini_neural_model,
    format_santorini_training_summary,
    get_santorini_model_data,
    load_santorini_model_package,
)
from app.games.santorini.neural_watch import (
    format_santorini_watch_table,
    train_santorini_neural_with_watch,
)


def run_train_santorini_neural_command():
    if _has_flag("--watch"):
        run_train_santorini_neural_watch_command()
        return

    options = read_santorini_neural_training_options()
    print("Entraînement neuronal Santorini")
    print("Modèle séparé : " + str(SANTORINI_NEURAL_MODEL_FILE))
    print("Dataset sauvegardé : " + str(SANTORINI_DATASET_FILE))
    print("Parties simulées :", options["games_count"])
    print("Exemples max :", options["max_examples"])
    print("Simulations par coup :", options["simulations_per_move"])
    print("Couche cachée :", options["hidden_size"])
    print("Époques :", options["epochs"])
    print("Taux d'apprentissage :", options["learning_rate"])
    print()

    package = build_train_and_save_santorini_neural_model(
        model_file=SANTORINI_NEURAL_MODEL_FILE,
        dataset_file=SANTORINI_DATASET_FILE,
        games_count=options["games_count"],
        max_examples=options["max_examples"],
        simulations_per_move=options["simulations_per_move"],
        hidden_size=options["hidden_size"],
        epochs=options["epochs"],
        learning_rate=options["learning_rate"],
        seed=SANTORINI_DATASET_SEED,
        show_progress=True,
    )

    print()
    print(format_santorini_training_summary(package["training_summary"]))
    print()
    print("Modèle Santorini sauvegardé dans : " + str(SANTORINI_NEURAL_MODEL_FILE))


def run_train_santorini_neural_watch_command():
    options = read_santorini_neural_training_options()
    print("Entraînement surveillé neuronal Santorini")
    print("Modèle séparé : " + str(SANTORINI_NEURAL_MODEL_FILE))
    print("Dataset sauvegardé : " + str(SANTORINI_DATASET_FILE))
    print("Parties simulées :", options["games_count"])
    print("Exemples max :", options["max_examples"])
    print("Simulations par coup :", options["simulations_per_move"])
    print("Couche cachée :", options["hidden_size"])
    print("Paliers :", SANTORINI_NEURAL_WATCH_CHECKPOINTS_COUNT)
    print("Époques par palier :", SANTORINI_NEURAL_WATCH_EPOCHS_PER_CHECKPOINT)
    print("Parties d'évaluation par palier :", SANTORINI_NEURAL_WATCH_EVALUATION_GAMES_COUNT)
    print("Comparaison appairée : mêmes placements de départ pour modèle et random")
    print("Taux d'apprentissage :", options["learning_rate"])
    print()

    package = train_santorini_neural_with_watch(
        model_file=SANTORINI_NEURAL_MODEL_FILE,
        dataset_file=SANTORINI_DATASET_FILE,
        games_count=options["games_count"],
        max_examples=options["max_examples"],
        simulations_per_move=options["simulations_per_move"],
        hidden_size=options["hidden_size"],
        checkpoints_count=SANTORINI_NEURAL_WATCH_CHECKPOINTS_COUNT,
        epochs_per_checkpoint=SANTORINI_NEURAL_WATCH_EPOCHS_PER_CHECKPOINT,
        learning_rate=options["learning_rate"],
        evaluation_games_count=SANTORINI_NEURAL_WATCH_EVALUATION_GAMES_COUNT,
        seed=SANTORINI_DATASET_SEED,
        show_progress=True,
    )

    print()
    print(format_santorini_watch_table(package))
    print()
    print(format_santorini_training_summary(package["training_summary"]))
    print("Meilleur palier : " + str(package["training_summary"]["best_checkpoint"]))
    print("Écart meilleur modèle - random : " + str(round(package["training_summary"]["best_delta_vs_random"], 2)) + " points")
    print()
    print("Modèle Santorini sauvegardé dans : " + str(SANTORINI_NEURAL_MODEL_FILE))


def run_evaluate_santorini_neural_command():
    games_count = _read_int_arg(2, SANTORINI_NEURAL_EVALUATION_GAMES_COUNT)
    package = load_santorini_model_package(SANTORINI_NEURAL_MODEL_FILE)
    model_data = get_santorini_model_data(package)

    print("Évaluation du modèle neuronal Santorini")
    print("Fichier : " + str(SANTORINI_NEURAL_MODEL_FILE))
    print("Parties d'évaluation : " + str(games_count))
    print("Comparaison appairée : mêmes placements de départ pour modèle et random")
    print()

    if not model_data:
        print("Aucun modèle Santorini valide trouvé.")
        return

    network = SimpleNeuralNetwork.from_dict(model_data)
    paired_results = evaluate_santorini_neural_vs_random_paired(
        network=network,
        games_count=games_count,
        seed=SANTORINI_DATASET_SEED,
    )
    neural_results = paired_results["neural_results"]
    baseline_results = paired_results["baseline_results"]
    neural_summary = summarize_o_results(neural_results)
    baseline_summary = summarize_o_results(baseline_results)
    print(format_o_evaluation_comparison(neural_summary, baseline_summary))


def read_santorini_neural_training_options():
    args = _positional_args_without_flags()
    return {
        "games_count": _read_int_from_args(args, 0, SANTORINI_NEURAL_TRAINING_GAMES_COUNT),
        "max_examples": _read_int_from_args(args, 1, SANTORINI_NEURAL_MAX_EXAMPLES),
        "simulations_per_move": _read_int_from_args(args, 2, SANTORINI_NEURAL_SIMULATIONS_PER_MOVE),
        "hidden_size": _read_int_from_args(args, 3, SANTORINI_NEURAL_HIDDEN_SIZE),
        "epochs": _read_int_from_args(args, 4, SANTORINI_NEURAL_EPOCHS),
        "learning_rate": _read_float_from_args(args, 5, SANTORINI_NEURAL_LEARNING_RATE),
    }


def _has_flag(flag):
    return flag in sys.argv[2:]


def _positional_args_without_flags():
    return [arg for arg in sys.argv[2:] if not arg.startswith("--")]


def _read_int_from_args(args, index, default_value):
    if len(args) <= index:
        return default_value

    try:
        value = int(args[index])
    except ValueError:
        return default_value

    return max(1, value)


def _read_float_from_args(args, index, default_value):
    if len(args) <= index:
        return default_value

    try:
        value = float(args[index])
    except ValueError:
        return default_value

    if value <= 0:
        return default_value

    return value


def _read_int_arg(position, default_value):
    if len(sys.argv) <= position:
        return default_value

    try:
        value = int(sys.argv[position])
    except ValueError:
        return default_value

    return max(1, value)


def _read_float_arg(position, default_value):
    if len(sys.argv) <= position:
        return default_value

    try:
        value = float(sys.argv[position])
    except ValueError:
        return default_value

    if value <= 0:
        return default_value

    return value
